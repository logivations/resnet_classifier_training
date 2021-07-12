import argparse
import os
from PIL import Image
import shutil
import torch
import time

from train import init_resnet, get_transforms


@torch.no_grad()
def infer_images(model, input_images):
    """Runs inference on a list of read image names moving the image to the
     output folder corresponding to it's predicted class. For example, if the
     predicted class is "1" the image will be moved from "inference/input" to
    "inference/output/1".

    :param model: Trained prediction model.
    :param input_images: List of image names.
    """

    while input_images:
        image_name = input_images.pop()
        source_path = os.path.join(input_dir, image_name)
        image = Image.open(source_path)
        image = transforms(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        output = model(image)
        prediction = torch.round(torch.sigmoid(output))
        predicted_class = class_map[int(prediction)]
        target_path = os.path.join(output_dir, predicted_class, image_name)
        shutil.move(source_path, target_path)
        print("{} ---> {}".format(image_name, predicted_class))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="inference/input",
                        help="Interactive inference input folder")
    parser.add_argument("--output_dir", type=str, default="inference/output",
                        help="Interactive inference output folder")
    parser.add_argument("--load_path", type=str,
                        default="checkpoints/epoch_100.model",
                        help="Saved model checkpoint")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    allowed_types = [".png", ".jpg", ".jpeg"]
    class_map = {0: "1", 1: "2"}
    running = True

    model = init_resnet(out_features=1)
    model.load_state_dict(torch.load(args.load_path))
    model.eval()
    transforms = get_transforms(train="False")
    print("Interactive session started, put images in {}".format(input_dir))
    while running:
        input_images = [image_path for image_path in os.listdir(input_dir)
                        if any([image_path.endswith(t) for t in allowed_types])]
        if input_images:
            infer_images(model, input_images)
            time.sleep(0.1)
