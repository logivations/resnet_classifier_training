import argparse
import random
random.seed(0)
import torch
import torchvision
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from my_folder_dataset import MyFolderDataset
from augmentations import get_augmentations

def train(num_epochs, train_path, val_path):
    """Trains the model for the given number of epochs.

    :param model: Model instance.
    :param num_epochs: Number of epochs that the model is being trained for.
    :return: Trained Model.
    """
    model = init_resnet(out_features=1)
    optimizer = Adam(model.parameters(), lr=args.lr)
    train_dataset = MyFolderDataset(train_path, train_transforms, label_dict=label_dict, balanced=True)
    val_dataset = MyFolderDataset(val_path, val_transforms, label_dict=label_dict, balanced=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    running_acc = None
    train_logger = Metrics_Logger("Train")
    for epoch in range(num_epochs):
        if num_epochs - epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
                print("lr changed to", param_group['lr'])
        for images, labels in train_loader:
            if not images.shape[0] == batch_size:
                continue
            model.train()
            images = images.cuda()
            labels = labels.float().cuda()
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.reshape(outputs.shape[0])
            classification_loss = loss_fn(outputs, labels)
            num_correct = count_correct_predictions(outputs, labels)
            train_logger.update(classification_loss,
                                num_correct,
                                outputs.shape[0])
            loss = torch.sum(classification_loss)
            loss.backward()
            optimizer.step()
        # Write a summary to stdout
        print("")
        print("Epoch  " + str(epoch + 1))
        train_logger.write_summary()
        if not args.no_validation:
            val_logger = validate(model, val_loader)
            acc = val_logger.write_summary()
            if running_acc is None:
                running_acc = acc
            else:
                running_acc = 0.8 * running_acc + 0.2 * acc
        train_logger.reset()
        if (epoch + 1) % model_save_epoch == 0:
            save_model(epoch, model)
    print("Training finished.")
    return running_acc


@torch.no_grad()
def validate(model, val_loader):
    """Full validation cycle. Calculates classification loss and accuracy on
    the validation data.

    :param model: Trained model.
    :return: Validation metrics in form of a Metrics_Logger instance.
    """
    model.eval()
    val_logger = Metrics_Logger("Validation")
    for i, sample in enumerate(val_loader):
        images, labels = sample
        images = images.cuda()
        labels = labels.float().cuda()
        outputs = model(images)
        outputs = outputs.reshape(outputs.shape[0])
        loss = loss_fn(outputs, labels)
        num_correct = count_correct_predictions(outputs, labels)
        val_logger.update(loss, num_correct, outputs.shape[0])
    return val_logger


def init_resnet(out_features, pretrained=True):
    """Load a pretrained resnet model, reinitializes the last layer to match
    the given number of outputs, copies it to gpu and freezes some parameters.

    :param out_features: Number of output features for the reinitialized layer.
    :return: Resnet instance.
    """

    model = torchvision.models.resnet18(pretrained=pretrained)
    model._modules["fc"] = torch.nn.Linear(
        in_features=model._modules["fc"].in_features, out_features=out_features)
    model.cuda()
    model = freeze_resnet(model)
    return model


def freeze_resnet(model):
    """Freezes some of the models parameters permanently. Keeping early layers
    of pretrained models frozen can prevent overfitting and thus increase the
    performance.

    Resnet block names are: [conv1, layer1, layer2, layer3, layer4, avgpool, fc]

    :param model: A Resnet instance.
    :return: Resnet instance with partially frozen parameters.
    """

    frozen_layers = ["conv1", "layer1"]
    for layer_name in frozen_layers:
        for param in model._modules[layer_name].parameters():
            param.requires_grad = False
    return model


def save_model(epoch, model):
    """Write the model parameters to a file.

    :param epoch: Epoch number.
    :param model: Pytorch model instance.
    """

    state_dict = model.state_dict()
    if not os.path.exists(os.getcwd() + "/checkpoints"):
        os.makedirs(os.getcwd() + "/checkpoints")
    torch.save(state_dict, "checkpoints/epoch_{}.model".format(epoch + 1))
    print("Checkpoint saved")




def count_correct_predictions(outputs, labels):
    """Count the number of correctly predicted labels.

    :param outputs: Model outputs (logits).
    :param labels: Class annotations, matching the output shape.
    :return: Number of correct predictions, over all outputs.
    """

    predictions = torch.round(torch.sigmoid(outputs))
    num_correct = torch.sum(predictions == labels).squeeze().double()
    #print("samples", outputs.shape[0])
    #print("num_cor", num_correct)
    return num_correct


class Metrics_Logger():
    def __init__(self, prefix):
        self.prefix = prefix
        self.loss = 0.
        self.total_correct = 0.
        self.num_logged = 0.00000001
    @torch.no_grad()
    def update(self, loss, num_correct, num_logged):
        """Updates the stored metrics for the logged batch.

        :param loss: Classification loss.
        :param num_correct: Number of correct classifications.
        :param num_logged: Number of samples logged. (Size of the batch).
        """

        self.num_logged += num_logged
        self.loss += loss
        self.total_correct += num_correct

    def reset(self):
        """Resets the stored metrics.
        """

        self.loss = 0.
        self.total_correct = 0.
        self.num_logged = 0.00000001

    def write_summary(self):
        """Calculates human interpretable metrics and prints them.
        """

        print("{}\n Loss: {:.8f}, Accuracy: {:.3f}".format(
            self.prefix,
            self.loss / self.num_logged,
            self.total_correct / self.num_logged),
        )
        return self.total_correct / self.num_logged


if __name__ == "__main__":
    FOLDS = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20,
        help="Total Number of Epochs")
    parser.add_argument("--save-freq", type=int, default=20,
        help="Number of train epochs between saving the model file")
    parser.add_argument("--bs", type=int, default=48, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-validation", dest="no_validation",
        action="store_true", help="Don't do validation cycles during training")
    parser.add_argument("--train_path", type=str, default="/data/FST/demo/dataset/split/train")
    parser.add_argument("--val_path", type=str, default="/data/FST/demo/dataset/split/val")
    n_folds = 5
    label_dict = {"box": 0, "nobox": 1}
    args = parser.parse_args()
    batch_size = args.bs
    max_epochs = args.epochs
    model_save_epoch = args.save_freq
    print("Training started with settings: {}".format(args))
    results =  []
    train_transforms, val_transforms = get_augmentations()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    res = train(args.epochs, args.train_path, args.val_path)
    print("Training finished. \n Accuracy: {}".format(res))
