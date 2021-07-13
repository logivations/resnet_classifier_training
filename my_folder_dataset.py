import torch.utils.data as data
import cv2
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class MyFolderDataset(data.Dataset):
    def __init__(self, root_path, transforms, label_dict, balanced=True):
        self.root_path = root_path
        self.transforms = transforms
        self.label_dict = label_dict
        self.image_paths = []
        self.labels = []
        if balanced:
            self.init_balanced_names()
        else:
            self.init_names()


    def init_names(self):
        self.classes = os.listdir(self.root_path)
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            for filename in [f for f in filenames if f.endswith(".jpg")]:
                path = os.path.join(dirpath, filename)
                self.image_paths.append(os.path.join(dirpath, filename))
                self.labels.append(path.split("/")[-3])

    def init_balanced_names(self):
        self.classes = os.listdir(self.root_path)
        slots = sorted(os.listdir(os.path.join(self.root_path, self.classes[0])))
        for s in slots:
            class_occurences = []
            image_names = []
            for c in self.classes:
                class_dir = os.path.join(self.root_path, c, s)
                if os.path.exists(class_dir):
                    class_im_names = os.listdir(class_dir)
                    class_occurences.append(len(class_im_names))
                    image_names.append(class_im_names)
                else:
                    class_occurences.append(0)
            longest = max(class_occurences)
            for c, imn, occ in zip(self.classes, image_names, class_occurences):
                if occ > 0:
                    for j in range(longest // occ):
                        self.image_paths.extend([os.path.join(self.root_path, c, s, imname) for imname in imn])
                        self.labels.extend([c] * len(imn))


    def __getitem__(self, item):
        label = self.label_dict[self.labels[item]]
        path = self.image_paths[item]
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)