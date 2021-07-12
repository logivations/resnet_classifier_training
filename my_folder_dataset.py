import torch.utils.data as data
import cv2
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class MyFolderDataset(data.Dataset):
    def __init__(self, root_path, transforms, label_dict):
        self.root_path = root_path
        self.transforms = transforms
        self.label_dict = label_dict
        self.image_paths = []
        self.labels = []
        self.init_names()


    def init_names(self):
        self.classes = os.listdir(self.root_path)
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            for filename in [f for f in filenames if f.endswith(".jpg")]:
                path = os.path.join(dirpath, filename)
                self.image_paths.append(os.path.join(dirpath, filename))
                self.labels.append(path.split("/")[-3])


    def __getitem__(self, item):
        label = self.label_dict[self.labels[item]]
        path = self.image_paths[item]
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)