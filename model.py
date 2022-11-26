import numpy as np
import torch
import torchvision
from numba import jit, cuda
import os
import cv2
from numpy import asarray
from torchvision.transforms import transforms
from PIL import Image
import glob
import random


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=15, kernel_size=6, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=15, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #self.linear_1 = torch.nn.Linear(7 * 7 * 20, 128)
        self.linear_1 = torch.nn.Linear(7 * 7 * 30, 11)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv_1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)

        x = self.max_pool2d(x)
        #print(x.shape)

        x = self.conv_2(x)
        #print(x.shape)

        x = self.relu(x)
        x = self.max_pool2d(x)
        #print(x.shape)

        x = x.reshape(x.size(0), -1)
        #x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear_1(x)

        return pred

    # def load_images(path):
    #     training_data = []
    #     images = glob.glob(path + "/*.png")
    #     for image in images:
    #         with open(image, 'rb') as file:
    #             img = Image.open(file)
    #             print(type(img))
    #             image2 = Image.fromarray(img)
    #
    #             print(type(image2))
    #             training_data.append([image2])
    #     return training_data

    def load_images(path):
        images = []
        num = 1
        base = path + "/%d.jpg"
        lower = np.array([0, 0, 0], dtype = "uint8")
        upper = np.array([38, 255, 255], dtype = "uint8")
        while os.path.isfile(base % num):
          print(num)
          img=cv2.imread (base % num, 1)
          blur = cv2.GaussianBlur(img, (3,3), 0)
          hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
          skinMask = cv2.inRange(hsv, lower, upper)
          # blur the mask to help remove noise, then apply the
          # mask to the frame
          skin = cv2.bitwise_and(img, img, mask = skinMask)
          #immage=cv2.bitwise_and(min_sat,max_hue)
          image = Image.fromarray(np.asarray(skin))
          image = image.convert('RGB')
          image = image.resize((32, 32))
          images.append(transforms.Compose([transforms.ToTensor()])(image))
          # images.append(np.asarray(Image.open(f)).convert("RGBA"))
          num += 1
        return images

    def convert_rgb_to_rgba(self, row):
        result = []
        """Convert an RGB image to RGBA. This method assumes the
        alpha channel in result is already correctly initialized.
        """
        for i in range(3):
            result[i::4] = row[i::3]
        return result


    def load_labels(path):
        labels = []
        with open(path, 'r') as f:
            for line in f.readlines():
       #         labels.append(transforms.Compose([transforms.ToTensor()])(int(line)))
                labels.append(int(line))
        return labels

    def load_labels_to_images(data, labels):
        print(len(data))
        print(len(labels))
        set = [[0] * 2 for i in range(len(labels))]
        for i in range(len(labels)):
            set[i] = [data[i], labels[i]]
        return set

    def load_labels_to_images_with_augmentation(data, labels):
        my_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor()])
        img_num = 0
        
        print(len(data))
        print(len(labels))
        set = [[0] * 2 for i in range(10*len(labels))]
        for i in range(len(labels)):
          for j in range (10):
            set[i*10+j] = [my_transforms(data[i]), labels[i]]
        return set
    