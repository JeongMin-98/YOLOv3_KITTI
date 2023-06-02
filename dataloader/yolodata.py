import torch
from torch.utils.data import Dataset
import os, sys
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.tools import *

class YoloData(Dataset):
    annotation_dir = ""
    file_dir = ""
    file_txt = ""
    train_dir = "D:\\KITTI\\training"
    train_txt = "train.txt"
    valid_dir = "D:\\KITTI\\eval"
    valid_txt = "eval.txt"
    class_str = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    num_class = None
    img_data = []

    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(YoloData, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['class']
        if self.is_train:
            self.file_dir = self.train_dir + "\\JPEGImages\\"
            self.file_txt = self.train_dir + "\\ImageSets\\" + self.train_txt
            self.anno_dir = self.train_dir + "\\Annotations\\"
        else:
            self.file_dir = self.valid_dir + "\\JPEGImages\\"
            self.file_txt = self.valid_dir + "\\ImageSets\\" + self.valid_txt
            self.anno_dir = self.valid_dir + "\\Annotations\\"

        # declare image_names and image_data
        img_names = []
        img_data = []

        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [i.replace('\n', "") for i in f.readlines()]

        for img_name in img_names:
            if os.path.exists(self.file_dir + img_name + ".jpg"):
                img_data.append(img_name + ".jpg")
            elif os.path.exists(self.file_dir + img_name + ".JPG"):
                img_data.append(img_name + ".JPG")
            elif os.path.exists(self.file_dir + img_name + ".png"):
                img_data.append(img_name + ".png")
            elif os.path.exists(self.file_dir + img_name + ".PNG"):
                img_data.append(img_name + ".PNG")
        self.img_data = img_data

    def __getitem__(self, index):

        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            # img_origin_height, img_origin_weight = img.shape[:2]

        if os.path.isdir(self.anno_dir):
            txt_name = self.img_data[index]
            for ext in ['.png', '.PNG', '.jpg', 'JPG']:
                txt_name = txt_name.replace(ext, ".txt")
            anno_path = self.anno_dir + txt_name

            if not os.path.exists(anno_path):
                return

            bbox = []
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    line = line.replace("\n", "")
                    get_data = [l for l in line.split(" ")]
                    if len(get_data) < 5:
                        continue
                    class_label = float(get_data[0])
                    center_x = float(get_data[1])
                    center_y = float(get_data[2])
                    weight = float(get_data[3])
                    height = float(get_data[4])

                    bbox.append([class_label, center_x, center_y, weight, height])

            bbox = np.array(bbox)

            empty_target = False
            if bbox.shape[0] == 0:
                empty_target = True
                bbox = np.array([[0, 0, 0, 0, 0]])

            # data transform
            if self.transform is not None:
                img, bbox = self.transform((img, bbox))

            if not empty_target:
                batch_idx = torch.zeros((bbox.shape[0]))
                target_data = torch.cat((batch_idx.view(-1, 1), torch.tensor(bbox)), dim=1)
            else:
                return

            return img, target_data, anno_path
        else:
            bbox = np.array([[0, 0, 0, 0, 0]])
            if self.transform is not None:
                img, _ = self.transform((img, bbox))
            return img, None, None

    def __len__(self):
        return len(self.img_data)
