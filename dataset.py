import os
from PIL import Image, ImageEnhance
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import numbers
import random

from preproc import cv_random_flip, random_crop, random_rotate, color_enhance, random_gaussian, random_pepper
from config import Config


class CoData(data.Dataset):
    def __init__(self, image_root, label_root, image_size, max_num, is_train):

        class_list = os.listdir(image_root)
        self.size_train = image_size
        self.size_test = image_size
        self.data_size = (self.size_train, self.size_train) if is_train else (self.size_test, self.size_test)
        self.image_dirs = list(map(lambda x: os.path.join(image_root, x), class_list))
        self.label_dirs = list(map(lambda x: os.path.join(label_root, x), class_list))
        self.max_num = max_num
        self.is_train = is_train
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.load_all = False

    def __getitem__(self, item):
        names = os.listdir(self.image_dirs[item])
        num = len(names)
        image_paths_0 = list(map(lambda x: os.path.join(self.image_dirs[item], x), names))
        label_paths_0 = list(map(lambda x: os.path.join(self.label_dirs[item], x[:-4]+'.png'), names))
        # path2image, path2label = {}, {}
        # for image_path, label_path in zip(image_paths, label_paths):
        #     path2image[image_path] = Image.open(image_path).convert('RGB')
        #     path2label[label_path] = Image.open(label_path).convert('L')
        other_cls_ls = list(range(len(self.image_dirs)))
        other_cls_ls.remove(item)
        other_items = random.sample(set(other_cls_ls), Config().loadN)
        other_nums = []
        other_image_paths_list, other_label_paths_list = [image_paths_0], [label_paths_0]

        if self.is_train:
            if Config().loadN > 1:
                # random pick one category

                for other_item in other_items:
                    other_names = os.listdir(self.image_dirs[other_item])
                    other_num = len(other_names)
                    other_image_paths = list(map(lambda x: os.path.join(self.image_dirs[other_item], x), other_names))
                    other_label_paths = list(map(lambda x: os.path.join(self.label_dirs[other_item], x[:-4]+'.png'), other_names))

                    other_nums.append(other_num)
                    other_image_paths_list.append(other_image_paths)
                    other_label_paths_list.append(other_label_paths)

                if Config().auto_pad:
                    if Config().auto_pad == 'fixed':
                        final_num = self.max_num
                    else:
                        final_num = min(max(num, *other_nums), self.max_num)
                else:
                    final_num = min(num, *other_nums, self.max_num)

                image_paths, label_paths = [], []
                for idx_sel, (selected_image_paths, selected_label_paths) in enumerate(zip(other_image_paths_list, other_label_paths_list)):
                    if Config().auto_pad:
                        dup_times = final_num // len(selected_image_paths) + 1
                        sampled_list = random.sample(range(len(selected_image_paths) * dup_times), final_num)
                        image_paths += [(selected_image_paths * dup_times)[i] for i in sampled_list]
                        label_paths += [(selected_label_paths * dup_times)[i] for i in sampled_list]
                    else:
                        sampled_list = random.sample(range(len(selected_image_paths)), final_num)
                        image_paths += [selected_image_paths[i] for i in sampled_list]
                        label_paths += [selected_label_paths[i] for i in sampled_list]

                final_num = final_num * Config().loadN
            else:
                final_num = min(num, self.max_num)
                sampled_list = random.sample(range(len(image_paths_0)), final_num)
                image_paths = [image_paths_0[i] for i in sampled_list]
                label_paths = [label_paths_0[i] for i in sampled_list]
        else:
            final_num = num
            image_paths = image_paths_0
            label_paths = label_paths_0

        images = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])
        labels = torch.Tensor(final_num, 1, self.data_size[1], self.data_size[0])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):
            if self.load_all:
                # TODO
                image = self.images_loaded[idx]
                label = self.labels_loaded[idx]
            else:
                if not os.path.exists(image_paths[idx]):
                    image_paths[idx] = image_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else image_paths[idx].replace('.png', '.jpg')
                image = Image.open(image_paths[idx]).convert('RGB')
                if not os.path.exists(label_paths[idx]):
                    label_paths[idx] = label_paths[idx].replace('.jpg', '.png') if label_paths[idx][-4:] == '.jpg' else label_paths[idx].replace('.png', '.jpg')
                label = Image.open(label_paths[idx]).convert('L')

            subpaths.append(os.path.join(image_paths[idx].split('/')[-2], image_paths[idx].split('/')[-1][:-4]+'.png'))
            ori_sizes.append((image.size[1], image.size[0]))

            # loading image and label
            if self.is_train:
                if 'flip' in Config().preproc_methods:
                    image, label = cv_random_flip(image, label)
                if 'crop' in Config().preproc_methods:
                    image, label = random_crop(image, label)
                if 'rotate' in Config().preproc_methods:
                    image, label = random_rotate(image, label)
                if 'enhance' in Config().preproc_methods:
                    image = color_enhance(image)
                if 'pepper' in Config().preproc_methods:
                    label = random_pepper(label)

            image, label = self.transform_image(image), self.transform_label(label)

            images[idx] = image
            labels[idx] = label

        if self.is_train:
            if Config().loadN > 1:
                cls_ls = [item] * (final_num // Config().loadN)
                for other_item in other_items:
                    cls_ls += [other_item] * (final_num // Config().loadN)
                return images, labels, subpaths, ori_sizes, cls_ls
            else:
                cls_ls = [item] * final_num
                return images, labels, subpaths, ori_sizes, cls_ls
        else:
            return images, labels, subpaths, ori_sizes

    def __len__(self):
        return len(self.image_dirs)


def get_loader(img_root, gt_root, img_size, batch_size, max_num = float('inf'), istrain=True, shuffle=False, num_workers=0, pin=False):
    dataset = CoData(img_root, gt_root, img_size, max_num, is_train=istrain)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader
