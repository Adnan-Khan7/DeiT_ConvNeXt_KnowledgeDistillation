# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from PIL import Image
import torch
import pandas as pd
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torchvision
import scipy.io
from PIL import Image
from torch.utils.data import ConcatDataset


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == "CUB":
        print("reading from datapath", args.data_path)
        root = args.data_path
        if is_train:
            dataset = CUBDataset(image_root_path=root, transform=transform, split="train")
        else:
            dataset = CUBDataset(image_root_path=root, transform=transform, split="test")

        nb_classes = 200
        assert len(dataset.class_to_idx) == nb_classes

    elif args.data_set == "CUB_DOG":
        # This time we will have 2 datasets
        root1 = args.data_path.split(' ')[0]
        root2 = args.data_path.split(' ')[1]
        print("reading from datapath", root1)
        print("reading from datapath", root2)
        if is_train:
            dataset1 = CUBDataset(image_root_path=root1, transform=transform, split="train")
            dataset2 = DOGDataset(image_root_path=root2, transform=transform, split="train")
            dataset = ConcatDataset([dataset1, dataset2])
        else:
            dataset1 = CUBDataset(image_root_path=root1, transform=transform, split="test")
            dataset2 = DOGDataset(image_root_path=root2, transform=transform, split="test")
            dataset = ConcatDataset([dataset1, dataset2])

        nb_classes = 320
        # assert len(dataset.class_to_idx) == nb_classes

    elif args.data_set == "FOOD":

        if is_train:
            train_df = pd.read_csv(f'{args.data_path}/annot/train_info.csv', names=['image_name', 'label'])
            train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{args.data_path}/train_set/', x))
            dataset = FOODDataset(train_df, transform=transform)
        else:
            val_df = pd.read_csv(f'{args.data_path}/annot/val_info.csv', names=['image_name', 'label'])
            val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{args.data_path}/val_set/', x))
            dataset = FOODDataset(val_df, transform=transform)
        nb_classes = 251
        # assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / testz
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content


class DOGDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for DOG Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0] for o in image_info]

        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0] for o in split_info]
        self.split_info = {}
        if split == 'train':
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split == 'test':
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"

        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

        ## modify class index as we are going to concat to first dataset
        self.class_to_idx = {class_: idx + 200 for idx, class_ in enumerate(self.class_to_idx)}

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(os.path.join(path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        ## modify target class index as we are going to concat to first dataset
        return img, target + 200

    @staticmethod
    def get_file_content(file_path):
        content = scipy.io.loadmat(file_path)
        return content['file_list']


class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.data_transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.data_transform(Image.open(row["path"])), row['label']
        )
