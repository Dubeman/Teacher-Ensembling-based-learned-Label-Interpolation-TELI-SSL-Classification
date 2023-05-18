import logging
import math
import os.path
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.io import read_image
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode, TrivialAugmentWide
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from augmentation import RandAugmentCIFAR, RandAugmentSVHN, RandAugmentTinyImagenet
import imageio
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
logger = logging.getLogger(__name__)

cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
cifar100_mean = (0.507075, 0.486549, 0.440918)
cifar100_std = (0.267334, 0.256438, 0.276151)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_Fashion_MNIST(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.FashionMNIST(args.data_path, train=True, download=True)
    
    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    

    train_labeled_dataset = FashionMNIST_SSL(
        args.data_path, train_labeled_idxs,
        transform= transforms.Compose([transforms.ToTensor()])
    )
    finetune_dataset = FashionMNIST_SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_unlabeled_dataset = FashionMNIST_SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_dataset = datasets.FashionMNIST(args.data_path, train=False,
                                    transform=transforms.Compose([transforms.ToTensor()]), download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset







def get_MNIST(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.MNIST(args.data_path, train=True, download=True)
    
    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    

    train_labeled_dataset = MNIST_SSL(
        args.data_path, train_labeled_idxs,
        transform= transforms.Compose([transforms.ToTensor()])
    )
    finetune_dataset = MNIST_SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_unlabeled_dataset = MNIST_SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_dataset = datasets.MNIST(args.data_path, train=False,
                                    transform=transforms.Compose([transforms.ToTensor()]), download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset





def get_cifar10(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)
    
    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
    finetune_dataset = CIFAR10SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transform_finetune
    )
    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL_cifar(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset

def get_tinyimagenet(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentTinyImagenet(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    root = os.getcwd()
    train_dataset = ImageFolder(root= os.path.join(root, "tiny-imagenet-200/train"))
    valid_dataset = ImageFolder(root= os.path.join(root, "tiny-imagenet-200/val/images")) 
    test_dataset = ImageFolder(root= os.path.join(root, "tiny-imagenet-200/test")) 
    tinyimagenet_targets = [labels for images,labels in train_dataset]
    labels_dataframe = pd.DataFrame(tinyimagenet_targets)
    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, tinyimagenet_targets)
    
    train_labeled_dataset =  TinyImagenetSSL(root = "./tiny-imagenet-200",indexs = train_labeled_idxs,
        annotations_dataframe = labels_dataframe, train_dataset=train_dataset, transform=transform_labeled, target_transform=None
    )
    finetune_dataset =  TinyImagenetSSL(root = "./tiny-imagenet-200",indexs = finetune_idxs,
        annotations_dataframe = labels_dataframe, img_dir = os.path.join(root,"train"), transform=transform_finetune, target_transform=None
    )
    train_unlabeled_dataset = TinyImagenetSSL(root = "./tiny-imagenet-200",indexs = train_unlabeled_idxs,
         annotations_dataframe = labels_dataframe, img_dir = os.path.join(root,"train"), transform=transform_labeled, target_transform=None
    )
    
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset



def get_SVHN(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentSVHN(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    #base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)
    base_dataset = datasets.SVHN(args.data_path, split = 'train',transform = transforms.ToTensor(), download=True) 
    svhn_targets = [labels for images,labels in base_dataset]
    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, svhn_targets)

    train_labeled_dataset = SVHNSSL(
        args.data_path, train_labeled_idxs, split = 'train',
        transform=transform_labeled,targets = svhn_targets
    )
    finetune_dataset = SVHNSSL(
        args.data_path, train_labeled_idxs, split = 'train',
        transform=transform_labeled,targets = svhn_targets
    )
    train_unlabeled_dataset = SVHNSSL(
        args.data_path, train_unlabeled_idxs,
        split = 'train',targets = svhn_targets,
        transform=TransformMPL_SVHN(args, mean=cifar10_mean, std=cifar10_std) #MEAN AND STD TO BE CHANGED
    )

    test_dataset = datasets.SVHN(args.data_path, split = 'test',
                                    transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset

def get_cifar100(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
  
     
    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
    finetune_dataset = CIFAR100SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transform_finetune
    )
    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL_cifar(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx_ex = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx_ex)
        np.random.shuffle(labeled_idx)
        return labeled_idx_ex, unlabeled_idx, labeled_idx
    else:
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx, labeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx

class TransformMPL_MNIST(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugmentMNIST(n=n, m=m)])
        
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        
       
        return self.normalize(ori), self.normalize(aug)



class TransformMPL_SVHN(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugmentSVHN(n=n, m=m)])
        
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        
       
        return self.normalize(ori), self.normalize(aug)


class TransformMPL_cifar(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugmentCIFAR(n=n, m=m)])
        
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        
       
        return self.normalize(ori), self.normalize(aug)

class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split,targets,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split = split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(targets)[indexs]           #np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST_SSL(datasets.FashionMNIST):
  def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

  def __getitem__(self, index):
      img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
      img = Image.fromarray(img.numpy(), mode="L")
      #img = Image.fromarray(img)

      if self.transform is not None:
            img = self.transform(img)

      if self.target_transform is not None:
            target = self.target_transform(target)

      return img, target



class MNIST_SSL(datasets.MNIST):
  def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

  def __getitem__(self, index):
      img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
      img = Image.fromarray(img.numpy(), mode="L")
      #img = Image.fromarray(img)

      if self.transform is not None:
            img = self.transform(img)

      if self.target_transform is not None:
            target = self.target_transform(target)

      return img, target


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TinyImagenetSSL(Dataset):
  def __init__(self,root,indexs, annotations_dataframe, train_dataset, transform=None, target_transform=None):
        self.img_labels = annotations_dataframe
        self.data = self.data[indexs]
        self.transform = transform
        self.target_transform = target_transform

  def __len__(self):
        return len(self.img_labels)

  def __getitem__(self, idx):
        
        img, target = self.train_dataset[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def restructure_val_TinyImagenet(data_dir):
  print("---preparing Val set of Tinyimagenet")
  val_dir = os.path.join(data_dir, 'val')
  val_data = pd.read_csv(f'{val_dir}/val_annotations.txt', 
                       sep='\t', 
                       header=None, 
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])
  # Create separate validation subfolders for the validation images based on
  # their labels indicated in the val_annotations txt file
  val_img_dir = os.path.join(val_dir, 'images')

  # Open and read val annotations text file
  fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
  data = fp.readlines()

  # Create dictionary to store img filename (word 0) and corresponding
  # label (word 1) for every line in the txt file (as key value pair)
  val_img_dict = {}
  for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
  fp.close()                     


  # Create subfolders (if not present) for validation images based on label ,
  # and move images into the respective folders
  for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
  
  # Save class names (for corresponding labels) as dict from words.txt file
  class_to_name_dict = dict()
  fp = open(os.path.join(data_dir, 'words.txt'), 'r')
  data = fp.readlines()
  for line in data:
    words = line.strip('\n').split('\t')
    class_to_name_dict[words[0]] = words[1].split(',')[0]
  fp.close()

# Display first 20 entries of resulting dictionary
  {k: class_to_name_dict[k] for k in list(class_to_name_dict)[:20]}
  print("--preparation complete--")



dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

def download_and_unzip(URL, root_dir):
  error_message = "Download is not yet implemented. Please, go to {URL} urself."
  raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""
class TinyImageNetPaths:
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))

"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""
class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
               transform=None, download=False, max_samples=None):
    tinp = TinyImageNetPaths(root_dir, download)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

    if self.preload:
      load_desc = "Preloading {} data...".format(mode)
      self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
      self.label_data = np.zeros((self.samples_num,), dtype=np.int)
      for idx in tqdm(range(self.samples_num), desc=load_desc):
        s = self.samples[idx]
        img = imageio.imread(s[0])
        img = _add_channels(img)
        self.img_data[idx] = img
        if mode != 'test':
          self.label_data[idx] = s[self.label_idx]

      if load_transform:
        for lt in load_transform:
          result = lt(self.img_data, self.label_data)
          self.img_data, self.label_data = result[:2]
          if len(result) > 2:
            self.transform_results.update(result[2])

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    if self.preload:
      img = self.img_data[idx]
      lbl = None if self.mode == 'test' else self.label_data[idx]
    else:
      s = self.samples[idx]
      img = imageio.imread(s[0])
      lbl = None if self.mode == 'test' else s[self.label_idx]
    sample = {'image': img, 'label': lbl}

    if self.transform:
      sample = self.transform(sample)
    return sample


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'SVHN' : get_SVHN,
                   'MNIST' : get_MNIST,
                   'tinyimagenet' : get_tinyimagenet,
                   'FashionMNIST' : get_Fashion_MNIST}
