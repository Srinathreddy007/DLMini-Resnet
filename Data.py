import os 
import torchvision
import torch
from torch.utils.data import DataLoader, Subset

class LoadData:
    def __init__(self, data_path, batch_size_train, validation_split, *args, **kwargs):
        self.data_path = data_path
        if os.path.isabs(self.data_path):
            self.data_path = os.path.abspath(self.data_path)
        self.BATCH_SIZE_TRAIN = batch_size_train
        self.valid_split = validation_split
        self._init_vars()
  
    def _init_vars(self):
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.BATCH_SIZE_TEST = 1000
  
    def _get_data(self):
        self.train_set = torchvision.datasets.CIFAR10(
            self.data_path, 
            train=True, 
            download=True,
            transform=self.transforms_train
        )

        train_len = int(len(self.train_set) * (1 - self.valid_split))
        train_indices = torch.arange(train_len)
        val_indices = torch.arange(train_len, len(self.train_set))

        train_subset = Subset(self.train_set, train_indices)
        val_subset = Subset(self.train_set, val_indices)

        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.BATCH_SIZE_TRAIN, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=self.BATCH_SIZE_TEST, 
            shuffle=False
        )

        self.test_set = torchvision.datasets.CIFAR10(
            self.data_path, 
            train=False, 
            download=True,
            transform=self.transforms_test
        )

        self.test_loader = DataLoader(
            self.test_set, 
            batch_size=self.BATCH_SIZE_TEST, 
            shuffle=True
        )

        return self.train_loader, self.val_loader, self.test_loader  
    
    def __get_length__(self):
        print(f'Length of Train Data: {len(self.train_loader)}\n'
              f'Length of Validation Data: {len(self.val_loader)}\n'
              f'Length of Test Data: {len(self.test_loader)}')
  
    def _get_class_length(self):
        self.class_count = {}
        for _, index in self.train_set:
            label = self.train_set.classes[index]
            self.class_count[label] = self.class_count.get(label, 0) + 1
        print(f'Class Count: {self.class_count}')
