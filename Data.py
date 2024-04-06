import os 
import torchvision
import torch
from torch.utils.data import DataLoader, Subset

class LoadData:
    def __init__(self, data_path, *args, **kwargs):
        self.data_path = data_path
        if os.path.isabs(self.data_path):
            self.data_path = os.path.abspath(self.data_path)
        self._init_vars()
  
    def _init_vars(self):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.valid_split = 0.1
        self.BATCH_SIZE_TRAIN = 50
        self.BATCH_SIZE_TEST = 1000
  
    def _get_data(self):
        self.train_set = torchvision.datasets.CIFAR10(
            self.data_path, 
            train=True, 
            download=True,
            transform=self.transforms
        )

        self.train_len = int(len(self.train_set) * (1 - self.valid_split))
        train_indices = torch.arange(self.train_len)
        val_indices = torch.arange(self.train_len, len(self.train_set))

        train_subset = Subset(self.train_set, train_indices)
        self.val_subset = Subset(self.train_set, val_indices)

        train_loader = DataLoader(
            train_subset, 
            batch_size=self.BATCH_SIZE_TRAIN, 
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_subset, 
            batch_size=self.BATCH_SIZE_TEST, 
            shuffle=False
        )

        self.test_set = torchvision.datasets.CIFAR10(
            self.data_path, 
            train=False, 
            download=True,
            transform=self.transforms
        )

        test_loader = DataLoader(
            self.test_set, 
            batch_size=self.BATCH_SIZE_TEST, 
            shuffle=True
        )

        return train_loader, val_loader, test_loader  
    def __get_length__(self):
        print(f'Length of Train Data: {self.train_len}\n'
              f'Length of Validation Data: {len(self.val_subset)}\n'
              f'Length of Test Data: {len(self.test_set)}')
  
    def _get_class_length(self):
        self.class_count = {}
        for _, index in self.train_set:
            label = self.train_set.classes[index]
            self.class_count[label] = self.class_count.get(label, 0) + 1
        print(f'Class Count: {self.class_count}')

