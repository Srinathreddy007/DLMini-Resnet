import os 
import torchvision
import torch
from torch.utils.data import DataLoader, Subset

class LoadData:
    def __init__(self, data_path, batch_size_train, validation_split, *args, **kwargs):
        """
        Initialize the data loader with the given parameters.

        Args:
            data_path (str): Path to the CIFAR10 dataset.
            batch_size_train (int): Batch size for training.
            validation_split (float): Proportion of training data to use for validation split.
        """
        self.data_path = data_path
        # Convert to absolute path if not already
        if os.path.isabs(self.data_path):
            self.data_path = os.path.abspath(self.data_path)
        self.BATCH_SIZE_TRAIN = batch_size_train
        self.valid_split = validation_split
        self._init_vars()  # Initialize additional variables and setup
  
    def _init_vars(self):
        """
        Initialize variables and setup transformations for data loading.
        """
        # Define image transformations for training
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # Define image transformations for testing
        self.transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.BATCH_SIZE_TEST = 1000  # Larger batch size for testing
  
    def _get_data(self):
        """
        Load CIFAR10 dataset, split into training and validation sets, and create DataLoader instances.
        
        Returns:
            tuple: DataLoader instances for training, validation, and test sets.
        """
        # Load CIFAR10 training data with transformations
        self.train_set = torchvision.datasets.CIFAR10(
            self.data_path, 
            train=True, 
            download=True,
            transform=self.transforms_train
        )
        # Calculate indices for training/validation split
        self.train_len = int(len(self.train_set) * (1 - self.valid_split))
        train_indices = torch.arange(self.train_len)
        val_indices = torch.arange(self.train_len, len(self.train_set))

        # Subset the data into training and validation parts
        train_subset = Subset(self.train_set, train_indices)
        self.val_subset = Subset(self.train_set, val_indices)

        # DataLoader for training and validation sets
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.BATCH_SIZE_TRAIN, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_subset, 
            batch_size=self.BATCH_SIZE_TEST, 
            shuffle=False
        )

        # Load CIFAR10 test data
        self.test_set = torchvision.datasets.CIFAR10(
            self.data_path, 
            train=False, 
            download=True,
            transform=self.transforms_test
        )
        # DataLoader for test set
        self.test_loader = DataLoader(
            self.test_set, 
            batch_size=self.BATCH_SIZE_TEST, 
            shuffle=True
        )

        return self.train_loader, self.val_loader, self.test_loader  # Return all data loaders
    
    def __get_length__(self):
        """
        Print the length of the datasets for debugging or information.
        """
        print(f'Length of Train Data: {self.train_len}\n'
              f'Length of Validation Data: {len(self.val_subset)}\n'
              f'Length of Test Data: {len(self.test_set)}')
  
    def _get_class_length(self):
        """
        Calculate the number of instances per class in the training set and print the class count.
        """
        self.class_count = {}
        for _, index in self.train_set:
            label = self.train_set.classes[index]
            self.class_count[label] = self.class_count.get(label, 0) + 1
        print(f'Class Count: {self.class_count}')
