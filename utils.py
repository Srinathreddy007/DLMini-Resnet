import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import os 

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

    @staticmethod
    def conv3x3(
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1
        ) -> nn.Conv2d:
        """
        Defines a 3x3 convolutional layer without bias.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Default is 3.
            stride (int): Stride of the convolution. Default is 1.
            padding (int): Padding size. Default is 1.

        Returns:
            nn.Conv2d: 3x3 convolutional layer.
        """
        return nn.Conv2d(
           in_channels, 
           out_channels,
           kernel_size=kernel_size, 
           stride=stride, 
           padding=padding, 
           bias=False
           )
  
    @staticmethod
    def conv1x1( 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 1, 
        stride: int = 1
        ) -> nn.Conv2d:
        """
        Defines a 1x1 convolutional layer without bias.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Default is 1.
            stride (int): Stride of the convolution. Default is 1.

        Returns:
            nn.Conv2d: 1x1 convolutional layer.
        """
        return nn.Conv2d(
           in_channels, 
           out_channels, 
           kernel_size=kernel_size, 
           stride=stride, 
           bias=False
           )

class EarlyStopper:
    def __init__(
          self, 
          patience: int = 1, 
          min_delta: int = 0
          ) -> None:
        """
        Initializes the early stopper.

        Args:
            patience (int): Number of epochs to wait for improvement before early stopping. Default is 1.
            min_delta (int): Minimum change in loss for improvement. Default is 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(
          self, 
          validation_loss: float
          ) -> bool:
        """
        Determines if training should be stopped early based on validation loss performance.

        Args:
            validation_loss (float): Validation loss.

        Returns:
            bool: True if early stopping criteria met, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Plotter:
    def __init__(self):
        pass  
    
    @staticmethod
    def plot_train_loss(
        train_losses: List[float], 
        epochs: List[int],
        model_name: str
        ) -> None:
        """
        Plots training loss over epochs.

        Args:
            train_losses (List[float]): List of training losses.
            epochs (List[int]): List of epoch numbers.
            model_name (str): Name of the model.
        """
        sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train Loss vs Epochs')
        plt.legend()

        # Ensure directory exists and save the plot
        os.makedirs("./Plots/", exist_ok=True)
        plt.savefig(os.path.join("./Plots/", f'train_loss_ResNet-{model_name}.png'))
        plt.show()

    @staticmethod
    def plot_train_accuracy( 
        train_accuracies: List[float], 
        epochs: List[int],
        model_name: str
        ) -> None:
        """
        Plots training accuracy over epochs.

        Args:
            train_accuracies (List[float]): List of training accuracies.
            epochs (List[int]): List of epoch numbers.
            model_name (str): Name of the model.
        """
        sns.lineplot(x=epochs, y=train_accuracies, label='Train Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train Accuracy vs Epochs')
        plt.legend()

        # Ensure directory exists and save the plot
        os.makedirs("./Plots/", exist_ok=True)
        plt.savefig(os.path.join("./Plots/", f'train_accuracy_ResNet-{model_name}.png'))
        plt.show()

    @staticmethod
    def plot_loss_comparison(
        train_losses: List[float], 
        val_losses: List[float], 
        epochs: List[int], 
        model_name: str
        ) -> None:
        """
        Compares training and test loss over epochs.

        Args:
            train_losses (List[float]): List of training losses.
            val_losses (List[float]): List of validation losses.
            epochs (List[int]): List of epoch numbers.
            model_name (str): Name of the model.
        """
        sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
        sns.lineplot(x=epochs, y=val_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train Loss vs Test Loss')
        plt.legend()

        # Ensure directory exists and save the plot
        os.makedirs("./Plots/", exist_ok=True)
        plt.savefig(os.path.join("./Plots/", f'loss_ResNet-{model_name}.png'))
        plt.show()

    @staticmethod
    def plot_accuracy_comparison(
        train_accuracies: List[float], 
        val_accuracies: List[float], 
        epochs: List[int],
        model_name: str
        ) -> None:
        """
        Compares training and test accuracy over epochs.

        Args:
            train_accuracies (List[float]): List of training accuracies.
            val_accuracies (List[float]): List of validation accuracies.
            epochs (List[int]): List of epoch numbers.
            model_name (str): Name of the model.
        """
        sns.lineplot(x=epochs, y=train_accuracies, label='Train Accuracy')
        sns.lineplot(x=epochs, y=val_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train Accuracy vs Test Accuracy')
        plt.legend()

        # Ensure directory exists and save the plot
        os.makedirs("./Plots/", exist_ok=True)
        plt.savefig(os.path.join("./Plots/", f'accuracy_ResNet-{model_name}.png'))

        plt.show()
