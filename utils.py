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
        # Defines a 3x3 convolutional layer without bias
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
        # Defines a 1x1 convolutional layer without bias
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
        # Initializes the early stopper with patience and minimum delta for loss improvement
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(
          self, 
          validation_loss: float
          ) -> bool:
        # Determine if training should be stopped early based on validation loss performance
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
        # Plots training loss over epochs
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
        # Plots training accuracy over epochs
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
        # Compares training and Test loss over epochs
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
        # Compares training and Test accuracy over epochs
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
