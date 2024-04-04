import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

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
    return nn.Conv2d(
       in_channels, 
       out_channels,
       kernel_size=kernel_size, 
       stride=stride, 
       padding = padding, 
       bias=False
       )
  
  @staticmethod
  def conv1x1( 
      in_channels: int, 
      out_channels: int,
      kernel_size: int = 1, 
      stride: int = 1,  
      ) -> nn.Conv2d:
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
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(
          self, 
          validation_loss: float
          ) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class Plotter:
  def __init__(self):
    pass  

  def plot_train_loss(
        self, 
        train_losses: List[float], 
        epochs: List[int]
        ) -> plt:
    sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epochs')
    plt.legend()
    plt.show()

  def plot_train_accuracy(
        self, 
        train_accuracies: List[float], 
        epochs: List[int]
        ) -> plt:
    sns.lineplot(x=epochs, y=train_accuracies, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Epochs')
    plt.legend()
    plt.show()

  def plot_loss_comparison(
        self, 
        train_losses: List[float], 
        val_losses: List[float], 
        epochs: List[int]
        ) -> plt:
    sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
    sns.lineplot(x=epochs, y=val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Validation Loss')
    plt.legend()
    plt.show()

  def plot_accuracy_comparison(
        self, 
        train_accuracies: List[float], 
        val_accuracies: List[float], 
        epochs: List[int]
        ) -> plt:
    sns.lineplot(x=epochs, y=train_accuracies, label='Train Accuracy')
    sns.lineplot(x=epochs, y=val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Validation Accuracy')
    plt.legend()
    plt.show()
