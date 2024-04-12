from time import time
from tqdm import tqdm
from Data import *
from ResNet import ResNet
import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from typing import Type, Union, List, Optional, Callable, Any
from utils import EarlyStopper
from utils import Plotter 

class Train:
    def __init__(self):
        # Initialize to track the best accuracy achieved across all tests
        self.best_acc = 0

    def train(
        self, 
        epoch: int,
        model: ResNet, 
        device: torch.device,
        criterion: Type[nn.Module],
        optimizer: Optimizer,
        train_loader: Type[DataLoader]
        ) -> Union[float, float]:
        # Initialize training parameters and set the model to training mode
        train_loss = total = correct_preds = 0  
        model.to(device)
        model.train()
        start = time()  # Record the start time

        # Train the model for one epoch
        for batch, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagate errors
            optimizer.step()  # Update model parameters

            # Aggregate statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_preds += predicted.eq(labels).sum().item()  

        end = time()  # Record the end time
        train_accuracy = 100. * correct_preds / total  # Calculate training accuracy
        # Print training results for the epoch
        print("########################################################################################################################")
        print(f'Epoch: {epoch + 1} | Train Loss: {train_loss / (batch + 1):.2f} | Train Accuracy: {train_accuracy:.2f} | Time Elapsed: {end - start:.2f}sec')

        return train_accuracy, train_loss/(batch + 1)

    def test(
            self, 
            epoch: int,
            model: ResNet, 
            device: torch.device,
            criterion: Type[nn.Module],
            test_loader: Type[DataLoader],
            option: str, 
            model_name: str
            ) -> Union[float, float]:
        # Initialize testing parameters and set model to evaluation mode
        test_loss = correct_preds = total = 0
        start = time()  # Record the start time
        with torch.no_grad():  # Disable gradient computation
            for batch, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Aggregate statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_preds += predicted.eq(labels).sum().item()
        end = time()  # Record the end time
        test_accuracy = 100. * correct_preds / total  # Calculate test accuracy
        # Print test results for the epoch
        print(f'Epoch: {epoch + 1} | {option} Loss: {test_loss/(batch + 1):.2f} | {option} Accuracy: {test_accuracy:.2f} | Time Elapsed: {end-start:.2f}sec')
        print("############################################################################################################################")

        # Save the model if it's the best one based on accuracy
        save_dir = './saved_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if test_accuracy > self.best_acc:
            model.cpu()
            model_scripted = torch.jit.script(model)
            model_scripted.save(os.path.join(save_dir, f'ResNet-{model_name}.pt'))
            print("Saved Model")
            self.best_acc = test_accuracy

        return test_accuracy, test_loss/(batch + 1)

    def run(
            self, 
            epochs: int, 
            model: ResNet, 
            device: torch.device, 
            criterion: Type[nn.Module],
            optimizer: Optimizer,
            scheduler: torch.optim.lr_scheduler, 
            trainloader: Type[DataLoader], 
            testloader: Type[DataLoader], 
            option: str,
            model_name: str,
            early_stopper: EarlyStopper = EarlyStopper(),
            patience: int = 5,
            ):
        # Manage the training and testing process across multiple epochs
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        for epoch in tqdm(range(epochs)):  # Progress bar for epochs
            train_acc, train_loss = self.train(epoch, model, device, criterion, optimizer, trainloader)
            val_acc, val_loss = self.test(epoch, model, device, criterion, testloader, option, model_name)
            scheduler.step()  # Update the learning rate

            # Record performance metrics
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            # Check for early stopping
            if early_stopper.early_stop(val_loss):
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Plot Training and Testing performance
        Plotter.plot_train_loss(train_losses=train_loss_history, epochs=range(1, len(train_loss_history) + 1), model_name=model_name)
        Plotter.plot_train_accuracy(train_accuracies=train_acc_history, epochs=range(1, len(train_acc_history) + 1), model_name=model_name)
        Plotter.plot_loss_comparison(train_losses=train_loss_history, val_losses=val_loss_history, epochs=range(1, len(val_loss_history) + 1), model_name=model_name)
        Plotter.plot_accuracy_comparison(train_accuracies=train_acc_history, val_accuracies=val_acc_history, epochs=range(1, len(train_acc_history) + 1), model_name=model_name)
