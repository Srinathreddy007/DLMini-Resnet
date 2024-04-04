from train import *
from Data import *
from BasicBlock import BasicBlock
from BottleNeck import BottleNeck
from torchinfo import summary
from torch import optim
import torch.nn as nn
import argparse

def main(
        data_path: str,

        blocks: Type[Union[BasicBlock, BottleNeck]],
        num_blocks: List[int],
        channel_size: List[int],
        conv_kernel_size: int,
        he_init: bool,
        zero_init_residual: bool, 

        min_delta: float,

        epochs: int,
        learning_rate: float, # 0.01
        momentum: Optional[float], # 0.9
        weight_decay: Optional[float] ,# 5e-04
        option: str,

) -> None:
    train = Train() 

    print('Loading Model....')
    model = ResNet(
        block=blocks, 
        num_blocks=num_blocks, 
        channel_size=channel_size, 
        conv_kernel_size=conv_kernel_size,
        he_init=he_init,
        zero_init_residual=zero_init_residual
        )
    
    if summary(model).total_params > 5e+06:
        exit("Total Number of Parameters greater than 5 Million")
    
    print("Checking for GPU...")
    if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # Remove if you don't have MacBook
        device = "mps:0"
        print("Device set as MPS")
    elif torch.has_cuda and torch.cuda.is_available():
        device = "cuda:0"
        print("Device set as CUDA")
    else:
        device = "cpu"
        print("No GPU available using CPU")
    
    print("Loading Data....")
    data = LoadData(data_path)
    train_loader, val_loader, test_loader = data._get_data()
    data.__get_length__()
    data._get_class_length()

    print('Setting Parameters')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

    early_stopper = EarlyStopper(min_delta)

    print("Training....")
    print()
    train.run( 
        epochs=epochs,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        testloader=val_loader,
        option=option,
        early_stopper=early_stopper,
    )
              


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./Data/')
    parser.add_argument('--blocks', default=[BasicBlock, BasicBlock, BasicBlock, BasicBlock])
    parser.add_argument('--num_blocks', default=[2, 2, 2, 2])
    parser.add_argument('--channel_size', default=[64, 128, 232, 268])
    parser.add_argument('--conv_kernel_size', default=3)
    parser.add_argument('--he_init', default=False)
    parser.add_argument('--zero_init_residual', default=False)
    parser.add_argument('--min_delta', default=0.001)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=5e-04)
    parser.add_argument('--option', default='Validation')

    args = parser.parse_args()

    main(args.data_path, 
         args.blocks, 
         args.num_blocks, 
         args.channel_size, 
         args.conv_kernel_size, 
         args.he_init,
         args.zero_init_residual, 
         args.min_delta, 
         args.epochs, 
         args.learning_rate, 
         args.momentum, 
         args.weight_decay,
         args.option
        )
