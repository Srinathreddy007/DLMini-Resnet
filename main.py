from train import *
from Data import *
from BasicBlock import BasicBlock
from BottleNeck import BottleNeck
from torchinfo import summary
from torch import optim
import torch.nn as nn
from typing import Type, Union, List, Optional, Callable, Any
import argparse
import json

def main(
        data_path: str,
        batch_size_train: int,
        validation_split: float,

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
        model_name: str,

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
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = "mps:0"
        print("Device set as MPS")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        device = "cuda:0"
        print("Device set as CUDA")
    else:
        device = "cpu"
        print("No GPU available using CPU")

    print("Loading Data....")
    data = LoadData(data_path, batch_size_train, validation_split)
    train_loader, val_loader, test_loader = data._get_data()# test loader
    data.__get_length__()
    data._get_class_length()

    print('Setting Parameters')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

    early_stopper = EarlyStopper(min_delta)

    print("Training....")
    print()
    print(type(epochs))
    train.run(
        epochs=epochs,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        testloader=test_loader,
        option=option,
        model_name=model_name,
        early_stopper=early_stopper,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./Data/', type=str)
    parser.add_argument('--batch_size_train', default=50, type=int)
    parser.add_argument('--validation_split', default=0.1, type=float)
    parser.add_argument('--blocks', default=[BasicBlock, BasicBlock, BasicBlock, BasicBlock], type=List[Type[Union[BasicBlock, BottleNeck]]])
    parser.add_argument('--num_blocks', default='[2, 2, 2, 2]',type=str)
    parser.add_argument('--channel_size', default='[64, 128, 232, 268]', type=str)
    parser.add_argument('--conv_kernel_size', default=3, type=int)
    parser.add_argument('--he_init', default=False, type=bool)
    parser.add_argument('--zero_init_residual', default=False, type=bool)
    parser.add_argument('--min_delta', default=0.001, type=float)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--option', default='Test', type=str)
    parser.add_argument('--model_name',type=str)

    args = parser.parse_args()
    args.channel_size = json.loads(args.channel_size)
    args.num_blocks = json.loads(args.num_blocks)

    main(args.data_path,
         args.batch_size_train,
         args.validation_split,
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
         args.option,
         args.model_name
        )
