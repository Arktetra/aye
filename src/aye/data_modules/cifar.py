"""CIFAR-10 data module."""
from aye.data.data_module import DataModule, load_and_print_info 
from aye.stem.cifar import CIFAR10Stem
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10 as TorchCIFAR10

import argparse 
import aye.metadata.cifar as metadata 
import os

class CIFAR10(DataModule):
    """A data module for CIFAR-10 dataset."""
    
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__(args)
        
        self.data_dir = metadata.DOWNLOADED_DATA_DIRNAME
        self.input_dims = metadata.DIMS 
        self.output_dims = metadata.OUTPUT_DIMS 
        self.mapping = metadata.MAPPING 
        self.train_transforms = CIFAR10Stem(augment = True)
        self.test_transforms = CIFAR10Stem(augment = False)
        
    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test CIFAR-10 dataset from PyTorch canonical source."""
        if not os.path.exists(self.data_dir):
            TorchCIFAR10(root = self.data_dir, train = True, download = True)
            TorchCIFAR10(root = self.data_dir, train = False, download = True)
        
    def setup(self, stage = None) -> None:
        """Split into train, val, and test sets."""
        cifar_full = TorchCIFAR10(self.data_dir, train = True, transform = self.train_transforms)
        self.train_dataset, self.val_dataset = random_split(cifar_full, [metadata.TRAIN_SIZE, metadata.VAL_SIZE])
        self.test_dataset = TorchCIFAR10(self.data_dir, train = False, transform = self.test_transforms)
        
    def __repr__(self):
        basic = (
            "CIFAR-10 Dataset\n"
            f"  Num classes: {len(self.mapping)}\n"
            f"  Mapping: {self.mapping}\n"
            f"  Dims: {self.input_dims}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            return basic 
        
        x, y = next(iter(self.train_dataloader()))
        
        data = (
            f"  Train/val/test sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}\n"
            f"  Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"  Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}"
        )
        
        return basic + data
        
        
if __name__ == "__main__":
    load_and_print_info(CIFAR10)