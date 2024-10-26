"""MNIST DataModule."""
from aye.data.data_module import DataModule, load_and_print_info
from aye.stem.image import MNISTStem
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST

import argparse
import aye.metadata.mnist as metadata

class MNIST(DataModule):
    """A data module for MNIST dataset."""
    
    def __init__(self, args = argparse.Namespace) -> None:
        
        super().__init__(args)
        
        self.data_dir = metadata.DOWNLOADED_DATA_DIRNAME
        self.transform = MNISTStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.mapping = metadata.MAPPING
        
    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMNIST(root = self.data_dir, train = True, download = True)
        TorchMNIST(root = self.data_dir, train = False, download = True)
        
    def setup(self, stage = None) -> None:
        """Split into train, val, and test sets."""
        mnist_full = TorchMNIST(self.data_dir, train = True, transform = self.transform)
        self.train_dataset, self.val_dataset = random_split(mnist_full, [metadata.TRAIN_SIZE, metadata.VAL_SIZE])
        self.test_dataset = TorchMNIST(self.data_dir, train = False, transform = self.transform)
        
    def __repr__(self):
        basic = (
            "MNIST Dataset\n"
            f"  Num Classes: {len(self.mapping)}\n"
            f"  Mapping: {self.mapping}\n"
            f"  Dims: {self.input_dims}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        
        data = (
            f"  Train/val/test sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}\n"
            f"  Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"  Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        
        return basic + data
        
if __name__ == "__main__":
    load_and_print_info(MNIST)