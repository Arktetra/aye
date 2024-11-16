from aye.data.data_module import DataModule, _download_raw_dataset, load_and_print_info
from aye.data.utils import BaseDataset, split_dataset
from aye.stem.hotdog import HotDogStem
from aye.utils import temporary_working_directory
from pathlib import Path
from torchvision.transforms import v2

from PIL import Image

import aye.metadata.hotdog as metadata
import os
import toml
import torch
import zipfile

METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME

TRAIN_FRAC = 0.8

class HotDog(DataModule):
    """A data module for the hot-dog-not-hot-dog dataset."""
    
    def __init__(self, args = None):
        super().__init__()
        
        self.data_dir = metadata.DL_DATA_DIRNAME
        self.mapping = metadata.MAPPING
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        
        self.train_transforms = HotDogStem(augment = True)
        self.test_transforms = HotDogStem(augment = False)
        
    def prepare_data(self):
        if not os.path.exists(PROCESSED_DATA_DIRNAME):
            _download_and_process_dataset()
            
    def setup(self):
        def _load_dataset(split, transform, target_transform):
            images, labels = _load_images_and_labels(split = "train")
            return BaseDataset(images, labels, transform, target_transform)
        
        target_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype = torch.float32, scale = True)
        ])
        
        train_val_dataset = _load_dataset(split = "train", transform = self.train_transforms, target_transform = target_transform)
        self.train_dataset, self.val_dataset = split_dataset(train_val_dataset, fraction = TRAIN_FRAC, seed = 41)
        self.test_dataset = _load_dataset(split = "test", transform = self.test_transforms, target_transform = target_transform)
        
    def __repr__(self):
        basic = (
            "HotDog Dataset\n"
            f"  Num classes: {len(self.mapping)}\n"
            f"  Mapping: {self.mapping}\n"
            f"  Dims: {self.input_dims}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        
        data = (
            f"  Train/val/test sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}\n"
            f"  Batch x stats:  {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"  Batch y stats:  {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        
        return basic + data
            
def _download_and_process_dataset():
    metadata = toml.load(METADATA_FILENAME)
    if not os.path.exists(DL_DATA_DIRNAME):
        _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)
    
def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping HotDog...")
    
    with temporary_working_directory(dirname):
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(PROCESSED_DATA_DIRNAME)
            
def _load_images_and_labels(split: str):
    hotdog_files = []
    not_hotdog_files = []
    
    hotdog_files = _png_filenames(label = 1, split = split)
    not_hotdog_files = _png_filenames(label = 0, split = split)
        
    images = [Image.open(file) for file in hotdog_files]
    labels = [1 for _ in hotdog_files]
    images.extend([Image.open(file) for file in not_hotdog_files])
    labels.extend([0 for _ in not_hotdog_files])
    
    return images, labels

def _png_filenames(label: int, split: str):
    subdir = None
    if label == 0:
        subdir = "not-hotdog"
    else:
        subdir = "hotdog"
        
    return list((PROCESSED_DATA_DIRNAME / "hotdog" / split / subdir).glob("*.png"))
            
if __name__ == "__main__":
    load_and_print_info(HotDog)