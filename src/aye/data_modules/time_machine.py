from aye.data import Vocab
from aye.data.data_module import DataModule, _download_raw_dataset
from aye.data.utils import BaseDataset, split_dataset

import aye.metadata.time_machine as metadata
import os 
import toml
import torch

METADATA_FILENAME = metadata.METADATA_FILENAME
TRAIN_SIZE = metadata.TRAIN_SIZE 
VAL_SIZE = metadata.VAL_SIZE
TIME_STEPS = metadata.TIME_STEPS

class TimeMachine(DataModule):
    def __init__(
        self, 
        shuffle: int = True,
        batch_size: int = None,
        train_size: int = None, 
        val_size: int = None,
        time_steps: int = None
    ):
        super().__init__()
        self.train_transform = None
        self.data_dir = metadata.DL_DATA_DIRNAME
        
        self.shuffle = shuffle
        
        if batch_size:
            self.batch_size = batch_size
            
        self.train_size = train_size if train_size else TRAIN_SIZE
        self.val_size = val_size if val_size else VAL_SIZE 
        self.total_size = self.train_size + self.val_size
        
        self.time_steps = time_steps if time_steps else TIME_STEPS
        
    def prepare_data(self):
        metadata = toml.load(METADATA_FILENAME)
        if not os.path.exists(self.data_dir):
            _download_raw_dataset(metadata, self.data_dir)
            
        with open(self.data_dir / metadata["filename"], "r") as f:
            self.data = f.read()
            
    def setup(self):
        def _load_dataset(transform, target_transform):
            corpus, self.vocab = self.build()
            array = torch.tensor([corpus[i:i + self.time_steps + 1]
                                for i in range(len(corpus) - self.time_steps)])
            array = array[:self.total_size, :self.total_size]
            x, y = array[:, :-1], array[:, 1:]
            return BaseDataset(x, y, transform, target_transform)
        
        target_transform = None
        
        train_val_dataset = _load_dataset(transform = self.train_transform, target_transform = target_transform)
        
        self.train_dataset, self.val_dataset = split_dataset(
            train_val_dataset, 
            fraction = self.train_size / self.total_size, 
            seed = 41
        )
        
    def __repr__(self):
        basic = (
            "TimeMachine Dataset\n"
            f"  Time Steps: {self.time_steps}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None:
            return basic 
        
        x, y = next(iter(self.train_dataloader()))
        
        data = (
            f"  Vocab size: {len(self.vocab)}\n"
            f"  Train/val sizes: {len(self.train_dataset)}, {len(self.val_dataset)}\n"
            f"  Batch x stats: {(x.shape, x.dtype)}\n"
            f"  Batch y stats: {(y.shape, y.dtype)}\n"
        )
        
        return basic + data
        
            
    def _tokenize(self):
        return list(self.data)
    
    def build(self):
        tokens = self._tokenize()
        vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab