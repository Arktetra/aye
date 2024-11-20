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

class TimeMachine(DataModule):
    def __init__(
        self, 
        batch_size: int = None,
        train_size: int = None, 
        val_size: int = None
    ):
        super().__init__()
        self.train_transform = None
        self.data_dir = metadata.DL_DATA_DIRNAME
        
        if batch_size:
            self.batch_size = batch_size
            
        self.train_size = train_size if train_size else TRAIN_SIZE
        self.val_size = val_size if val_size else VAL_SIZE 
        self.total_size = self.train_size + self.val_size
        
    def prepare_data(self):
        metadata = toml.load(METADATA_FILENAME)
        if not os.path.exists(self.data_dir):
            _download_raw_dataset(metadata, self.data_dir)
            
        with open(self.data_dir / metadata["filename"], "r") as f:
            self.data = f.read()
            
    def setup(self):
        def _load_dataset(transform, target_transform):
            corpus, self.vocab = self.build()
            array = torch.tensor([corpus[i:i + metadata.TIME_STEPS + 1]
                                for i in range(len(corpus) - metadata.TIME_STEPS)])
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
        
            
    def _tokenize(self):
        return list(self.data)
    
    def build(self):
        tokens = self._tokenize()
        vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab