from aye.data import Vocab
from aye.data.data_module import DataModule, _download_raw_dataset
from aye.data.utils import BaseDataset, split_dataset, no_space, pad_or_trim
 
from typing import List, Tuple

import aye.metadata.fra_eng as metadata 
import matplotlib.pyplot as plt
import toml
import torch

DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
RAW_DATA_DIRNAME = metadata.RAW_DATA_DIRNAME
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME
METADATA_FILENAME = metadata.METADATA_FILENAME
TRAIN_FRAC = metadata.TRAIN_FRAC
TIME_STEPS = metadata.TIME_STEPS

class FraEng(DataModule):
    def __init__(
        self, 
        batch_size = None,
        time_steps = None,  
        max_examples: int = None,
        train_frac: float = None
    ):
        super().__init__()
        self.max_examples = max_examples
        self.time_steps = time_steps if time_steps else TIME_STEPS
        self.train_frac = train_frac if train_frac else TRAIN_FRAC
        self.train_transform = None
        
        if batch_size:
            self.batch_size = batch_size
        
    def prepare_data(self):
        metadata = toml.load(METADATA_FILENAME)
        
        if not DL_DATA_DIRNAME.exists():
            _download_raw_dataset(metadata, DL_DATA_DIRNAME)
            
        with open(PROCESSED_DATA_DIRNAME / "fra-eng" / "fra.txt", encoding = "utf-8") as f:
            self.raw_text = f.read()
            
    def setup(self):
        def __load_dataset(transform, target_transform):
            text = self.__preprocess()
            src, tgt = self.__tokenize(text)
            
            self.src_array, self.src_vocab, self.src_valid_len = self.__build_array(src)
            self.tgt_array, self.tgt_vocab, self.tgt_valid_len = self.__build_array(tgt, is_target = True)
            
            if self.max_examples:
                return BaseDataset(
                    (self.src_array[:self.max_examples], self.tgt_array[:self.max_examples, :-1], self.src_valid_len),
                    self.tgt_array[:self.max_examples, 1:], 
                    transform, 
                    target_transform
                )
            else:
                return BaseDataset(
                    (self.src_array, self.tgt_array[:, :-1], self.src_valid_len),
                    self.tgt_array[:, 1:], 
                    transform, 
                    target_transform
                )  
        
        target_transform = None 
        
        train_val_dataset = __load_dataset(self.train_transform, target_transform)
        
        self.train_dataset, self.val_dataset = split_dataset(
            train_val_dataset,
            fraction = self.train_frac,
            seed = 41
        )
        
    def __repr__(self):
        basic = (
            "FraEng Dataset\n"
            f"  Time Steps: {self.time_steps}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None:
            return basic 
        
        src, tgt, valid_src_len, label = next(iter(self.train_dataloader()))
        
        data = (
            f"  Source vocab size: {len(self.src_vocab)}\n"
            f"  Target vocab size: {len(self.tgt_vocab)}\n"
            f"  Train/val sizes: {len(self.train_dataset)}, {len(self.val_dataset)}\n"
            f"  Batch src stats: {(src.shape, src.dtype)}\n"
            f"  Batch valid_src_len stats: {(valid_src_len.shape, valid_src_len.dtype)}\n"
            f"  Batch tgt stats: {(tgt.shape, tgt.dtype)}\n"
            f"  Batch label stats: {(label.shape, label.dtype)}\n"
        )
        
        return basic + data
            
    def __preprocess(self):
        text = self.raw_text.replace("\u202f", " ").replace("\xa0", " ")
        processed_text = [" " + char if i > 0 and no_space(char, text[i - 1]) else char
                    for i, char in enumerate(text.lower())]
        return "".join(processed_text)
        
    def __tokenize(self, text) -> Tuple[List[List[str]], List[List[str]]]:
        src, tgt = [], []
        for i, parts in enumerate(text.split("\n")):
            if self.max_examples and i >= self.max_examples:
                return src, tgt
        
            parts = parts.split("\t")
            
            if len(parts) == 2:
                src.append([t for t in f"{parts[0]} <eos>".split(" ") if t])
                tgt.append([t for t in f"{parts[1]} <eos>".split(" ") if t])
                
        return src, tgt
    
    def __build_array(self, sentences, vocab = None, is_target = False):
        sentences = [pad_or_trim(seq, self.time_steps) for seq in sentences]
        
        if is_target:
            sentences = [["<bos>"] + seq for seq in sentences]
            
        if vocab is None:
            vocab = Vocab(sentences, min_freq = 2)
            
        array = torch.tensor([vocab[s] for s in sentences])
        valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(dim = 1)
        
        return array, vocab, valid_len
    
    def build(self, src_sentences, tgt_sentences):
        raw_text = "\n".join([src + "\t" + tgt for src, tgt in zip(src_sentences, tgt_sentences)])
        src, tgt = self.__tokenize(raw_text)
        src_array, _, src_valid_len = self.__build_array(src, vocab = self.src_vocab)
        tgt_array, _, _ = self.__build_array(tgt, vocab = self.tgt_vocab, is_target = True)
        return src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]
    
    def show_list_len_pair_hist(self):
        text = self._preprocess()
        src, tgt = self._tokenize(text)
        plt.hist([[len(o) for o in src], [len(o) for o in tgt]], label = ("source", "target"))
        plt.xlabel("# tokens per sequence")
        plt.ylabel("count")
        plt.legend()
        plt.show()