from PIL import Image
from typing import Sequence, Union, Callable, Optional, Tuple, Any

import torch

SequenceOrTensor = Union[Sequence, torch.Tensor]

class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class that processed data and targets through optional transforms.

    Args:
            data (SequenceOrTensor): torch tensors, numpy arrays or PIL images.
            targets (SequenceOrTensor): torch tensors or numpy arrays.
            transform (Optional[Callable], optional): function that transforms a datum. Defaults to None.
            target_transform (Optional[Callable], optional): function that transforms a target. Defaults to None.

    Raises:
        ValueError: raised if length of data and targets is not equal.
    """
    
    def __init__(
        self, 
        data: Union[SequenceOrTensor, Tuple], 
        targets: Optional[SequenceOrTensor] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(data, tuple):
            if not all(len(x) == len(targets) for x in data):
                raise ValueError("Data and targets must be of equal length.")
        else:
            if len(data) != len(targets):
                raise ValueError("Data and targets must be of equal length.")
        
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.targets)
    
    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Return a datum and its target after processing by transforms."""
        if isinstance(self.data, tuple):
            datum = [d[idx] for d in self.data]
            target = self.targets[idx]
            
            if self.transform is not None:
                datum = [self.transform[d] for d in datum] 
                
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            return *datum, target
        else:
            datum, target = self.data[idx], self.targets[idx]
            
            if self.transform is not None:
                datum = self.transform(datum)       
        
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            return datum, target

def split_dataset(
    base_dataset: BaseDataset,
    fraction: float,
    seed: int
) -> Tuple[BaseDataset, BaseDataset]:
    """split input base dataset into two base datasets, the first with size
    `fraction * size of the base dataset` and the other with size `(1 - size
    of the base dataset)`.

    Args:
        base_dataset (BaseDataset): the base dataset to split.
        fraction (float): used for splitting the base dataset.
        seed (int): for determinism.

    Returns:
        Tuple[BaseDataset, BaseDataset]: splitted base dataset.
    """
    first_split_size = int(len(base_dataset) * fraction)
    second_split_size = len(base_dataset) - first_split_size
    
    return torch.utils.data.random_split(
        dataset = base_dataset, 
        lengths = [first_split_size, second_split_size], 
        generator = torch.Generator().manual_seed(seed)
    )
    
def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    
    if scale_factor == 1:
        return image
    
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample = Image.BILINEAR)

def no_space(char: str, prev_char: str):
    """Check whether there is space before a punctuation or not."""
    
    return char in ",.!?" and prev_char != " "

def pad_or_trim(seq: str, time_steps: int):
    """Pad or trim a sequence depending on the value of time steps."""
    
    return seq[:time_steps] if len(seq) > time_steps else seq + ["<pad>"] * (time_steps - len(seq))