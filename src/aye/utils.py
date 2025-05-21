from collections.abc import Mapping
from pathlib import Path
from typing import Union, Tuple, Any, Sequence
from tqdm import tqdm
from urllib import request

from PIL import Image

import contextlib
import hashlib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

def to_cpu(x):
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    return x.detach().cpu()

def has_instance(list: Sequence, type: Any):
    for o in list:
        if isinstance(o, type):
            return True
    return False

def hasattrs(obj: Any, attrs: Tuple[str]) -> bool:
    """Checks if an object obj has all attribute in attrs.

    Args:
        obj (Any): an object.
        attrs (Tuple[str]): a tuple of attributes.

    Returns:
        bool: True if the condition holds, otherwise False.
    """
       
    return all(hasattr(obj, attr) for attr in attrs)

def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype = "uint8")[y]

def read_image_pil(image_uri: Union[Path, str], grayscale = False) -> Image:
    """Read PIL Image from an uri.

    Args:
        image_uri (Union[Path, str]): an image uri.
        grayscale (bool, optional): condition to grayscale the image. Defaults to False.

    Returns:
        Image: a PIL Image.
    """
    
    def read_image_from_filename(image_filename: Union[Path, str], grayscale: bool) -> Image:
        with Image.open(image_filename) as image:
            if grayscale:
                image = image.convert(model = "L")
            else:
                image = image.convert(mode = image.mode)
            return image
        
    def read_image_from_url(image_url: str, grayscale: bool) -> Image:
        url_response = request.urlopen(str(image_url))
        # image_array = np.array(bytearray(url_response.read()), dtype = np.uint8)
        return Image.open(url_response)
    
    local_file = os.path.exists(image_uri)
    
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, grayscale)
        else:
            img = read_image_from_url(image_uri, grayscale)
    except Exception as e:
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))

    return img
        
def read_image(image_uri: Union[Path, str], grayscale = False) -> np.ndarray:
    return read_image_pil(image_uri, grayscale)

def show_image(img: Union[np.ndarray, torch.Tensor], ax = None, figsize = None, title = None, noframe = True, **kwargs) -> plt.Axes:
    """Display an image from array or tensor.

    Args:
        img (Union[np.ndarray, torch.Tensor]): image in array or tensor form.
        figsize (_type_, optional): size of the figure. Defaults to None.
        title (_type_, optional): title of the figure. Defaults to None.
        noframe (bool, optional): condition to frame the figure. Defaults to True.

    Returns:
        plt.Axes: an axes in which the image is plotted.
    """
    
    if hasattrs(img, ("cpu", "detach", "permute")):
        img = img.detach().cpu()
        if len(img.shape) == 3 and img.shape[0] < 5:
            img = img.permute(1, 2, 0)
    elif not isinstance(img, np.ndarray):
        img = np.array(img)
    
    if img.shape[-1] == 1:
        img = img[..., 0]
    
    if ax is  None:
        _, ax = plt.subplots(figsize = figsize)
    
    ax.imshow(img, **kwargs)
    
    if title is not None:
        ax.set_title(title)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    if noframe:
        ax.axis("off")
        
    return ax

def get_grid(n: int, nrows: int = None, ncols: int = None, title: str = None, weight: str = "bold", size: int = 14, **kwargs):
    """Return a grid of `nrows` and `ncols` with `n` axes.

    Args:
        n (int): Number of axes.
        nrows (int, optional): Number of rows. Defaults to None.
        ncols (int, optional): Number of cols. Defaults to None.
        title (str, optional): Title of the figure. Defaults to None.
        weight (str, optional): Title font weight. Defaults to "bold".
        size (int, optional): Ritle font size. Defaults to 14.
    """
    if nrows:
        ncols = ncols or int(np.floor(n / nrows))
    elif ncols:
        nrows = nrows or int(np.ceils(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    
    # Make the excess axes invisible
    for i in range(n, nrows * ncols):
        axes.flat[i].set_axis_off()
    
    if title is not None:
        fig.suptitle(title, weight = weight, size = size)
        
    return fig, axes

def show_images(sample: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], mapping = None, nrows = 1, ncols = 1, figsize = None):
    """Display a grid of images from the sample.

    Args:
        sample (Tuple[torch.Tensor, torch.Tensor]): sample from which image is displayed.
        mapping (_type_, optional): a mapping to actual labels. Defaults to None.
        nrows (int, optional): number of rows in the grid. Defaults to 1.
        ncols (int, optional): number of columns in the grid. Defaults to 1.
        figsize (_type_, optional): size of the figure. Defaults to None.
    """    
    if len(sample) == 2:
        imgs, labels = sample
    else:
        imgs, labels = sample, None
    
    if len(imgs.shape) == 4 and imgs.shape[1] < 5:
        imgs = imgs.permute(0, 2, 3, 1)
        
    if nrows == 1 and ncols == 1:
        title = None
        if mapping is not None:
            title = mapping[labels[0]]
        elif labels is not None:
            title = labels[0]
        return show_image(imgs[0], title = title)
    
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    
    if nrows == 1 or ncols == 1:
        for idx in range(nrows * ncols):
            title = None
            if mapping is not None:
                title = mapping[labels[idx]]
            elif labels is not None:
                title = labels[idx]
            axes[idx] = show_image(img = imgs[idx], ax = axes[idx], title = title)
    else:
        plt.subplots_adjust(hspace = 0.25, wspace = -0.5)
        for row in range(nrows):
            for col in range(ncols):
                if (row * ncols + col) > (len(imgs) - 1):
                    axes[row][col].axis("off")
                    continue
                title = None
                if mapping is not None:
                    title = mapping[labels[row * ncols + col]]
                elif labels is not None:
                    title = labels[row * ncols + col]
                axes[row][col] = show_image(img = imgs[row * ncols + col], ax = axes[row][col], title = title)

@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[Path, str]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curr_dir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curr_dir)

def compute_sha1(filename: Union[Path, str]):
    """Returns SHA1 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

def compute_sha256(filename: Union[Path, str]):
    """Returns SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
    
def compute_md5(filename: Union[Path, str]):
    """Returns md5sum of a file."""
    with open(filename, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

class TqdmUpTo(tqdm):
    
    def update_to(self, blocks: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)
    
def download_url(url: str, filename: Union[Path, str]):
    """Download a file from url to filename, with a progress bar."""

    with TqdmUpTo(unit = "B", unit_scale = True, unit_divisor = 1024, miniters = 1) as t:
        request.urlretrieve(url, filename, reporthook = t.update_to, data = None)
        
def check_len(tensor: torch.Tensor, n):
    assert len(tensor) == n, \
        f"length of tensor is {len(tensor)} which is different from {n}."
        
def check_shape(tensor: torch.Tensor, shape):
    """Check the shape of a tensor."""
    assert tensor.shape == shape, \
        f"shape of tensor is {tensor.shape} which is different from {shape}."
        
def random_float_test(module: torch.nn.Module, shape, device = "cpu"):
    model = module.to(device)
    rand_input = torch.randn(shape).to(device)
    print(f"Input shape: {rand_input.shape}")
    output = model(rand_input)
    if isinstance(output, tuple):
        output = output[0]
    print(f"Output shape: {output.shape}")
    
def random_int_test(module: torch.nn.Module, shape, device = "cpu"):
    model = module.to(device)
    rand_input = torch.randint(100, 1000, shape).to(device)
    print(f"Input shape: {rand_input.shape}")
    output = model(rand_input)
    if isinstance(output, tuple):
        output = output[0]
    print(f"Output shape: {output.shape}")
    
def load_gpt2_test(module: torch.nn.Module, gpt2, input, device = "cpu"):
    model = module.to(device)
    model.load_state_dict(gpt2.state_dict(), strict = False)
    print(f"Input shape: {input.shape}")
    output = model(input)
    if isinstance(output, tuple):
        output = output[0]
    print(f"Output shape: {output.shape}")
    try:
        reference_output = gpt2(input)
    except:
        reference_output = gpt2(input, input, input)
    print(f"Reference output shape: {reference_output.shape}")
    comparison = torch.isclose(output, reference_output, atol = 1e-4, rtol = 1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct.")
    
def gelu_new(input: torch.Tensor):
    """Implementation of GeLU used by GPT2, which is subtly different from PyTorch's."""
    return (
        0.5 
        * input
        * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )

class Timer:
    def __init__(self):
        self.times = []
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()