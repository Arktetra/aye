from .image import ImageStem
import aye.metadata.cifar as metadata

import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

IMAGE_SHAPE = metadata.DIMS[1:]

class CIFAR10Stem(ImageStem):
    """A stem for handling images from CIFAR-10 dataset."""
    
    def __init__(
        self,
        augment = False,
        color_jitter_kwargs = None,
        random_affine_kwargs = None,
        random_perspective_kwargs = None,
        gaussian_blur_kwargs = None,
        sharpness_kwargs = None
    ) -> None:
        
        super().__init__()
        
        if not augment:
            self.pil_transforms = transforms.Compose([
                transforms.CenterCrop(IMAGE_SHAPE)
            ])
        else:
            if color_jitter_kwargs is None:
                color_jitter_kwargs = {
                    "brightness": 0.5,
                    "contrast": 0.5,
                    "saturation": 0.5,
                    "hue": 0.5
                }
            
            if random_affine_kwargs is None:
                random_affine_kwargs = {
                    "degrees": 3,
                    "scale": (0.95, 1),
                    "shear": 6,
                    "interpolation": transforms.InterpolationMode.BILINEAR
                }
            
            if random_perspective_kwargs is None:
                random_perspective_kwargs = {
                    "distortion_scale": 0.2,
                    "p": 0.5,
                    "interpolation": transforms.InterpolationMode.BILINEAR 
                }
                
            if gaussian_blur_kwargs is None:
                gaussian_blur_kwargs = {
                    "kernel_size": (3, 3),
                    "sigma": (0.1, 1.0)
                }
                
            if sharpness_kwargs is None:
                sharpness_kwargs = {
                    "sharpness_factor": 2,
                    "p": 0.5
                }
                
            self.pil_transforms = transforms.Compose([
                transforms.ColorJitter(**color_jitter_kwargs),
                transforms.Resize(size = (224, 224)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.RandomRotation(degrees = 3, interpolation = transforms.InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(size = IMAGE_SHAPE, scale = (0.25, 1)),
            ])
            
        self.pil_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4605, 0.4693, 0.4594], [0.2589, 0.2567, 0.2574])
        ])