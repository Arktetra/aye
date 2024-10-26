from torchvision import transforms

import torch

class ImageStem:
    """A stem for models operating on images.
    
    Images are presumed to be provided as PIL images.
    """
    
    def __init__(self):
        self.pil_transforms = transforms.Compose([])
        self.pil_to_tensor = transforms.ToTensor()
        self.torch_transforms = torch.nn.Sequential()
        
    def __call__(self, img):
        img = self.pil_transforms(img)
        img = self.pil_to_tensor(img)
        
        with torch.no_grad():
            img = self.torch_transforms(img)
            
        return img
    
class MNISTStem(ImageStem):
    """A stem for handling images from the MNIST dataset."""
    
    def __init__(self):
        super().__init__()
        self.torch_transforms = torch.nn.Sequential(transforms.Normalize((0.1307, ), (0.3081, )))
        
class FashionMNISTStem(ImageStem):
    """A stem for handling images from the Fashion-MNIST dataset."""
    
    def __init__(self):
        super().__init__()
        self.torch_transforms = torch.nn.Sequential(transforms.Normalize((0.2860, ), (0.3205, )))