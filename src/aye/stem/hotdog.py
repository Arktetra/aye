from aye.stem.image import ImageStem

import aye.metadata.hotdog as metadata
import torchvision.transforms.v2 as v2
import torch

IMAGE_SIZE = metadata.DIMS[1:]

class HotDogStem(ImageStem):
    """A stem for handling images from the hot-dog-not-hot-dog dataset."""
    def __init__(
        self, 
        augment: bool = False,
    ):
        super().__init__()
        
        if not augment:
            self.pil_transforms = v2.Compose([
                v2.Resize(size = (256, 256)),
                v2.CenterCrop(size = IMAGE_SIZE)
            ])
        else:
            self.pil_transforms = v2.Compose([
                v2.Resize(size = (256, 256)),
                v2.RandomResizedCrop((224, 224), scale = (0.25, 1)),
                v2.RandomHorizontalFlip(),
            ])
        
        self.pil_to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype = torch.float32, scale = True),
            v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])