"""Metadata for Fashion-MNIST dataset"""

import aye.metadata.shared as shared

DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME

DIMS = (1, 28, 28)
OUTPUT_DIMS = (1, )
MAPPING = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

TRAIN_SIZE = 55_000
VAL_SIZE = 5_000