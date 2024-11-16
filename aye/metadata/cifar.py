"""Metadata for CIFAR-10 dataset."""
import aye.metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "CIFAR10"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME / "CIFAR10"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "cifar_10"

DIMS = (3, 32, 32)
OUTPUT_DIMS = (1, )
MAPPING = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

TRAIN_SIZE = 45_000
VAL_SIZE = 5_000