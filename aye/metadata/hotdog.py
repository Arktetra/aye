import aye.metadata.shared as shared 

DL_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME / "HotDog"
RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "HotDog"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"

DIMS = (3, 224, 224)
OUTPUT_DIMS = (1,)
MAPPING = [
    "not hot-dog",
    "hot-dog",
]

