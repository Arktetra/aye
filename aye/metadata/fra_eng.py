import aye.metadata.shared as shared

DL_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME / "fra-eng"
RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "FraEng"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"

TRAIN_FRAC = 0.95
TIME_STEPS = 10