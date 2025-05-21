"""Metadata for The Time Machine."""

import aye.metadata.shared as shared

DL_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME / "TimeMachine"
RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "TimeMachine"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"

TIME_STEPS = 10     # number of tokens in each subsequence
TRAIN_SIZE = 10000
VAL_SIZE = 5000