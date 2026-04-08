"""Project configuration and constants."""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model settings
EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
CHUNK_SIZE = 256  # tokens per passage
TOP_K = 10  # number of retrieved passages

# FP-Growth settings
# min_support lowered from 0.01 to 0.005 after Check-in 3 showed only
# 20/41 diseases were covered at the higher threshold.
MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.5
