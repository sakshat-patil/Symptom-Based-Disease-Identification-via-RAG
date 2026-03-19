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
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
