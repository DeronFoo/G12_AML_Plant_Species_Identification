# common/config.py
from pathlib import Path

# .../Approach3/common/config.py  →  PROJECT_ROOT = .../Approach3
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIRNAME = "AML_project_herbarium_dataset"
DATA_ROOT = PROJECT_ROOT / DATASET_DIRNAME   # <-- no "dataset" here

TRAIN_DIR = DATA_ROOT / "train"             # train/herbarium, train/photo
TEST_DIR = DATA_ROOT / "test"               # test/*.jpg

LIST_DIR = DATA_ROOT / "list"
TRAIN_LIST = LIST_DIR / "train.txt"
TEST_LIST = LIST_DIR / "test.txt"
SPECIES_LIST = LIST_DIR / "species_list.txt"
GROUNDTRUTH = LIST_DIR / "groundtruth.txt"
CLASS_WITH_PAIRS = LIST_DIR / "class_with_pairs.txt"
CLASS_WITHOUT_PAIRS = LIST_DIR / "class_without_pairs.txt"

NUM_CLASSES = 100

IMAGE_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
