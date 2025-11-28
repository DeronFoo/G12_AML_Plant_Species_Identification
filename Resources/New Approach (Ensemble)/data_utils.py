# common/data_utils.py
from pathlib import Path
from typing import List, Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from config import (
    DATA_ROOT,
    TRAIN_DIR,
    TRAIN_LIST,
    TEST_LIST,
    GROUNDTRUTH,
    SPECIES_LIST,
    CLASS_WITH_PAIRS,
    CLASS_WITHOUT_PAIRS,
    NUM_CLASSES,
)
from transforms import build_train_transform, build_eval_transform
_LABEL_MAP = None


def _get_label_map():
    """
    Build mapping from raw species id (e.g. 105951) to [0 .. NUM_CLASSES-1]
    using species_list.txt, which is semicolon-separated, e.g.:

        105951; Trema orientalis; something...

    We only care about the first field before ';'.
    """
    global _LABEL_MAP
    if _LABEL_MAP is not None:
        return _LABEL_MAP

    mapping = {}
    idx = 0
    with SPECIES_LIST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # take text before first ';'
            left = line.split(";", 1)[0].strip()
            # skip non-numeric junk safely
            try:
                raw_id = int(left)
            except ValueError:
                continue

            if raw_id not in mapping:
                mapping[raw_id] = idx
                idx += 1

    _LABEL_MAP = mapping
    return mapping


def _map_label(raw_label: int) -> int:
    mapping = _get_label_map()
    if raw_label not in mapping:
        raise KeyError(f"Raw label {raw_label} not found in species_list.txt")
    return mapping[raw_label]

def _load_raw_id_list(path: Path):
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # take the first token before ';' or whitespace
            token = line.split(";", 1)[0].split()[0]
            try:
                raw_id = int(token)
            except ValueError:
                continue
            ids.append(raw_id)
    return ids

def get_with_without_label_sets():
    """
    Returns two sets of *mapped* label indices:
      - with_set: classes that have herbarium-photo pairs
      - without_set: classes that do not
    """
    mapping = _get_label_map()
    with_ids = _load_raw_id_list(CLASS_WITH_PAIRS)
    without_ids = _load_raw_id_list(CLASS_WITHOUT_PAIRS)

    with_set = {mapping[i] for i in with_ids if i in mapping}
    without_set = {mapping[i] for i in without_ids if i in mapping}
    return with_set, without_set

def _parse_train_list(path: Path, root: Path) -> List[Dict]:
    """
    Returns a list of dicts:
    {
        "path": Path,
        "label": int,
        "domain": int,     # 0 = herbarium, 1 = photo
        "rel_path": str
    }
    """
    samples: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label_str = line.split()
            label = _map_label(int(label_str))
            full_path = root / rel_path

            if "herbarium" in rel_path:
                domain = 0
            elif "photo" in rel_path:
                domain = 1
            else:
                # fallback, in case naming is different
                domain = 0

            samples.append(
                {
                    "path": full_path,
                    "label": label,
                    "domain": domain,
                    "rel_path": rel_path,
                }
            )
    return samples


def _parse_test_list_with_groundtruth(
    test_list_path: Path, gt_path: Path, dataset_root: Path
) -> List[Dict]:
    """
    Parse list/test.txt and list/groundtruth.txt.

    * test.txt contains one relative path per line (e.g. "test/1745.jpg").
    * groundtruth.txt may contain either plain class ids ("105951") or
      "path class" pairs ("test/1745.jpg 105951").

    Returns samples ready for HerbFieldDataset.
    """

    rel_paths: List[str] = []
    with test_list_path.open("r") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            rel_paths.append(rel_path)

    gt_entries: List[Tuple[str, int]] = []
    with gt_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                # Legacy format: only the label is stored, rely on test.txt for the path
                gt_entries.append(("", _map_label(int(parts[0]))))
            else:
                rel_from_gt = parts[0]
                label = _map_label(int(parts[-1]))
                gt_entries.append((rel_from_gt, label))

    if len(rel_paths) != len(gt_entries):
        raise ValueError(
            "test.txt and groundtruth.txt length mismatch: "
            f"{len(rel_paths)} vs {len(gt_entries)}"
        )

    samples: List[Dict] = []
    for idx, rel_path in enumerate(rel_paths):
        rel_from_gt, label = gt_entries[idx]
        if rel_from_gt and rel_from_gt != rel_path:
            raise ValueError(
                "Mismatch between test.txt and groundtruth.txt entries at line "
                f"{idx + 1}: '{rel_from_gt}' vs '{rel_path}'"
            )

        full_path = dataset_root / rel_path
        samples.append(
            {
                "path": full_path,
                "label": label,
                "domain": 1,  # test images come from the field/photo domain
                "rel_path": rel_path,
            }
        )

    return samples


class HerbFieldDataset(Dataset):
    """
    Generic dataset for train/test. Returns:
        image: Tensor
        label: int
        domain: int (0 = herbarium, 1 = photo)
        rel_path: str
    """

    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": s["label"],
            "domain": s["domain"],
            "rel_path": s["rel_path"],
        }

def build_train_dataset() -> HerbFieldDataset:
    samples = _parse_train_list(TRAIN_LIST, DATA_ROOT)
    return HerbFieldDataset(samples, transform=build_train_transform())


def build_test_dataset() -> HerbFieldDataset:
    samples = _parse_test_list_with_groundtruth(TEST_LIST, GROUNDTRUTH, DATA_ROOT)
    return HerbFieldDataset(samples, transform=build_eval_transform())



def compute_class_weights(samples: List[Dict]) -> torch.Tensor:
    """
    For class-balanced CE.
    Returns a tensor of shape [NUM_CLASSES].
    """
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float)
    for s in samples:
        counts[s["label"]] += 1.0

    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / torch.log1p(counts)
    return weights
