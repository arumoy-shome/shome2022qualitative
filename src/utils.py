from aif360.datasets import (
    AdultDataset,
    CompasDataset,
    BankDataset,
    GermanDataset,
    MEPSDataset21,
)

import itertools
import csv

PRIVILEGED_CLASSES_MAP = {
    "adult": {"sex": [["Male"]], "race": [["White"]]},
    "compas": {"sex": [["Female"]], "race": [["Caucasian"]]},
    "bank": {"age": [lambda x: x > 25]},
    "german": {"sex": [["male"]], "age": [lambda x: x > 25]},
    "meps": {"RACE": [["White"]]},
}
DATASET_MAP = {
    "adult": AdultDataset,
    "compas": CompasDataset,
    "bank": BankDataset,
    "german": GermanDataset,
    "meps": MEPSDataset21,
}

memory = {}


def write_csv(filename, rows):
    """Write data to csv file.

    Args:
        filename: Str, path-like name of csv file
        rows: List[Dict], data to save

    Returns:
        None

    """
    with open(filename, "w", newline="") as f:
        header = list(set(itertools.chain(*[row.keys() for row in rows])))
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def train_test_split(dataset_label, protected, features_to_keep):
    dataset = DATASET_MAP[dataset_label]
    full = dataset(
        protected_attribute_names=[protected],
        privileged_classes=PRIVILEGED_CLASSES_MAP[dataset_label][protected],
        features_to_keep=features_to_keep,
    )

    return full.split([0.75], shuffle=True)
