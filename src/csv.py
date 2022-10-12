"""Module for writing csv files.

This module provides functions to manipulate csv files. This project
utilises csv files to store experimental data under the data/
directory.
"""

import itertools
import csv


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


def new_row(kwargs):
    """Return new row.

    Args:
        kwargs: Dict

    Returns:
        row: Dict

    """
    row = {}
    for k, v in kwargs.items():
        row[k] = v

    return row
