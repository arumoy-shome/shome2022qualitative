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
