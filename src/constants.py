""" Constants across module.
"""

from typing import List

PATH_TO_DATA_DIR: str = "../src/data"
DATASET_NAME: str = "auto-mpg.data"

COLUMN_NAMES: List[str] = [
    "mpg", 
    "cylinders", 
    "displacement", 
    "horsepower", 
    "weight", 
    "acceleration", 
    "year",
    "origin",
]
