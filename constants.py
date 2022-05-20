from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in os.name:
    data_dir = Path("/home/alex/Data/BraggMirror")
else:
    data_dir = Path("E:\measurementdata\BraggMirror")
    try:
        os.scandir(data_dir)
    except FileNotFoundError:
        data_dir = Path(r"C:\Users\Laptop\Desktop\BraggMirror")
