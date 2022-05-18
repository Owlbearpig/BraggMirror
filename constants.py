from pathlib import Path
import os
import numpy as np

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in os.name:
    data_dir = Path("/home/alex/Data/BraggMirror")
else:
    data_dir = Path("E:\measurementdata\BraggMirror")

