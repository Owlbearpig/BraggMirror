from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from scipy.constants import pi

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in os.name:
    top_dir = Path("/home/alex/Data/BraggMirror")
else:
    top_dir = Path("E:\measurementdata\BraggMirror")
    try:
        os.scandir(top_dir)
    except FileNotFoundError:
        top_dir = Path(r"C:\Users\Laptop\Desktop\BraggMirror")

mm2m = 10 ** -3
THz = 10 ** 12

d_sub = 0.711 * mm2m  # sub
# d_sam = (1.207 - d_sub)*mm2m  # approximate sample thickness (measured), 0.5 fab dimension
d_sam = 0.500 * mm2m

settings_bg = {"regex": r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.\d{6})|(?<=\[)(.+?)(?=\])",
               "enable_preprocessing": 1}

settings_refsq = {"regex": r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.\d{6})|([0-9]*[.]?[0-9]+) ",
                  "enable_preprocessing": 1}

# 2: sam 0.5 mm, 1: sub 0.711 mm, 3: sam+sub 1.207 mm (all in mm)
