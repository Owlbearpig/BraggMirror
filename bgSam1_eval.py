import matplotlib.pyplot as plt
from constants import *
from datapoint import DataPoint
from functions import find_dp, extract_phase, get_datapoints, find_files



if __name__ == '__main__':
    data_dir = top_dir / "msr_220608_Sam1BG" / "THz"

    DataPoint.reg_str = r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.\d{6})|(?<=\[)(.+?)(?=\])"

    all_dps = get_datapoints(dir_=data_dir, dp_class=DataPoint)



