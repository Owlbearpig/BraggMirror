from constants import data_dir
from functions import find_files
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import re


class DataPoint:

    def __init__(self, file_path):
        self.file_path = file_path
        self.time = None
        self.t, self.y = None, None
        self.x_pos, self.y_pos = None, None

        self.set_metadata()
        self.format_data()

    def set_metadata(self):
        split_path = self.file_path.name.split("_")
        self.time = datetime.strptime(split_path[0], "%Y-%m-%dT%H-%M-%S.%f-")
        match_str = r"-?\d{1,3}.\d{1,3}"
        self.x_pos = re.match(match_str, split_path[-2]).group(0)
        self.y_pos = re.match(match_str, split_path[-1]).group(0)

    def format_data(self):
        data = np.loadtxt(self.file_path)
        t, y = data[:, 0], data[:, 1]
        self.t, self.y = t, y - np.mean(y)

    def plot_td(self):
        plt.plot(self.t, self.y)
        plt.show()


ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

for point in sam_points:
    print(point.x_pos, point.y_pos)
