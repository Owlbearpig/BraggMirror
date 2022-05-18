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
        self.x_pos, self.y_pos = None, None

        self.t, self.y = None, None
        self.val = None

        self.set_metadata()
        self.format_data()

    def set_metadata(self):
        split_path = self.file_path.name.split("_")
        self.time = datetime.strptime(split_path[0], "%Y-%m-%dT%H-%M-%S.%f-")
        match_str = r"-?\d{1,3}.\d{1,3}"
        self.x_pos = float(re.match(match_str, split_path[-2]).group(0))
        self.y_pos = float(re.match(match_str, split_path[-1]).group(0))

    def format_data(self):
        data = np.loadtxt(self.file_path)
        t, y = data[:, 0], data[:, 1]
        self.t, self.y = t, y - np.mean(y)
        self.val = np.max(np.abs(y))

    def plot_td(self):
        plt.plot(self.t, self.y)
        plt.show()


if __name__ == '__main__':

    ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
    sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
    sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

    for sam_point in sam_points:
        if (abs(sam_point.x_pos - 12.75) < 0.25) and (abs(sam_point.y_pos - 13.00) < 0.25):
            print(sam_point.file_path)
            sam_point.plot_td()

    x_coords, y_coords = set([point.x_pos for point in sam_points]), set([point.y_pos for point in sam_points])
    x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    img = np.zeros((len(x_coords), len(y_coords)))

    for sam_point in sam_points:
        x_ind = int((sam_point.x_pos - x_min) / 0.25)
        y_ind = int((sam_point.y_pos - y_min) / 0.25)

        img[x_ind, y_ind] = sam_point.val

    plt.imshow(np.log(img), extent=[x_min, x_max, y_min, y_max], aspect=0.5)
    plt.xlabel("x (Owis) (mm)")
    plt.ylabel("y (TMCL) (mm)")
    plt.show()
