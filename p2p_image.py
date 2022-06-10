from functions import find_files
from constants import top_dir
from datapoint import DataPoint
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_dir = top_dir / "3x3mmRefSquare_Lab2"

    ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
    sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
    sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

    x_coords, y_coords = set([point.x_pos for point in sam_points]), set([point.y_pos for point in sam_points])
    x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    img = np.zeros((len(y_coords), len(x_coords)))

    for sam_point in sam_points:
        x_ind = int((sam_point.x_pos - x_min) / 0.25)
        y_ind = int((sam_point.y_pos - y_min) / 0.25)

        img[y_ind, x_ind] = sam_point.get_val_td()

    plt.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect=1)
    #plt.imshow(img)
    plt.title("Peak to peak")
    plt.xlabel("x (Owis) (mm)")
    plt.ylabel("y (TMCL) (mm)")
    plt.savefig('sample_xy_map_p2p.pdf', format='pdf', dpi=1200)
    plt.show()
