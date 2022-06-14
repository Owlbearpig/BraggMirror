import matplotlib.pyplot as plt
import numpy as np
from constants import *
from datapoint import DataPoint
from functions import find_dp, get_datapoints, find_and_plot_dp, point_grid, average_dp

"""
Expect reflection to be in range: 660 GHz - 545 GHz
x,y = 9, 7.5
"""


if __name__ == "__main__":
    data_dir = top_dir / "msr_220608_Sam1BGOffSub" / "THz" #/ "Sam1MT_daten_linescan0.1mm_5xnebeneinander0.4mm_10Avg_constNormal"
    data_dir = top_dir / "msr_220608_Sam1BGOffSub" / "Scan3" / "Map"

    DataPoint.settings = settings_bg

    all_dps = get_datapoints(dir_=data_dir, dp_class=DataPoint)

    """
    bkgrnd_dp = find_dp(all_dps, x_pos=5, y_pos=8)
    for dp in all_dps:
        continue
        dp.subtract_background(bkgrnd_dp)
    """
    #coord_lst = [(10, 2), (15, 5), (9, 7.5), (8.5, 7.5)]
    #coord_lst = [(9, 7.5), (8.5, 7.5), (6.5, 7.5), (6.5, 8)]
    #coord_lst = [(16.5, 9), (10, 9), (9, 12)]
    coord_lst = [(6, 9), (5.75, 9), (10, 11), (9, 9), (16.5, 9)]
    find_and_plot_dp(all_dps, coord_lst, td=False)

    area_coords = point_grid(square = [[5.75, 6.25], [8.75, 9.25]], rez=0.25)
    area_dps = find_dp(all_dps, area_coords)
    for dp in area_dps:
        print(dp.file_path)
        #dp.plot_fft()
    avg_dp = average_dp(area_dps)
    #avg_dp.plot_fft()
    #plt.show()

    #all_dps = [dp for dp in all_dps if (dp.y_pos > 5)*(dp.x_pos < 11)*(dp.x_pos > 5)]

    rez_x, rez_y = 0.25, 0.25
    x_coords, y_coords = set([point.x_pos for point in all_dps]), set([point.y_pos for point in all_dps])
    x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    len_x, len_y = int(round(abs(x_min-x_max) / rez_x, 0)) + 1, int(round(abs(y_min-y_max) / rez_y, 0)) + 1

    img = np.zeros((len_y, len_x))

    for dp in all_dps:
        x_ind = int(round(abs(dp.x_pos - x_min) / rez_x, 0))
        y_ind = int(round(abs(dp.y_pos - y_min) / rez_y, 0))

        #img[y_ind, x_ind] = dp.get_tof()
        #img[y_ind, x_ind] = dp.get_val_td()
        img[y_ind, x_ind] = dp.get_val_fd()

    img = np.log10(img)

    plt.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect=rez_y/rez_x)
    plt.colorbar()
    plt.title("2D map")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    # plt.savefig('sample_xy_map_p2p.pdf', format='pdf', dpi=1200)
    plt.show()
