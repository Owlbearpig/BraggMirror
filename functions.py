import matplotlib.pyplot as plt
import numpy as np
from datapoint import DataPoint
from constants import *


def get_datapoints(search_str=""):
    return [DataPoint(file) for file in find_files(data_dir, search_str, ".txt")]

def find_files(top_dir=ROOT_DIR, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results


def find_dp(datapoint_lst, x_pos, y_pos):
    for sam_point in datapoint_lst:
        if (abs(sam_point.x_pos - x_pos) < 0.25) and (abs(sam_point.y_pos - y_pos) < 0.25):
            print(sam_point.file_path)

            return sam_point


def extract_phase(f, T, plot=False):
    phase = np.unwrap(np.angle(T))
    phase = np.abs(phase)

    p = np.polyfit(f, phase, 1)
    phase -= p[1]

    if plot:
        plt.plot(f, phase, label='phase')
        plt.plot(f, p[0] * f, label='lin. interpol')
        plt.xlim((0, 1.1))
        # plt.ylim((0, 18))
        plt.legend()
        # plt.show()

    return phase


def average_dps(dp_lst):
    y_avg = np.zeros_like(dp_lst[0].get_y())
    for dp in dp_lst:
        y_avg += dp.get_y()

    avg_data = []
    for ty_tuple in zip(dp_lst[0].get_t(), y_avg/len(dp_lst)):
        avg_data.append(ty_tuple)

    return DataPoint(data=np.array(avg_data))


if __name__ == '__main__':
    from datapoint import do_fft

    file = str(find_files(data_dir, "2022-05-17T17-32-27.543616", ".txt")[0])
    data = np.loadtxt(file)
    t, y = data[:, 0], data[:, 1]
    t, y = t, y - np.mean(y)
    f, Y = do_fft(t, y)

    plt.plot(f, Y)
    plt.show()
