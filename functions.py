import matplotlib.pyplot as plt
import numpy as np
from datapoint import DataPoint
from constants import *


def get_datapoints(search_str="", dir_=top_dir, dp_class=DataPoint):
    return [dp_class(file_path) for file_path in find_files(dir_, search_str, ".txt")]


def find_files(top_dir=ROOT_DIR, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results


def find_dp(datapoint_lst, coord_list):
    testing = False
    matches = []
    for coords in coord_list:
        best_match, smallest_diff = None, np.inf
        for dp in datapoint_lst:
            x_pos, y_pos = coords
            diff = abs(dp.x_pos - x_pos)+abs(dp.y_pos - y_pos)
            if diff < smallest_diff:
                best_match, smallest_diff = dp, diff

        if testing:
            print(best_match.file_path)
        matches.append(best_match)

    if len(coord_list) == 1:
        return matches[0]
    else:
        return list(set(matches))


def find_and_plot_dp(datapoint_lst, coords, td=True, **kwargs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    found_dps = find_dp(datapoint_lst, coords)
    for dp in found_dps:
        if td:
            dp.plot_td(**kwargs)
        else:
            dp.plot_fft(ax1, **kwargs)

    ax1.legend()

    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks

    def tick_function(X):
        V = 10**6 * c0 / (X * THz)
        return [f"{np.round(z, 1)}" for z in V]

    ax2 = ax1.twiny()
    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(tick_function(ax2Ticks))
    ax2.set_xlabel(r"Wavelength $(\mu m)$")

    plt.title(f"FFT of {len(found_dps)} selected points")
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    f_min, f_max = kwargs["freq_range"]
    if kwargs["intensity_plot"]:
        plot_type = "Intensity"
    else:
        plot_type = "log"

    plt.savefig(f"FFT_{plot_type}_{len(found_dps)}_points_({round(f_min, 3)}-{round(f_max, 3)})_THz.pdf",
                format="pdf", dpi=1200)

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()


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


def point_grid(square, rez=0.25):
    x_range = np.arange(square[1][0], square[1][1]+rez, rez)
    y_range = np.arange(square[0][0], square[0][1]+rez, rez)

    points = []
    for x in x_range:
        for y in y_range:
            points.append((x, y))
    return points


def average_dp(dp_lst):
    y_avg = np.zeros_like(dp_lst[0].get_y())
    for dp in dp_lst:
        y_avg += dp.get_y()

    avg_data = []
    for ty_tuple in zip(dp_lst[0].get_t(), y_avg / len(dp_lst)):
        avg_data.append(ty_tuple)

    return DataPoint(data=np.array(avg_data))



if __name__ == '__main__':
    from datapoint import do_fft

    square = [[7, 10], [4, 6]]
    print(point_grid(square))

    data_dir = top_dir / "3x3mmRefSquare_Lab2"
    file = str(find_files(data_dir, "2022-05-17T17-32-27.543616", ".txt")[0])
    data = np.loadtxt(file)
    t, y = data[:, 0], data[:, 1]
    t, y = t, y - np.mean(y)
    f, Y = do_fft(t, y)

    plt.plot(f, Y)
    plt.show()
