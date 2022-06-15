from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from THz.preprocessing import butter_highpass_filter

plt.rcParams.update({"font.size": 16,
                     "axes.grid": True})

"""


"""
def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]


class DataPoint:
    settings = {"regex": r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.\d{6})-([^-]*)-|([0-9]*[.]?[0-9]+)",
                "enable_preprocessing": False,
                "fft_sum_range": (0.545, 0.660)}

    def __init__(self, file_path=None, data=None):
        self.label = None
        self.file_path = file_path
        self.time = None
        self.x_pos, self.y_pos, self.z_pos = None, None, None

        self._t, self._y = None, None
        self._f, self._Y = None, None
        self._val_td, self._val_fd, self._tof = None, None, None
        self._data = None

        self.set_metadata()

        if data is not None:
            self._data = data
            self._format_data()

    def __eq__(self, other):
        return self.file_path == other.file_path

    def __hash__(self):
        return hash(('file_path', self.file_path))

    def set_metadata(self):
        if self.file_path is None:
            self.label = "avg. point"
            return
        
        matches = [match.group() for match in re.finditer(DataPoint.settings['regex'], self.file_path.name)]
        self.time = datetime.strptime(matches[0], "%Y-%m-%dT%H-%M-%S.%f")

        if self.file_path.name.split("-")[-1][0:3] == "avg":
            self.x_pos, self.y_pos, self.z_pos = eval(matches[2])
        else:
            self.x_pos, self.y_pos = float(matches[-2]), float(matches[-1])

        self.label = f"x={self.x_pos} mm, y={self.y_pos} mm"  # self.file_path.stem
        
    def _preprocess(self, data):
        t, y = data[:, 0], data[:, 1]
        y = y - np.mean(y[0:25])
        fs = 1 / np.float(np.mean(np.diff(t)))
        nyq = 0.5 * fs
        low = 0.5 / nyq

        y = butter_highpass_filter(y, low, fs, order=5, plot=False)

        data[:, 0], data[:, 1] = t, y

        return data

    def _format_data(self):
        if (self._data is None) and self.file_path:
            self._data = np.loadtxt(self.file_path)
            if DataPoint.settings['enable_preprocessing']:
                self._data = self._preprocess(self._data)
        elif self._data is not None:
            pass
        else:
            return

        self._t, self._y = self._data[:, 0], self._data[:, 1]
        self._val_td = np.max(np.abs(self._y))
        self._tof = self._t[np.argmax(np.abs(self._y))]

        self._f, self._Y = do_fft(self._t, self._y)

        f_min, f_max = DataPoint.settings["fft_sum_range"][0], DataPoint.settings["fft_sum_range"][1]
        f_min, f_max = np.argmin(np.abs(self._f - f_min)),  np.argmin(np.abs(self._f - f_max))
        self._val_fd = np.sum(np.abs(self._Y[f_min:f_max]))

    def plot_td(self, **kwargs):
        self._format_data()
        kwargs["label"] = self.file_path.stem

        plt.plot(self._t, self._y, **kwargs)
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (arb. u.)")

    def plot_fft(self, ax=None, **kwargs):
        self._format_data()
        f_min, f_max = kwargs["freq_range"][0] * 0.95, kwargs["freq_range"][1] * 1.05
        idx = (self._f > f_min) & (self._f < f_max)
        ax.set_xlim((f_min, f_max))

        if kwargs["intensity_plot"]:
            y_label = "Intensity $[|FFT|^2]$ (Arb. u.)"
            Y_db = np.abs(self._Y)**2
            met_t, bragg_t = 90, 10
        else:
            y_label = r"Amplitude $[20*log_{10}\left(|FFT|\right)]$ (dB)"
            Y_db = 20 * np.log10(np.abs(self._Y))
            met_t, bragg_t = 20, 10

        test_val = Y_db[np.argmin(np.abs(self._f - 0.6))]

        if test_val > met_t:
            point_type = "Metal reference"
        elif (test_val > bragg_t) and (test_val < met_t):
            point_type = "BraggMirror"
        else:
            point_type = "Background"

        label = self.label + f" ({point_type})"
        if ax is not None:
            ax.plot(self._f[idx], Y_db[idx], label=label)
            ax.set_xlabel("Frequency (THz)")
            ax.set_ylabel(y_label)
        else:
            plt.plot(self._f[idx], Y_db[idx], label=label)

        plt.xlabel("Frequency (THz)")

    def get_t(self):
        if self._t is not None:
            return self._t
        else:
            self._format_data()
            return self._t

    def get_y(self):
        if self._y is not None:
            return self._y
        else:
            self._format_data()
            return self._y

    def get_f(self):
        if self._f is not None:
            return self._f
        else:
            self._format_data()
            return self._f

    def get_Y(self):
        if self._Y is not None:
            return self._Y
        else:
            self._format_data()
            return self._Y

    def get_val_td(self):
        if self._val_td is not None:
            return self._val_td
        else:
            self._format_data()
            return self._val_td

    def get_val_fd(self):
        if self._val_fd is not None:
            return self._val_fd
        else:
            self._format_data()
            return self._val_fd

    def get_tof(self):
        if self._tof is not None:
            return self._tof
        else:
            self._format_data()
            return self._tof

    def subtract_background(self, background_dp):
        self._format_data()
        self._data[:, 1] -= background_dp.get_y()
        self._format_data()
