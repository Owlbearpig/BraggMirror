from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import re


def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]


class DataPoint:
    reg_str = None

    def __init__(self, file_path=None, data=None):
        self.file_path = file_path
        self.time = None
        self.x_pos, self.y_pos = None, None

        self._t, self._y = None, None
        self._f, self._Y = None, None
        self._val_td, self._val_fd = None, None
        self._data = None

        self.set_metadata()

        if data is not None:
            self._data = data
            self._format_data()

    def set_metadata(self):
        if self.file_path is None:
            return

        match = re.match(DataPoint.reg_str, self.file_path.name)
        print(match.groups())

        split_path = self.file_path.name.split("_")
        self.time = datetime.strptime(split_path[0], "%Y-%m-%dT%H-%M-%S.%f-")
        match_str = r"-?\d{1,3}.\d{1,3}"

        self.x_pos = float(re.match(match_str, split_path[-2]).group(0))
        self.y_pos = float(re.match(match_str, split_path[-1]).group(0))

    def _format_data(self):
        if self._data is None and self.file_path:
            self._data = np.loadtxt(self.file_path)
        elif self._data is not None:
            pass
        else:
            return

        t, y = self._data[:, 0], self._data[:, 1]
        self._t, self._y = t, y - np.mean(y)
        self._val_td = np.max(np.abs(self._y))

        self._f, self._Y = do_fft(self._t, self._y)
        f_min, f_max = np.argmin(np.abs(self._f - 0.9)), np.argmin(np.abs(self._f - 1.2))
        self._val_fd = np.sum(np.abs(self._Y[f_min:f_max]))

    def plot_td(self, **kwargs):
        self._format_data()
        plt.plot(self._t, self._y, **kwargs)
        plt.xlabel("time (ps)")
        plt.ylabel("amplitude (arb. u.)")

    def plot_fft(self, **kwargs):
        self._format_data()

        idx = (self._f > 0.3) & (self._f < 1.0)

        plt.plot(self._f[idx], np.log10(np.abs(self._Y))[idx], **kwargs)
        plt.title(self.file_path.stem)

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
