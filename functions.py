import matplotlib.pyplot as plt

from constants import *


def find_files(top_dir=ROOT_DIR, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results


def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]


if __name__ == '__main__':
    file = str(find_files(data_dir, "2022-05-17T17-32-27.543616", ".txt")[0])
    data = np.loadtxt(file)
    t, y = data[:, 0], data[:, 1]
    t, y = t, y - np.mean(y)
    f, Y = do_fft(t, y)

    plt.plot(f, Y)
    plt.show()