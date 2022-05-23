import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from constants import *
from functions import find_files, find_dp, extract_phase
from p2p_image import DataPoint


def loss(n, freq, T_sub):
    d_sub = 0.711 * mm2m  # sub
    # d_sam = (1.207 - d_sub)*mm2m  # approximate sample thickness (measured), 0.5 fab dimension
    d_sam = 0.500 * mm2m

    nr, ni = n.real, n.imag
    alpha, omega = 4 * pi * freq * ni / c0, 2 * pi * freq

    fp = 1 / (1 - exp(-alpha * d_sub) * exp(1j * 2 * nr * omega * d_sub / c0) * (n - 1) / (n + 1))

    T_mod = fp * exp(-alpha * d_sub / 2) * exp(1j * n.real * omega * d_sub / c0) * 4 * n / (n + 1) ** 2
    #print(T_mod.real, T_sub.real, T_mod.imag, T_sub.imag)

    # return (T_mod.real - T_sub.real) ** 2
    # return (T_mod.imag - T_sub.imag) ** 2
    print(np.abs(T_sub), np.abs(T_mod))
    return (T_mod.real - T_sub.real)**2 + (T_mod.imag - T_sub.imag) ** 2


if __name__ == '__main__':
    ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
    sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
    sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

    # 1: sam 0.5 mm, 2: sub 0.711 mm, 3: sam+sub 1.207 mm (all in mm)

    ref_dp = ref_points[0]
    sub_dp = find_dp(sam_points, x_pos=11.00, y_pos=19.00)

    f_ref, fft_ref = ref_dp.get_f(), ref_dp.get_Y()
    f_sub, fft_sub = sub_dp.get_f(), sub_dp.get_Y()
    # f_sam, fft_sam = sam_dp.get_f(), sam_dp.get_Y()

    idx = (f_ref > 0.3) & (f_ref < 1.1)

    T_sub = fft_sub / fft_ref

    #T_sub = T_sub / np.max(np.abs(T_sub))

    freqs, T_sub = f_sub[idx] * 10 ** 12, T_sub[idx]
    # n0 = 2.6 + 1j * 0.015
    rez_nr, rez_ni = 200, 200
    nr_arr, ni_arr = np.linspace(2.6, 2.6, rez_nr), np.linspace(0.01, 0.1, rez_ni)

    en_plot = False
    nr_res, ni_res = np.zeros_like(freqs), np.zeros_like(freqs)
    for f_idx, f in enumerate(freqs):
        grid_vals = np.zeros([rez_nr, rez_ni])
        for i in range(rez_nr):
            for j in range(rez_ni):
                n = nr_arr[i] + 1j * ni_arr[j]
                grid_vals[j, i] = loss(n, f, T_sub[f_idx])

        # grid_vals = np.log10(grid_vals)

        if en_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('Residual sum plot')
            fig.subplots_adjust(left=0.2)

            extent = [nr_arr[0], nr_arr[-1], ni_arr[0], ni_arr[-1]]
            img = ax.imshow(grid_vals,
                            origin='lower',
                            cmap=plt.get_cmap('jet'),
                            aspect='auto',
                            extent=extent)

            ax.set_xlabel('n real')
            ax.set_ylabel('n imag')

            cbar = fig.colorbar(img)
            cbar.set_label('log10(loss)', rotation=270, labelpad=10)

            plt.show()

        losses = []
        for ni in ni_arr:
            losses.append(loss(2.6 + 1j*ni, f, T_sub[f_idx]))
        plt.plot(ni_arr, losses)
        plt.xlabel("ni arr")
        plt.ylabel("losses")
        plt.show()

        g_min_idx = np.argmin(grid_vals)
        min_x, min_y = np.unravel_index(g_min_idx, grid_vals.shape)

        print(nr_arr[min_y], ni_arr[min_x], loss(nr_arr[min_y] + 1j * ni_arr[min_x], f, T_sub[f_idx]))

        nr_res[f_idx], ni_res[f_idx] = nr_arr[min_y], ni_arr[min_x]

    plt.scatter(freqs, nr_res)
    plt.show()
