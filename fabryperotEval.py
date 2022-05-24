import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from constants import *
from functions import find_dp, extract_phase, average_dps, get_datapoints


def loss_sub(n, freq, T_meas, phase_meas):
    nr, ni = n.real, n.imag

    alpha, omega = 4 * pi * freq * ni / c0, 2 * pi * freq

    fp = 1 / (1 - exp(-alpha * d_sub) * exp(1j * 2 * nr * omega * d_sub / c0) * (n - 1) ** 2 / (n + 1) ** 2)

    T_mod = fp * exp(-alpha * d_sub / 2) * exp(1j * n.real * omega * d_sub / c0) * 4 * n / (n + 1) ** 2

    phase_mod = (n.real - 1) * omega * d_sub / c0

    return (np.abs(T_mod) - np.abs(T_meas)) ** 2 + (phase_meas - phase_mod) ** 2


def loss_sam(n, freq, T_meas, phase_meas, n_sub):
    n1, n2 = n_sub, n
    d1, d2 = d_sub, d_sam

    omega = 2 * pi * freq

    r10, r01, r12, r20 = (n1 - 1) / (n1 + 1), (1 - n1) / (1 + n1), (n1 - n2) / (n1 + n2), (n2 - 1) / (n2 + 1)
    t12, t20, t10 = 2 * n1 / (n1 + n2), 2 * n2 / (n2 + 1), 2 * n1 / (n1 + 1)

    p1, p2, p3 = exp(1j * omega * n1 * d1 / c0), exp(1j * omega * n2 * d2 / c0), exp(1j * omega * d2 / c0)

    fp1, fp2, fp3 = 1 - r10 * r01 * p1 ** 2, 1 - r01 * r12 * p1 ** 2, 1 - r12 * r20 * p2 ** 2

    T_mod = fp1 * t12 * t20 * p2 / (t10 * p3 * fp2 * fp3)

    phase_mod = (n.real - 1) * omega * d2 / c0

    return (np.abs(T_mod) - np.abs(T_meas)) ** 2 + (phase_meas - phase_mod) ** 2


def optimize(loss, freqs, T_meas, *args):
    phase_meas = extract_phase(freqs, T_meas)

    en_plot = False

    rez_nr, rez_ni = 300, 300
    nr_arr, ni_arr = np.linspace(1.0, 3.0, rez_nr), np.linspace(0.01, 0.9, rez_ni)

    nr_res, ni_res = np.zeros_like(freqs), np.zeros_like(freqs)
    for f_idx, f in enumerate(freqs):
        args_idx = [arg[f_idx] if len(arg) > 1 else arg for arg in args]

        grid_vals = np.zeros([rez_nr, rez_ni])
        for i in range(rez_nr):
            for j in range(rez_ni):
                n = nr_arr[i] + 1j * ni_arr[j]
                grid_vals[j, i] = loss(n, freqs[f_idx], T_meas[f_idx], phase_meas[f_idx], *args_idx)

        grid_vals = np.log10(grid_vals)

        if en_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'Residual sum plot freq: {round(f / THz, 3)} (THz)')
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

        g_min_idx = np.argmin(grid_vals)
        min_x, min_y = np.unravel_index(g_min_idx, grid_vals.shape)

        print(f"{np.round(f/THz, 3)} THz, n={nr_arr[min_y]} + {ni_arr[min_x]}j")

        nr_res[f_idx], ni_res[f_idx] = nr_arr[min_y], ni_arr[min_x]

    return nr_res + 1j * ni_res


if __name__ == '__main__':
    ref_points = get_datapoints("Ref")
    sub_points = get_datapoints("Sub")
    sam_points = get_datapoints("Sam")

    ref_dp = ref_points[0]  # average_dps(ref_points[i])
    sub_dp = find_dp(sam_points, x_pos=11.00, y_pos=19.00)
    sam_dp = find_dp(sam_points, x_pos=13.50, y_pos=15.00)

    f_ref, fft_ref = ref_dp.get_f(), ref_dp.get_Y()
    f_sub, fft_sub = sub_dp.get_f(), sub_dp.get_Y()
    f_sam, fft_sam = sam_dp.get_f(), sam_dp.get_Y()

    T_sub = fft_sub / fft_ref
    T_sam = fft_sam / fft_sub

    idx = (f_ref > 0.3) & (f_ref < 1.1)
    freqs, T_sub, T_sam = f_sub[idx] * THz, T_sub[idx], T_sam[idx]

    try:
        n_sub = np.load("n_sub.npy")
    except FileNotFoundError:
        n_sub = optimize(loss_sub, freqs, T_sub)
        np.save("n_sub.npy", n_sub)

    n_sam = optimize(loss_sam, freqs, T_sam, n_sub)

    plt.plot(freqs, n_sub.real, label="n substrate")
    plt.plot(freqs, n_sam.real, label="n sample")
    plt.ylim((1, 3))
    plt.legend()
    plt.show()

    plt.plot(freqs, 4 * pi * n_sub.imag * freqs / (100 * c0), label=r"$\alpha$ substrate")
    plt.plot(freqs, 4 * pi * n_sam.imag * freqs / (100 * c0), label=r"$\alpha$ sample")
    plt.ylabel("absorption coefficient $(cm^-1)$")
    plt.legend()
    plt.show()
