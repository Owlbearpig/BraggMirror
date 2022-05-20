import matplotlib.pyplot as plt
from constants import *
from functions import find_files, find_dp, extract_phase
from p2p_image import DataPoint

if __name__ == '__main__':
    ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
    sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
    sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

    # 1: sam 0.5 mm, 2: sub 0.711 mm, 3: sam+sub 1.207 mm (all in mm)
    d1 = 0.711  # sub
    d2 = 1.207 - d1  # approximate sample thickness (measured), 0.5 fab dimension
    d2 = 0.500

    ref_dp = ref_points[0]
    sub_dp = find_dp(sam_points, x_pos=11.00, y_pos=19.00)
    sam_dp = find_dp(sam_points, x_pos=13.50, y_pos=15.00)

    # ref_dp.plot_td(label=f'ref_dp ({ref_dp.x_pos}, {ref_dp.y_pos})')
    # sub_dp.plot_td(label=f'sub_dp ({sub_dp.x_pos}, {sub_dp.y_pos})')
    # sam_dp.plot_td(label=f'sam_dp ({sam_dp.x_pos}, {sam_dp.y_pos})')
    # plt.legend(); plt.show()

    # ref.plot_fft()
    # sub_dp.plot_fft()
    # sam_dp.plot_fft()
    # plt.legend(); plt.show()

    f_ref, fft_ref = ref_dp.get_f(), ref_dp.get_Y()
    f_sub, fft_sub = sub_dp.get_f(), sub_dp.get_Y()
    f_sam, fft_sam = sam_dp.get_f(), sam_dp.get_Y()

    idx = (f_ref > 0.3) & (f_ref < 1.1)

    T_sub = fft_sub / fft_ref
    T_sam = fft_sam / fft_sub

    f = f_sub[idx]

    phase_sub = extract_phase(f, T_sub[idx])
    phase_sam = extract_phase(f, T_sam[idx])

    n_sub = 1 + c0 * phase_sub / (2 * np.pi * f * d1 * 10 ** 9)
    # n_sam = 1 + c0*phase_sam/(2*np.pi*f*d1*10**9) - d2/(d1*(n_sub-1))
    n_sam = 1 + c0 * phase_sam / (2 * np.pi * f * d2 * 10 ** 9)

    plt.figure()
    plt.plot(f, n_sub, label="n sub")
    plt.plot(f, n_sam, label="n sam")
    plt.xlabel("frequency (THz)")
    plt.ylabel("refractive index")
    plt.legend()
    plt.show()

    fc_sub = (n_sub + 1) ** 2 / (4 * n_sub)
    alpha_sub = -0.5 * np.log(np.abs(T_sub[idx]) * fc_sub) / (d1 * 10**-1)
    fc_sam = n_sam * (n_sub + 1) / ((n_sub + n_sam) * (n_sam + 1))
    alpha_sam = -0.5 * np.log(np.abs(T_sam[idx]) * fc_sam) / (d2 * 10**-1)
    plt.figure()
    plt.plot(f, alpha_sub, label=r"$\alpha_{sub}$")
    plt.plot(f, alpha_sam, label=r"$\alpha_{sam}$")
    # plt.ylim(2, 3)
    plt.ylabel(r"absorption coefficient $\left(cm^{-1}\right)$")
    plt.xlabel("frequency (THz)")
    plt.legend()
    plt.show()
