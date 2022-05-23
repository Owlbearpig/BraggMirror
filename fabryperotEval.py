import matplotlib.pyplot as plt
from numpy import exp
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
    # sam_dp = find_dp(sam_points, x_pos=13.50, y_pos=15.00)

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
    # f_sam, fft_sam = sam_dp.get_f(), sam_dp.get_Y()

    idx = (f_ref > 0.3) & (f_ref < 1.1)

    T_sub = fft_sub / fft_ref



    fp = 1 - exp(-alpha*d)*exp(1j*2*n.real*omega*d/c0)*(n-1)/(n+1)

    denum =

    f = f_sub[idx]


def loss(n, params):
    alpha = n
    fp_denum = 1 - exp(-alpha*d)*exp(2*1j*n.real*omega*d/c0)*(n-1)/(n+1)
    fp = 1 / fp_denum
    T_mod = exp(-alpha*d/2)*exp(1j*n.real*omega*d/c0)*4*n/(n+1)**2

    return


