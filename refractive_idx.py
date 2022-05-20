import matplotlib.pyplot as plt
from constants import *
from functions import do_fft, find_files, find_dp
from p2p_image import DataPoint


def extract_phase(f, T, plot=False):
    phase = np.unwrap(np.angle(T))
    phase = np.abs(phase)

    p = np.polyfit(f, phase, 1)
    phase -= p[1]

    if plot:
        plt.plot(f, phase, label='phase')
        plt.plot(f, p[0] * f, label='lin. interpol')
        plt.xlim((0, 1.1))
        #plt.ylim((0, 18))
        plt.legend()
        #plt.show()

    return phase


if __name__ == '__main__':
    ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
    sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
    sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

    # 1: sam 0.5 mm, 2: sub 0.711 mm, 3: sam+sub 1.207 mm
    d2 = 0.711 # mm # sub
    d1 = 1.207 - d2  # ~sam (measured), 0.5 is theory

    ref_dp, sub_dp, sam_dp = ref_points[0], sub_points[0], find_dp(sam_points, x_pos=13, y_pos=12)

    #ref.plot_td()
    #sub_dp.plot_td()
    #sam_dp.plot_td()

    #ref.plot_fft()
    #sub_dp.plot_fft()
    #sam_dp.plot_fft()

    f_ref, fft_ref = do_fft(ref_dp.t, ref_dp.y)
    f_sub, fft_sub = do_fft(sub_dp.t, sub_dp.y)
    f_sam, fft_sam = do_fft(sam_dp.t, sam_dp.y)

    idx = (f_ref > 0.3) & (f_ref < 1.0)

    T_sub = fft_sub/fft_ref
    T_sam = fft_sam/fft_ref

    f = f_sub[idx]

    phase_sub = extract_phase(f, T_sub[idx])
    phase_sam = extract_phase(f, T_sam[idx])

    n_sub = 1 + c0*phase_sub/(2*np.pi*f*d2*10**9)
    n_sam = 1 + c0*phase_sam/(2*np.pi*f*d1*10**9) - d2/(d1*(n_sub-1))

    plt.figure()
    plt.plot(f, n_sub, label="n substrate")
    plt.plot(f, n_sam, label="n sample")
    #plt.ylim(2, 3)
    plt.legend()
    plt.show()

    """
    alpha_sub = -0.5*d2*np.log(np.abs(T_sub[idx])*(n_sub+1)**2/(4*n_sub))

    plt.figure()
    plt.plot(f, alpha_sub)
    #plt.ylim(2, 3)
    plt.show()
    """

