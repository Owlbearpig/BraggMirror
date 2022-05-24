import matplotlib.pyplot as plt
from constants import *
from functions import find_files, find_dp, extract_phase, get_datapoints
from numpy import exp
from scipy.optimize import curve_fit


if __name__ == '__main__':
    d1 = d_sub
    d2 = d_sam

    ref_points = get_datapoints("Ref")
    sub_points = get_datapoints("Sub")
    sam_points = get_datapoints("Sam")

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

    n_sub = 1 + c0 * phase_sub / (2 * np.pi * f * THz * d1)
    # n_sam = 1 + c0*phase_sam/(2*np.pi*f*d1*10**9) - d2/(d1*(n_sub-1))
    n_sam = 1 + c0 * phase_sam / (2 * np.pi * f * THz * d2)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(f, n_sub, label="n substrate (measured)")
    ax1.plot(f, n_sam, label="n polymer (measured)")
    ax1.set_ylim((0.9, 3))
    ax1.set_yticks(np.arange(0.9, 3, 0.1))
    ax1.set_xlabel("Frequency (THz)")
    ax1.set_ylabel("Refractive index")
    ax1.grid(True)
    ax1.legend()

    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks

    def tick_function(X):
        V = 10**6 * c0 / (X * THz)
        return [f"{int(z)}" for z in V]

    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(tick_function(ax2Ticks))
    ax2.set_xlabel(r"Wavelength $(\mu m)$")

    plt.title("Refractive index")
    plt.savefig('refr_index.pdf', format='pdf', dpi=1200, bbox_inches="tight")
    plt.show()

    def func(x, a, b, c):
        return a * x**2 + b * x + c

    fc_sub = (n_sub + 1) ** 2 / (4 * n_sub)
    alpha_sub = -2 * np.log(np.abs(T_sub[idx]) * fc_sub) / d1
    # print(100*alpha_sub*c0/(4*np.pi*(fc_sub*10**12)))
    fc_sam = (n_sam + n_sub) * (n_sam + 1) / (2 * n_sam * (n_sub + 1))
    alpha_sam = -2 * np.log(np.abs(T_sam[idx]) * fc_sam) / d2

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    popt, _ = curve_fit(func, f, alpha_sub / 100)
    sub_label = r'$\alpha_{sub}$ fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)
    ax1.plot(f, func(f, *popt), 'r-', label=sub_label)
    ax1.plot(f, alpha_sub / 100, label=r"$\alpha_{sub}$ measured")

    popt, _ = curve_fit(func, f[8:], alpha_sam[8:] / 100)
    sam_label = r'$\alpha_{polymer}$ fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)
    ax1.plot(f, func(f, *popt), 'r-', label=sam_label, color="black")
    ax1.plot(f, alpha_sam / 100, label=r"$\alpha_{polymer}$ measured")

    # plt.ylim(2, 3)
    ax1.set_ylabel(r"Absorption coefficient $\left(cm^{-1}\right)$")
    ax1.set_xlabel(r"Frequency $\nu$ (THz)")
    ax1.legend()
    ax1.grid(True)

    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks

    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(tick_function(ax2Ticks))
    ax2.set_xlabel(r"Wavelength $(\mu m)$")

    plt.title(r"$\alpha$ substrate and sample. Fits: $a\nu^2+b\nu+c$")
    plt.savefig('abs_coeff_sample_only.pdf', format='pdf', dpi=1200, bbox_inches="tight")

    plt.show()

    k_sam = c0*alpha_sam / (4*pi*f*THz)
    # print(k_sam)
    """
    plt.plot(f, phase_sub, label="phase sub")
    plt.plot(f, extract_phase(f, T_mod), label="phase model")
    plt.legend()
    plt.show()
    """