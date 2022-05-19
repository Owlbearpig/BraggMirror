import matplotlib.pyplot as plt
from constants import *
from functions import do_fft, find_files
from p2p_image import DataPoint



if __name__ == '__main__':
    ref_points = [DataPoint(file) for file in find_files(data_dir, "Ref", ".txt")]
    sub_points = [DataPoint(file) for file in find_files(data_dir, "Sub", ".txt")]
    #sam_points = [DataPoint(file) for file in find_files(data_dir, "Sam", ".txt")]

    # sam 0.5 mm, sub 0.711 mm, sam+sub 1.207 mm
    d = 0.711  # mm

    ref, sample = ref_points[0], sub_points[0]

    freqs, fft_ref = do_fft(ref.t, ref.y)
    freqs_sample, fft_sample = do_fft(sample.t, sample.y)

    idx = (freqs > 0.3) & (freqs < 1.0)

    T = fft_sample/fft_ref

    f = freqs[idx]#*10**(-12)
    phase = np.unwrap(np.angle(T[idx]))
    phase = np.abs(phase)

    p = np.polyfit(f, phase, 1)
    phase -= p[1]

    plt.plot(f, phase, label='phase')
    plt.plot(freqs, p[0]*freqs, label='lin. interpol')
    plt.xlim((0, 1.1))
    plt.ylim((0, 18))
    plt.legend()
    plt.show()

    ri = 1 + c0*phase/(2*np.pi*f*d*10**9)
    plt.figure()
    plt.plot(f, ri)
    plt.ylim(2, 3)
    plt.show()

    alpha = -0.5*d*np.log(np.abs(T[idx])*(ri+1)**2/(4*ri))

    plt.figure()
    plt.plot(f, alpha)
    #plt.ylim(2, 3)
    plt.show()