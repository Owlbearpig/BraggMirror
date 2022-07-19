import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from datapoint import do_fft

path_T = r"E:\measurementdata\BraggMirror\3x3mmRefSquare_Lab2\2022-05-17T16-18-55.491982-_1percentRH_100avg-Ref-X_-10.000 mm-Y_10.000 mm.txt"
path_R = r"E:\measurementdata\BraggMirror\msr_220608_Sam1BGOffSub\Scan3\Map\2022-06-14T16-30-03.316298-BG1-[1263]-[16.5,9.0,0.0]-[1.0,0.0,0.0,0.0]-delta[0.007mm-0.0deg]-avg100.txt"

data_T = np.loadtxt(path_T)
data_R = np.loadtxt(path_R)

# Create two subplots and unpack the output array immediately
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

t, y = data_R[:, 0], data_R[:, 1]
ax1.plot(t, y)
ax1.set_title('Time domain')
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel("Amplitude (arb. unit)")

fig.suptitle('Reflection setup antennas (ambient air)')

f, Y = do_fft(t, y)

ax2.plot(f, 20*np.log10(np.abs(Y)))
ax2.set_xlabel("Frequency (THz)")
ax2.set_xlim((-0.05, 7.5))
ax2.set_ylabel(r"Amplitude $[20*log_{10}\left(|FFT|\right)]$ (dB)")
ax2.set_title('Frequency domain')


plt.show()
