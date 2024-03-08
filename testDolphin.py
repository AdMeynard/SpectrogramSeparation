#%% STFT of a dolphin recording
#
# Copyright (C) 2023 Adrien MEYNARD

# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Author: Adrien MEYNARD
# Email: adrien.meynard@ens-lyon.fr

#%%  Load signal
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as scisig
import Estimation_algorithm

plt.close('all')
plt.rcParams.update({'font.size': 28})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'palatino'

plt.rcParams["ps.usedistiller"] = 'xpdf'

Fs,signal = wavfile.read('data/61025001.wav')

#%% PSD

N = len(signal)
time = np.arange(N)/Fs

duration = (N-1)/Fs

f, (a0, a1) = plt.subplots(2, 1, height_ratios=[1, 3])

a0.plot(time,signal,'k')
plt.xlabel('Time (ms)')
plt.ylabel('Signal (V)')

tf_sig = np.fft.fft(signal) / Fs
frequ = np.arange(N)*Fs/N

freq, pxx = scisig.welch(signal,nperseg=2000,fs=Fs)

a1.loglog(frequ,np.abs(tf_sig)**2/duration,'b',freq,pxx,'r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V$^2$/Hz)')
plt.xlim((freq[1],Fs/2))

#%% STFT

winlength = 512
overlap = np.round(.95*winlength)
fTF,tTF, sTF = scisig.stft(signal,fs=Fs,nperseg=winlength,nfft=1024,noverlap=overlap)


plt.figure()
plt.pcolormesh(tTF,fTF,np.log1p(np.abs(sTF)**2))
plt.ylim(1000,15000)
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)') 

#%% My algo

freqband = np.where((fTF>1000) & (fTF<=15000))[0]
fTF = fTF[freqband]

Sz = np.log1p(np.abs(sTF[freqband,:]))
K = 500
Theta = 0.001
alpha = 0.1 # 0.8
beta  = 0.00001 # 0.0001
gamma  = 0.15
Niter  = 10

Sx,Sy,L,k = Estimation_algorithm.SpectrogramSeparation(Sz,K,Theta,alpha,beta,gamma,Niter)

#%% Figures

save_fig = 'figures'

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::8],fTF[::3],Sz[::3,::8])
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_dolphin.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sx[::3,::3])
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_x_dolphin.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sy[::3,::3])
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_y_dolphin.pdf")
plt.savefig(save_path1, bbox_inches='tight')