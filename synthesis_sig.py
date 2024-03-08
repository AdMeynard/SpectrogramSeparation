#%%  Load signal
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import Estimation_algorithm
import random
from sklearn.decomposition import NMF

plt.close('all')

plt.rcParams.update({'font.size': 28})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'palatino'

plt.rcParams["ps.usedistiller"] = 'xpdf'


#%% Synthese signal

N = 2**14
Fs = N
time = np.arange(N)/Fs
duration = (N-1)/Fs

x = np.zeros(N)

D = 9/Fs # peak width
indpeak = 0
while indpeak<(N-1000):
    theta = random.randint(574,1000)
    indpeak = indpeak + theta  
    xtmp = 0.5+0.5*np.cos(2*np.pi*(time-time[indpeak])/D)
    xtmp = xtmp*(time>=(time[indpeak]-D/2))*(time<=(time[indpeak]+D/2))
    x =  x + xtmp

phi_y = time+duration/(8*np.pi**2)*np.sin(2*np.pi*time/duration) - (time/4/np.pi/duration)*np.cos(2*np.pi*time/duration)
f0 = 1500
y = 0.04*np.cos(2*np.pi*f0*phi_y)

signal = x+y


#%% STFT

winlength = 512
overlap = np.round(.95*winlength)
fTF,tTF, sTF = scisig.stft(signal,fs=Fs,nperseg=winlength,nfft=1024,noverlap=overlap)


plt.figure()
plt.plot(time,signal,'k')

plt.figure()
plt.pcolormesh(tTF,fTF,np.abs(sTF)**2)
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim((0,5000))

#%% Spectrogram Separation

Sz = np.abs(sTF)**2
K = 300
Theta = 0.001
alpha = 10
beta  = 0.000001
gamma  = 0.15
Niter  = 10

Sx,Sy,L,k = Estimation_algorithm.SpectrogramSeparation(Sz,K,Theta,alpha,beta,gamma,Niter)

residual = Sz-(Sx+Sy)

#%% NMF

model = NMF(n_components=2, tol=5e-3)

cols = model.fit_transform(Sz)
rows = model.components_

Sx_est = np.outer(cols[:,0], rows[0,:])
Sy_est = np.outer(cols[:,1], rows[1,:])

#%% Plot figures
save_fig = 'figures'

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sz[::3,::3])
plt.ylim((0,5000))
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_mixture.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sx[::3,::3])
plt.ylim((0,5000))
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_x.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sy[::3,::3])
plt.ylim((0,5000))
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_y.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.plot(L)
plt.xlabel('Iteration')
plt.ylabel('Cost Function')

save_path1 = os.path.join(save_fig, "Cost_function.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],residual[::3,::3])
plt.colorbar()
plt.ylim((0,5000))
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

save_path1 = os.path.join(save_fig, "Residuals.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sx_est[::3,::3])
plt.ylim((0,5000))
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_x_nmf.pdf")
plt.savefig(save_path1, bbox_inches='tight')

plt.figure(figsize=(30,10))
plt.pcolormesh(tTF[::3],fTF[::3],Sy_est[::3,::3])
plt.ylim((0,5000))
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

save_path1 = os.path.join(save_fig, "Spectrogram_y_nmf.pdf")
plt.savefig(save_path1, bbox_inches='tight')
