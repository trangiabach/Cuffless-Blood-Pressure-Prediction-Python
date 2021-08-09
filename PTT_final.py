import pandas as pd
import numpy as np
import scipy.signal
from scipy.fft import fft, ifft
import mat73
import math
import csv

sample_file = mat73.loadmat(f'Part_{4}.mat')['Part_4']
rows = []


def movsum(lis , w):
    return np.around([sum(lis[i-(w-1):i+1]) if i>(w-1) else sum(lis[:i+1])  for i in range(len(lis))], 4)

def movmax(datas,k):
    result = np.empty_like(datas)
    start_pt = 0
    end_pt = int(np.ceil(k/2))

    for i in range(len(datas)):
        if i < int(np.ceil(k/2)):
            start_pt = 0
        if i > len(datas) - int(np.ceil(k/2)):
            end_pt = len(datas)
        result[i] = np.max(datas[start_pt:end_pt])
        start_pt += 1
        end_pt +=1
    return result

for d in range(3000):
    sample = sample_file[d]
    O1P = sample[0][0:1000]
    BP = sample[1][0:1000]
    O1E = sample[2][0:1000]

    Fs = 125
    Ts = 1/125

    T = np.arange(start=0, stop=7.999, step=0.008)
    W1 = 0.5/62.5
    W2 = 5/62.5

    b, a = scipy.signal.butter(3, [W1, W2], btype = 'bandpass')
    FP = scipy.signal.filtfilt(b,a, O1P , padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    Fy = np.gradient(FP)
    for j in range(1000):
        if Fy[j] <= 0:
            Fy[j] = 0
    T1 = movsum(Fy, 3)


    W1 = 0.5/62.5
    W2 = 40/62.5
    b, a = scipy.signal.butter(3, [W1, W2], btype = 'bandpass')
    FP1 = scipy.signal.filtfilt(b,a, O1E , padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    A = scipy.signal.detrend(FP1)
    E = scipy.signal.detrend(FP)

    D = movmax(T1, 3)
    
    loc1, pk1= scipy.signal.find_peaks(D)
    pk1 = []

    for location in loc1:
	    pk1.append(D[location])
    
    h = np.zeros(1000)

    for i in range(len(loc1)):
        h[loc1[i]] = 1
    
    j = 0
    
    for i in range(len(h)):
        if h[i] == 1:
            h[i] = pk1[j]
            j = j + 1
            print(i) 
    print(h[936])
    C = np.correlate(A, h, 'full')
    Lag = np.arange(start=-999, stop= 1000, step=1)
    VA = max(abs(C))
    print(VA)
    I = 0
    for i in range(len(abs(C))):
        if abs(C)[i] == VA:
            I = i
    Diff = Lag[I]/Fs
    rows.append([np.real(abs(Diff))])

    break

with open('ptt_newpart4_python.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	for row in rows:
		writer.writerow(row)