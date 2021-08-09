import pandas as pd
import numpy as np
import scipy.signal
from scipy.fft import fft, ifft
import mat73
import math
import csv

sample_file = mat73.loadmat(f'Part_{4}.mat')['Part_4']
rows = []

for d in range(30):
    sample = sample_file[d]
    O1P = sample[0][0:1000]
    BP = sample[1][0:1000]
    O1E = sample[2][0:1000]

    Fy = np.gradient(O1P)
	
    Fy1 = np.gradient(Fy)

    F = np.concatenate((O1P, Fy1, O1E))

    Ts=1/125

    T = np.arange(start=0, stop=7.999, step=0.008)

    loc, pk= scipy.signal.find_peaks(O1P)
    pk = []

    for location in loc:
    	pk.append(O1P[location])
	
    PPG1= max(O1P)-O1P

    loc1, pk1= scipy.signal.find_peaks(PPG1,  height = 0)
    pk1 = []

    for location in loc1:
	    pk1.append(PPG1[location])
    
    sys_time = 0
    for i in range(5):
        sys_time = sys_time + T[loc[i]] + T[loc1[i]]
    sys_time = sys_time/5

    dias_time = 0
    for i in range(5):
        dias_time = dias_time + T[loc[i+1]] + T[loc1[i]]
    dias_time = dias_time/5

    v = [0.1,0.25,0.33,0.5,0.66,0.75]

    ppg_21_st = []
    ppg_21_dt = []

    for j in range(6):
        for i in range(loc1[0], loc[0] + 1):
            if O1P[i] >= (v[j]*pk[0] + pk1[0]):
                a = i
                break

        for i in range(loc[0], loc1[1] + 1):
            if O1P[i] <= (v[j]*pk[0] + pk1[0]):
                b = i
                break
        
        ppg_21_st.append((loc[0]-a)*0.008)
        ppg_21_dt.append((b-loc[0])*0.008)
    

    loc2, pk2= scipy.signal.find_peaks(O1E, height= 0.6)
    pk2 = []

    for location in loc2:
	    pk2.append(O1E[location])
    
    loc3, pk3= scipy.signal.find_peaks(Fy1,  height =0.003)
    pk3 = []

    for location in loc3:
	    pk3.append(Fy1[location])
    
    m = 1
    n = loc2.shape[0]
    x = 1
    y = loc3.shape[0]

    P1 = []
    for location in loc2:
        P1.append(T[location])
    P = []
    for location in loc3:
        P.append(T[location])

    P11 = P1
    P2 = P
    ptt = 0
    temp = min([y,n])
    range_ = min([temp, 5])

    for i in range(range_):
        ptt = ptt + abs(P2[i] - P11[i])

    if range_ !=0:
        ptt = ptt/range_
    else:
        ptt = None

    lr = 1
    lr1 = loc1.shape[0]

    rationum = 0
    ratioden = 0
    ih = 0
    il = 0

    for i in range(0, lr1 - 1):
        rationum = rationum + pk[i]
        ratioden = ratioden + pk1[i]

    ih = rationum/(lr1-1)
    il = ratioden/(lr1-1)

    PIR = ih/il
    RR = np.diff(P1, axis = 0)
    HR = 60./RR
    hrfinal = 0

    lr = 1
    lr1 = HR.shape[0]

    tlr1 = lr1

    for i in range(tlr1):
        t = HR[i]
        if t <= 30 or t>=200:
            tlr1 = tlr1 - 1
        else:
            hrfinal = hrfinal + HR[i]

    if tlr1 !=0:
        hrfinal = hrfinal/tlr1
    else:
        hrfinal = 0


    Yy = fft(O1P)
    Z = Yy[0]
    Yy[0] = 0
    S = np.real(ifft(Yy))


    loc4, pk4= scipy.signal.find_peaks(S)
    pk4 = []

    for location in loc4:
        pk4.append(S[location])

    loc5, pk5= scipy.signal.find_peaks(BP)
    pk5 = []

    for location in loc5:
        pk5.append(BP[location])

    lr = 1
    lr1 = loc4.shape[0]

    iftmax = 0
    for i in range(0, lr1-1):
        iftmax = iftmax + pk4[i]

    meu = iftmax/(lr1-1)

    if hrfinal != None and (1060*hrfinal/meu) >=0:
        alpha = il*math.sqrt(1060*hrfinal/meu)
    else:
        alpha = None

    BP1 = max(BP)-BP

    loc6, pk6= scipy.signal.find_peaks(BP1)
    pk6 = []

    for location in loc6:
        pk6.append(BP1[location])

    lr = 1
    lr1 = loc5.shape[0]
    bpmax = 0
    for i in range(0, lr1-1):
        bpmax = bpmax + pk5[i]
    bpmax = bpmax/(lr1-1)

    lr = 1
    lr1 = loc6.shape[0]
    bpmin = 0
    for i in range(0, lr1-1):
        bpmin = bpmin + pk6[i]
    bpmin = bpmin/(lr1-1)

    # rows.append(np.around([ppg_21_dt[0], ppg_21_st[0]+ppg_21_dt[0] ,ppg_21_dt[0]/ppg_21_st[0], ppg_21_dt[1], ppg_21_st[1]+ppg_21_dt[1], ppg_21_dt[1]/ppg_21_st[1], ppg_21_dt[2], ppg_21_st[2]+ppg_21_dt[2], ppg_21_dt[2]/ppg_21_st[2], ppg_21_dt[3], ppg_21_st[3]+ppg_21_dt[3], ppg_21_dt[3]/ppg_21_st[3], ppg_21_dt[4], ppg_21_st[4]+ppg_21_dt[4], ppg_21_dt[4]/ppg_21_st[4], ppg_21_dt[5], ppg_21_st[5]+ppg_21_dt[5], ppg_21_dt[5]/ppg_21_st[5], sys_time, dias_time], 4))
    #rows.append([np.real(alpha), np.real(PIR), np.real(ptt), np.real(bpmax), np.real(bpmin), hrfinal, ih, il, meu])

    rows.append(np.concatenate((np.around([ppg_21_dt[0], ppg_21_st[0]+ppg_21_dt[0] ,ppg_21_dt[0]/ppg_21_st[0], ppg_21_dt[1], ppg_21_st[1]+ppg_21_dt[1], ppg_21_dt[1]/ppg_21_st[1], ppg_21_dt[2], ppg_21_st[2]+ppg_21_dt[2], ppg_21_dt[2]/ppg_21_st[2], ppg_21_dt[3], ppg_21_st[3]+ppg_21_dt[3], ppg_21_dt[3]/ppg_21_st[3], ppg_21_dt[4], ppg_21_st[4]+ppg_21_dt[4], ppg_21_dt[4]/ppg_21_st[4], ppg_21_dt[5], ppg_21_st[5]+ppg_21_dt[5], ppg_21_dt[5]/ppg_21_st[5], sys_time, dias_time], 4), [np.real(alpha), np.real(PIR), np.real(ptt), np.real(bpmax), np.real(bpmin), hrfinal, ih, il, meu])))

with open('features.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	for row in rows:
		writer.writerow(row)

