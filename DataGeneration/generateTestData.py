'''
Program to generate N artificial light curves using the TK95 or Emmanoulopoulos method. Half will be normally distributed, half lognormal.

INPUTS
PSD_range - min and max values. Each is drawn randomly. 
mean - desired light curve mean. CAREFUL: will change when exponentiated 

Need more features: Fvar? Chi-squared for Gauss and lognorm as before? Symmetry test, e.g. sum(xi - median) 

'''

import numpy as np
import matplotlib.pyplot as plt
import genTK95LC as TK #for TK95 simulation - as falsePositiveRate.py in PhD work folder
from DELCgen import * #for EMP13 simulation
import random as random 
import scipy.stats


def logNormal_normed(xdata, xmean, sigma):
    #print ( 1.0/(sigma*xdata*(2.0*np.pi)**0.5) )
    return ( 1.0/(sigma*xdata*(2.0*np.pi)**0.5) )*np.exp(-1.0*(np.log(xdata)-xmean)**2.0/(2.0*sigma**2.0))

test_flag_G = 0
test_flag_LN = 0
save_flag = 1
mode = 'TK95' #Timmer-Koenig method
#mode = 'EMP13' #Emmanoulopoulos method
N = 10000#00 #how many LCs of each form
p_min = 0.0 #minimum PSD index
p_max = 2.01 #maximum PSD index
LC_len = 128
mean_min = 5 #arbitrary
mean_max = 15
sd_min = 0.1
sd_max = 1


#arrays to store properties for Gaussian
mean_G = np.zeros(N)
sd_G = np.zeros(N)
p_G = np.zeros(N)
SW_G = np.zeros(N)
SWp_G = np.zeros(N)
AD_G = np.zeros(N)
skew_G = np.zeros(N)
sym_G = np.zeros(N)
Gval_G = np.ones(N) #classification - all 1 for Gaussian
LC_Gall = np.zeros(N*LC_len)
LC_Gall = np.reshape(LC_Gall, (N, LC_len)) #one row for each LC

#arrays to store properties for lognormal
mean_LN = np.zeros(N)
sd_LN = np.zeros(N)
p_LN = np.zeros(N)
SW_LN = np.zeros(N)
SWp_LN = np.zeros(N)
AD_LN = np.zeros(N)
skew_LN = np.zeros(N)
sym_LN = np.zeros(N)
Gval_LN = np.zeros(N) #classification - all 0 for LN
LC_LNall = np.zeros(N*LC_len)
LC_LNall = np.reshape(LC_LNall, (N, LC_len)) #one row for each LC

for i in range(N): #for Gaussian
    #randomly choose properties
    PSD_p = random.uniform(p_min, p_max) #low, high; range includes low but not high
    meanLC = random.uniform(mean_min, mean_max)
    sdLC = random.uniform(sd_min, sd_max)
    randomSeed = random.randrange(1000) #for light curve generation

    if mode == 'TK95':
        LC, PSDfqs, PSD = TK.lcsim_TK95(PSD_p, meanLC, sdLC,randomSeed, length=LC_len, tbin=1.0)
        if test_flag_G == 1:
            plt.figure()
            plt.plot(np.arange(len(LC)), LC)

            plt.figure()
            plt.hist(LC, bins=10)

            plt.show()
            a = raw_input(' ')
    if mode == 'EMP13':
        #PL has form A * (v**(-a)) + c for PL(v,A,a,c)
        #delc = Simulate_DE_Lightcurve(PL, (1.0,PSD_p,0.0), st.lognorm, (0.3, 0.0, 7.4), tbin=1, LClength=LC_len) #power-law, PSD-params, PDF, params
        #st.norm: loc=mean, scale=sd, input is (
        delc = Simulate_DE_Lightcurve(PL, (1.0,PSD_p,0.0), st.norm, (meanLC, sdLC), tbin=1, LClength=LC_len) #power-law, PSD-params, PDF, params
        LC = delc.flux
        if test_flag_G == 1:
            print ('mean, sd: ', meanLC, sdLC)
            delc.Plot_Stats()
            '''
            plt.figure()
            plt.plot(np.arange(len(LC)), LC)

            plt.figure()
            plt.hist(LC, bins=10)

            plt.show()
            '''
            a = raw_input(' ')


    #do some stats
    SW_dataG = scipy.stats.shapiro(LC)
    AD_dataG = scipy.stats.anderson(LC, dist='norm')
    median = np.median(LC)
    sym = np.sum(LC - median)

    #SW_dataLN = scipy.stats.shapiro(np.log10(LC))
    if test_flag_G == 1:
        print (SW_dataG, AD_dataG)

    SW_ts = SW_dataG[0] #test statistic
    SW_pval = SW_dataG[1]
    AD_ts = AD_dataG[0] #test statistic
    skew = scipy.stats.skew(LC)#, axis=0, bias=True, nan_policy='propagate')
    #print ('skewness:', skew)
    #a = input(' ')

    #store values
    LC_Gall[i] = LC
    mean_G[i] = meanLC
    sd_G[i] = sdLC
    p_G[i] = PSD_p
    SW_G[i] = SW_ts
    SWp_G[i] = SW_pval
    AD_G[i] = AD_ts
    skew_G[i] = skew
    sym_G[i] = sym

if save_flag == 1:
   print ('saving G light curves...')
   data_G = np.c_[mean_G, sd_G, p_G, SW_G, SWp_G, AD_G, skew_G, sym_G, Gval_G]
   np.savetxt('GaussianData.txt', data_G)
   np.savetxt('GaussianLCs.txt', LC_Gall) #each row is one light curve
   

#for LN note: mean and sd may change after exponentiation - re-normalise the mean?




for i in range(N): #for lognorm
    #randomly choose properties
    PSD_p = random.uniform(p_min, p_max) #low, high; range includes low but not high
    meanLC = random.uniform(mean_min, mean_max)
    sdLC = random.uniform(sd_min, sd_max)
    randomSeed = random.randrange(1000) #for light curve generation

    if mode == 'TK95':
        LC0, PSDfqs, PSD = TK.lcsim_TK95(PSD_p, meanLC, sdLC,randomSeed, length=LC_len, tbin=1.0)
        LC1 = np.exp(LC0) #make lognormal
        #renormalise to mean
        #LC = LC1 - np.mean(LC1) + meanLC
        LC = (LC1-np.mean(LC1))/np.std(LC1)*sdLC+meanLC #only changes axis scaling - preserves time-series and PSD (test below)

        if test_flag_LN == 1:
            print ('LN old, new means: ', meanLC, np.mean(LC) )
            print ('LN old, new sds: ', sdLC, np.std(LC) )
            '''
            plt.figure()
            plt.title('exponentiated')
            plt.plot(np.arange(len(LC1)), LC1)

            plt.figure()
            plt.title('exponentiated')
            plt.hist(LC1, bins=10)
            '''
            plt.figure()
            plt.title('renormalised')
            plt.plot(np.arange(len(LC)), LC)

            plt.figure()
            plt.title('renormalised')
            plt.hist(LC, bins=10)

            plt.show()
            a = raw_input(' ')
    if mode == 'EMP13':
        #PL has form A * (v**(-a)) + c for PL(v,A,a,c)
        #delc = Simulate_DE_Lightcurve(PL, (1.0,PSD_p,0.0), st.lognorm, (0.3, 0.0, 7.4), tbin=1, LClength=LC_len) #power-law, PSD-params, PDF, params
        #st.norm: loc=mean, scale=sd, input is (
        delc = Simulate_DE_Lightcurve(PL, (1.0,PSD_p,0.0), st.lognorm, (meanLC, sdLC), tbin=1, LClength=LC_len) #power-law, PSD-params, PDF, params
        LC = delc.flux
        if test_flag_G == 1:
            print ('mean, sd: ', meanLC, sdLC)
            delc.Plot_Stats()
            '''
            plt.figure()
            plt.plot(np.arange(len(LC)), LC)

            plt.figure()
            plt.hist(LC, bins=10)

            plt.show()
            '''
            a = raw_input(' ')


    #do some stats
    SW_dataLN = scipy.stats.shapiro(LC)
    AD_dataLN = scipy.stats.anderson(LC, dist='norm')
    median = np.median(LC)
    sym = np.sum(LC - median)

    #SW_dataLN = scipy.stats.shapiro(np.log10(LC))
    if test_flag_LN == 1:
        print (SW_dataLN, AD_dataLN)

    SW_ts = SW_dataLN[0] #test statistic
    SW_pval = SW_dataLN[1]
    AD_ts = AD_dataLN[0] #test statistic
    skew = scipy.stats.skew(LC)


    #store values
    LC_LNall[i] = LC
    mean_LN[i] = meanLC
    sd_LN[i] = sdLC
    p_LN[i] = PSD_p
    SW_LN[i] = SW_ts
    SWp_LN[i] = SW_pval
    AD_LN[i] = AD_ts
    skew_LN[i] = skew
    sym_LN[i] = sym

if save_flag == 1:
   print ('saving LN light curves...')
   data_LN = np.c_[mean_LN, sd_LN, p_LN, SW_LN, SWp_LN, AD_LN, skew_LN, sym_LN, Gval_LN]
   np.savetxt('LognormData.txt', data_LN)
   np.savetxt('LognormLCs.txt', LC_LNall) #each row is one light curve



