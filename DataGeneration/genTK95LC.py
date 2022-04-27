'''
Program containing a function to generate a light curve based on power law noise

created: 6/11/18

'''

import numpy as np
import random as random




# ==================== Light Curve Simulation Function ==========================

def lcsim_TK95(psdindex,mean,sd,randomSeed, length=1024, tbin=1.0):
    '''
    NOTES
    -----
    Generates an artificial lightcurve with the a given power spectral 
    density in frequency space, using the Timmer & Koenig (1995) method,
    assuming a power law PSD.
    
    ARGUMENTS
    ---------
    psdindex - spectral index of PSD
    mean - mean amplitude (counts/flux) of lightcurve to generate (and of Gaussian PDF)
    sd - standard deviation of amplitude of lightcurve to generate
    randomSeed - random number seed
    
    RETURNS
    -------
    lightcurve - (array)    array of amplitude values (cnts/flux) with the same
                 timing properties as entered, sampled once per tbin.
    PSD - simulated PSD with noise. Contains imaginary numbers, so only plot real part.
    frequency - the true frequency values corresponding to the PSD   
    
    DETAILS
    -------
    Takes conjugates and sums to move imag parts pi out of phase so they cancel                                                 
    '''                                  

    # lightcurve length
    longlength   = length*10# length of initial light curve  
    reallength = longlength*tbin
        
    # --------- Create power spectrum -------------------------------------------
    
    #Nyquist frequency is twice the maximum frequency
    t_arb = np.arange(1.0, longlength/2+1.0, 1.0) #div by 2 for Nyquist sampling
    times = t_arb*tbin
    
    # Create frequency array for initial light curve, up to the Nyquist freq
    frequency = t_arb/reallength
    powerlaw = frequency**(-psdindex) 


    # -------- Create two arrays of gaussian-dist numbers for each freq ---------
    
    np.random.seed(32+randomSeed)                               
    random1 = np.random.normal(0,1,(longlength/2)) #mean, sd, size to return
    np.random.seed(891+randomSeed) 
    random2 = np.random.normal(0,1,(longlength/2)) # /2 needed to Nyquist sample

    # -------- Add noise to the power law (PL) using the random numbers ----
    # (Multiply random no.s by the sqrt of half the PL value at each freq)

    real = (np.sqrt(powerlaw*0.5))*random1
    imag = (np.sqrt(powerlaw*0.5))*random2
    

    # ----- create array of Fourier components ----------------------------------
    # (+ve values use arrays above, -ve values are conjugates of +ve ones)

    #orig below
    positive = np.vectorize(complex)(real,imag) # array of positive, complex no.s
    negative = positive.conjugate()             # array of negative complex no.s - puts complex parts pi out phase so PSD is real
    revnegative = negative[::-1]               # reverse negative array: PSD is symmetric as has negative frequencies


    # join negative and positive arrays                           
    noisypowerlaw = np.append(positive[0:longlength/2-1],revnegative[1:longlength/2+1]) # Needed as taking the conjugate gives a real time series
    znoisypowerlaw = np.insert(noisypowerlaw,0,complex(0.0,0.0)) # add 0
    

    # --------- Fourier transform the noisy power law ---------------------------

    inversefourier = np.fft.ifft(znoisypowerlaw)

    longlightcurve = inversefourier.real  # take real part of the transform 

    # chop the light curve to the desired length, defined by 'length'
    lightcurve = np.take(longlightcurve,
                             range(longlength/2,length+longlength/2)) # take from middle


    # ---------- Normalise output lightcurve ------------------------------------
    #  (To desired mean and standard deviation, given by sd and mean)

    lightcurve = (lightcurve-np.mean(lightcurve))/np.std(lightcurve)*sd+mean
  
    return lightcurve, frequency, znoisypowerlaw#, longlightcurve #znoisypowerlaw is PSD, only returns real fqs

# ==================== Light Curve Simulation Function ==========================


