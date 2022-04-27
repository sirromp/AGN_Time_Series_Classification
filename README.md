# AGN Time Series Classification

The work in this repository builds on that of [[1]](#1). This work can be accessed [here](https://arxiv.org/abs/1908.04135). The problem definition will be summarised and objectives outlined. 

## Definition of the Problem

The analysis of astrophysical time-series, here defined as some variable quantity such as flux or brightness can reveal much about their underlying physical nature. Fig. 1 shows such an example for the blazar PKS 2155-304 

| ![Alt text](./figs/PKS2155_LC.png?raw=true) |
|:--:| 
| *Fig. 1: Fermi gamma-ray light curve for the blazar PKS2155-304. Photons are binned into 128 (monthly) time-bins which are integrated to compute their fluxes.* |


The problem comes down to measuring a quantity called the probability distribution function (PDF). This can be calculated by forming a histogram of the *y* (or flux) values in the above time series. Fig. 2 shows such a histogram for the Fig. 1 time-series. For astrophysical time-series, two functional forms for the PDF have been physically motivated: Gaussian (G) and lognormal (LN). In short, the latter has the property that the amount of variation is proportional to the brightness level, and that the underlying process is multiplicative, whereas the former does not, and the underlying process governing the variability is additive. Intuitively this follows because the log of a product can be represented as a sum, hence a lognormal PDF in linear space is normally distributed in logspace.  

| ![Alt text](./figs/PKS2155_PDF.png?raw=true) |
|:--:| 
| *Fig. 2: Histogram of flux values for the data shown in Fig.1 A chi squared metric is used to evaluate the best fit model, between the two which have been physically motivated.* |

Unfortunately, blazars such as PKS2155-304 are often classified as Gaussian or lognorm based on the chi-squared metric, exactly as shown in Fig. 2. In [[1]](#1), this was shown to give the wrong result in >60\% of cases. This method relied on the method of Timmer and Koenig (1995) [[2]](#2) to generate artificial time-series. 

## Generating Artificial Time-series

Another important property of a time-series is the power spectral density (PSD), which quantifies the amount of power in given frequencies sampled by the time series. It can be calculated by taking the discrete Fourier transform of a time-series, and typically has an approximate power-law functional form, i.e. *P(x) = Ax*<sup>*-p*</sup>. 

The method of Timmer and Koenig (1995) [[2]](#2) can be used to generate artificial time-series. It works by entering a user-defined power-law shape (i.e. the choice of the exponent *b* above)


#A Machine Learning Classification Approach

Paragraphs are separated by a blank line. 

2nd paragraph. *Italic*, **bold**, and `monospace`. Itemized lists
look like:

* this one
* that one
* the other one

> Block quotes are
> written like so.
>
> They can span multiple paragraphs,
> if you like.









![Alt text](./figs/nonStationarity.png?raw=true "Definition of the Problem")


"...the **go to** statement should be abolished..." [[1]](#1).

## References
<a id="1">[1]</a> 
Morris, P. J., Chakraborty, N. and Cotter, G. (2019).  
Deviations from normal distributions in artificial and real time series: a false positive prescription
MNRAS, 489, 2117-2129.

## References
<a id="2">[2]</a> 
Timmer, J. and Koenig, M. (1995).  
On generating power law noise
AAP, 300, 707.



