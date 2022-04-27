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
| *Fig. 2: Histogram of flux values for the data shown in Fig.1 A chi squared metric is used to evaluate the best fit model, between the two which have been physically motivated. Often the error bars on data are large enough that they overlap multiple bins, which limits how fine the binning can be.* |

Unfortunately, blazars such as PKS2155-304 are often classified as Gaussian or lognorm based on the chi-squared metric, exactly as shown in Fig. 2. In [[1]](#1), this was shown to give the wrong result in >60\% of cases. This method relied on the method of Timmer and Koenig (1995) [[2]](#2) to generate artificial time-series. 

## Generating Artificial Time-series

Another important property of a time-series is the power spectral density (PSD), which quantifies the amount of power in given frequencies sampled by the time series. It can be calculated by taking the discrete Fourier transform of a time-series, and typically has an approximate power-law functional form, i.e. *P(x) = Ax*<sup>*-p*</sup>. 

The method of Timmer and Koenig (1995) (hereafter TK95) [[2]](#2) can be used to generate artificial time-series. It works by entering a user-defined power-law shape (i.e. the choice of the exponent *p* above) with real and imaginary parts of Fourier amplitudes are drawn from a Gaussian distribution with a normalisation such that the variance is that of the observed (or user defined) lightcurve. The inverse Fourier transform gives us a time-series with the desired PSD.

Crucially, artifically generated time-series via [[2]](#2) have Gaussian PDFs. If we exponentiate the *y* values, taking care to re-adjust to the desired mean and variance, we return a time-series with a lognormal PDF, while approximately preserving the shape of PSD. 

Fig. 3 shows (top left) a user-defined PSD, with a TK95 generated time-series in the top right. The sub panels below the long time-series show extracted time-series, which may represent astronomically observed light curves. Below the PSD are the PDFs for these sub-light curves. Although this time-series had an intrinsically Gaussian PDF, the sampled time-series exhibit a range of PDF shapes (these are colour and linestyle matched), showing the problem with classifying them simply from a histogram alone.

|![Alt text](./figs/nonStationarity.png?raw=true "Definition of the Problem")|
|:--:| 
| *Fig. 3: A long artifical light curve with Gaussian PDF (top right) generated from a user-defined PSD with random noise (top-left). Sample observations in green, red and blue can return PDF shapes which are significantly different from the true Gaussian PDF, if the histogram method is used.* |

# A Machine Learning Classification Approach

We have already established that in theory an infinite number of artificial time-series which are known to have either a Gaussian or a lognormal PDF can be produced. This is incredibly advantageous for machine learning models, as often a large data set can reduce the need for hand-engineering or additional complexity. 

The objectives of the current project are as follows:

* To build a machine learning classifier which can correctly classify artificial time-series, then apply this to astrophysical data
* The inputs must be well defined: either the time-series directly or easy-to-measure quantities. This ensures reproducability.
* An ideal model should be extendable to be multivariate. In astrophysics, we can have many simulataneous time-series at different wavelenths.
* Any extension should allow for the inclusion of e.g. unevenly sampled data or other common data degradations. 

In the following sections I present the prelimanry results.


## Logistic Regression

As this is a binary classification model, it appears perfectly suited to logistic regression. We classify our Gaussian PDF lightcurves with *y=1* and those with lognormal PDFs as *y=0*. In these preliminary results, we use 10,000 artificially generated lightcurves each of length 128 timesteps, as is the case in the data in Fig. 1. 

Logistic regression can be thought of as a shallow neural network, with a single hidden layer. This network takes some features, $X$, computes some function, and feeds this into an activation function. In this example, a sigmoid is used to give either a 0 (lognormal) or a 1 (Gaussian) given our binary choice. Easily measurable features are chosen, which are:

* mean
* standard deviation
* PSD index (using a power-law PSD)
* Shapiro-Wilk test statistic
* Shapiro-Wilk p-value
* Anderson-Darling test statistic
* Anderson-Darling p-value
* skewness
* symmetry

Which can all be easlily computed in a single line of python. 

Fig. 4 shows a computation graph for the logistic regression model.

|![Alt text](./figs/logRmodel.png?raw=true "Definition of the Problem")|
|:--:| 
| *Fig. 4: Basic computation graph for our logistic regression model* |

### How to run Code

### Preliminary Results

# Convolutional Neural Network




## References
<a id="1">[1]</a> 
Morris, P. J., Chakraborty, N. and Cotter, G. (2019).  
Deviations from normal distributions in artificial and real time series: a false positive prescription,
MNRAS, 489, 2117-2129.

<a id="2">[2]</a> 
Timmer, J. and Koenig, M. (1995).  
On generating power law noise, AAP, 300, 707.



