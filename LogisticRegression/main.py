'''
Program to perform logistical regression to classify light curves as Gaussian or lognormal based on input test data

Created: 24.06.2021

NOTES
-Gaussian class denoted by 1
-Lognormal class denoted by 0


TO DO:
-add a module than automatically sets an appropriate regularisation parameter

-default model is linear, i.e. SUM ( theta_i*x_i)

- parameters in order are: mean, sd, p, SW, AD

FINDINGS:
-adding higher order polynomial terms appears not to improve the model, and gives evidence of high variance
   -a linear model tends to give around 90% for success fractiom, precision and recall on unseen test data set
   -increasing the model complexity tends not to increase the performance
   -including regularisation also does not improve it
   -The recall tends to be slightly better than the precision, implying that more false positives (LN classed as G) that false negative (G as LN)

   -increasing the sample size to 2x10000 and using third order (d=3) gives about 93% for success fraction, precision and recall (and F1) (takes about 30min to run)
     -this is only a few % improved on d=3 for a sample of 2x1000 (between 91% and 93%). d=1 gives the same results

   -some numerical effects in learning curves for high d (thought fixed but apparently not)

-main_2Dsimple.py plots the 2D linear classification boundary (used sd and SW_ts). It looks as if it is difficult to improve

QUESTIONS:
-are their similarities between the data that are not correctly classified?
-does the probability before classification give the prior, or the result?

TO ADD:
-dimensionality reduction to plot the data and classifications
-can neural networks do better (separate script)

'''

import numpy as np
from functions import *
import matplotlib.pyplot as plt

#fraction of data to train, cross-validate and test
train_frac = 0.6
cv_frac = 0.2
test_frac = 0.2

d = 1 #polynomial power
reg_param = 0#1 #regularisation parameter - currently set by hand but can edit later
learn_rate = 0.1 #make this smaller if using regularisation
Niters = int(5E4) #no. of iterations for gradient descent
pred_thresh = 0.5 #prediction threshold above which y=1 (Gaussian here)
params = ['bias', 'mean', 'sd', 'PSD index', 'SW TS', 'SW pval', 'AD TS', 'skewness', 'symmetry']
fs_flag = 1 #1 to use feature scaling
PCA_flag = 1
Ndim = 2 #needs to be 2 for plotting

#######################################
#load in data and make correct format #
#######################################
path = '../GenerateSampleData'

Gdata = np.loadtxt(path+'/GaussianData.txt') #
LNdata = np.loadtxt(path+'/LognormData.txt')

AllData = np.concatenate((Gdata, LNdata)) #joins arrays row wise

#print('shapes: ', np.shape(Gdata), np.shape(LNdata), np.shape(AllData) ) #verify shape

#print ('AllData[0:5]:', AllData[0:5])
#print ('AllData[-5:]:', AllData[-5:])

#randomise the order of AllData
#Alldata_shuffled = np.random.shuffle(AllData)
Alldata_shuffled =np.take(AllData,np.random.permutation(AllData.shape[0]),axis=0)

#print ('shuffled data:', Alldata_shuffled)#[0:5])

#extract y
y = np.array([Alldata_shuffled[:,-1]]).T #needs to be a column vector
#print ('y, shape: ', y, np.shape(y) )

#extract X
Xcut = Alldata_shuffled[ : , 0:-1 ] #all rows, minus last column
#print ('X, shape:', Xcut, np.shape(Xcut) )
"""
#investigate leaving out parameters
el_del = [0, 1, 2, 3, 4, 5, 6] #works slightly better when no: mean, #slightly worse:sd, SW_TS, AD_TS no effect: PSDindex -other parameters compensate
Xcut = np.delete(Xcut, el_del, axis=1)  # delete column of Xcut
params_orig = copy.deepcopy(params)
for i in range(len(el_del)):
    todelete = params_orig[el_del[i]+1]
    #print (todelete)
    #print (params_orig)
    #del params[el_del[i]+1] #this could remove the wrong elements
    params.remove(todelete)
#a = input(' ')
"""
#'''
#feature scale
FS = feature_scaling(Xcut)
X_fs = FS[0]
means_fs = FS[1] #need to undo on best fit parameters
sds_fs = FS[2] 
#print ('X_fs, shape:', X_fs, np.shape(X_fs) )
if fs_flag == 1:
    Xcut = X_fs

#'''


#add bias node to X
X = np.insert(Xcut, 0, np.ones(len(Xcut[:,0])), axis=1)
#print ('X, shape:', X, np.shape(X) )

##################################################################
#    Add option to binomially expand x for polnomial features    #
##################################################################

#test = np.array([[0, 1], [1, 2]])
#print ('test array: ', test, np.shape(test))
#poly_test = polynomial_terms(test, 3)
#print ('polynomial:', poly_test, np.shape(test), np.shape(poly_test) )
#a = input(' ')


X_poly = polynomial_terms(X, d)
#print ('X_poly: ', X_poly, np.shape(X_poly), np.shape(X) ) #correct shape
X = X_poly
#a = input(' ')

#normalise the parameters to keep them small

#cut the data to make a training set
el_train_end = int(train_frac*float(len(X[:,0])))
el_cv_end = int( (train_frac+cv_frac)*float(len(X[:,0])))

#print ('lens: ', el_train_end, el_cv_end, train_frac, cv_frac, test_frac)

X_train, y_train = X[0:el_train_end], y[0:el_train_end]
X_cv, y_cv = X[el_train_end:el_cv_end], y[el_train_end:el_cv_end]
X_test, y_test = X[el_cv_end:], y[el_cv_end:]

print ('cut shapes: ', np.shape(X_train), np.shape(X_cv), np.shape(X_test) ) #correct
#a = input(' ')

#define parameters and vectorise
theta = np.array([np.zeros(len(X[0]))]).T #number of parameters in X
#theta = np.array([np.ones(len(X[0]))]).T
#print ('theta, shape: ', theta, np.shape(theta) )
print ('Model is using ', len(X[0]), 'free parameters.')
print ('Training set has size ', len(X_train[:,0]) )

#if I want a more complex hypothesis, this should go here, default is just linear

############################################################
#            test the cost function and gradient code      #
############################################################

init_cost_info = costFunction_logR(theta, X, y, reg=reg_param)
J_init = init_cost_info[0]
grad_init = init_cost_info[1]
#print ('Initial J, gradient: ', J_init, grad_init[0:5])

#initialise iterations and regularisation parameters

############################################################
#                Perform Gradient Descent                  #
############################################################

#call gradient descent to minimise
#theta_fit, J_hist = gradientDescent_logR(X, y, theta, alpha=1E-4, num_iters=100000) #this does descend!
theta_fit, J_hist = gradientDescent_logR(X_train, y_train, theta, reg=reg_param, alpha=learn_rate, num_iters=Niters)#extra 000, 0.01 for alpha

#print ('J_hist: ', J_hist)
#a = input(' ')

if d == 1:
    for i in range(len(theta_fit)):
        print ('param, fit value: ', params[i], theta_fit[i])
#else:
#    print ('best fit parameters: ', theta_fit)

plt.figure()
plt.plot(np.arange(len(J_hist)), J_hist)
plt.xlabel('no. of iterations on training set')
plt.ylabel('cost function')
plt.savefig('main_costFn.png', bbox_inches='tight')

#very basic prediction (note this is on the training set!)
#preds = predict(theta_fit, X_train)
preds = predict(theta_fit, X_test, pred_thresh)
pval_pred = sigmoid( np.dot(X_test,theta_fit) )
#preds = predict(theta_fit, X_cv)

N_max = len(preds)

pvals_wrong = np.array([])
for i in range(len(preds)):
    #print ('Actual, predicted, p(G|theta):', y_test[i], preds[i], pval_pred[i])
    if  y_test[i] != preds[i]:
        pvals_wrong = np.append(pvals_wrong, pval_pred[i])
        #a = input(' ')
'''
plt.figure()
plt.hist(pval_pred, bins=20, density=True)
plt.xlabel('all p-vals')

plt.figure()
plt.hist(pvals_wrong, bins=10, density=True)
plt.xlabel('p-vals for mis-labelled data')
'''
N_correct = 0.0
true_pos = 0.0
false_pos = 0.0
true_neg = 0.0
false_neg = 0.0
for i in range(N_max):
    if preds[i] == y_test[i]:
        N_correct += 1.0
        if preds[i] == 1:
            true_pos += 1.0
        if preds[i] == 0:
            true_neg += 1.0    
    if preds[i] != y_test[i]:
        if preds[i] == 1: #y[i] must be zero
            false_pos += 1.0
        if preds[i] == 0: #y[i] must be one
            false_neg += 1.0      

Precision = true_pos/( true_pos + false_pos  )
Recall = true_pos/( true_pos + false_neg  )
F1 = 2.0*Precision*Recall/(Precision+Recall)

print ('Success fraction is: ', N_correct/float(N_max) )
print ('Precision = ', Precision )
print ('Recall = ', Recall )

#print ('original theta: ', theta)
#"""
error_train, error_cv = learningCurves_gradDec(X_train, y_train, X_cv, y_cv, theta, reg=reg_param, alpha=learn_rate, num_iters=500)

plt.figure()
plt.plot(np.arange(len(error_train)), error_train, label=r'$\alpha_{{\rm train}}$' )
plt.plot(np.arange(len(error_cv)), error_cv, label=r'$\alpha_{{\rm cv}}$')
plt.xlabel('no. of iterations on training set')
plt.ylabel('cost function')
plt.legend()
plt.savefig('main_performance.png', bbox_inches='tight')
#plt.show()
#"""
#print ('X_test: ', X_test)
#print ('X_test minus first col: ', X_test[:,1:])

#PCA analysis
if PCA_flag == 1 and fs_flag == 1: #can only use PCA with feature scaling
    Sigma = get_covar_matrix(X_test[:,1:])
    U, S, V = np.linalg.svd(Sigma)
    #print ('All U:', U)
    #print ('U first 5 rows:', U[0:5,:])
    #print ('U first 2 cols: ', U[0:5,0:2]) #correct
    #print ('Uall first 2 cols: ', U[:,0:Ndim], np.shape(U[:,0:Ndim]) ) #correct 8x2 shape
    print ('pre-shapes: ', np.shape(U), np.shape(U[:,0:Ndim].T), np.shape(X_test[:,1:]) ) #8x8, 2x8, 400x8

    U_reduce = U[:,0:Ndim]
    Z = np.dot(X_test[:,1:],U_reduce)
    #Z = np.dot( U[:,0:Ndim].T , X_test[:,1:] )
    print ('shapes: ', np.shape(Z), np.shape(y_test) ) #400x2 and 400x1

    plt.figure()
    LN_flag = 0
    G_flag = 0
    msize = 10 #radius around misclassified points
    for i in range(len(y_test)):
        if y_test[i] == 1:
            if G_flag == 0:
                plt.plot(Z[:,0][i], Z[:,1][i], 'cs', label=r'Gaussian')
                G_flag = 1
                if preds[i] != y_test[i]:
                    plt.plot( Z[:,0][i], Z[:,1][i], 'ko', markersize=msize,fillstyle='none')                
            if G_flag == 1:
                plt.plot(Z[:,0][i], Z[:,1][i], 'cs')
                if preds[i] != y_test[i]:
                    plt.plot( Z[:,0][i], Z[:,1][i], 'ko', markersize=msize,fillstyle='none')    
        if y_test[i] == 0:
            #print ('LNflag, pred, val:', LN_flag, preds[i], y_test[i])
            if LN_flag == 0:
                plt.plot(Z[:,0][i], Z[:,1][i], 'r<', label=r'Log-normal')
                LN_flag = 1
                if preds[i] != y_test[i]:
                    #print ('LN0 incorrect')
                    plt.plot( Z[:,0][i], Z[:,1][i], 'ko', markersize=msize,fillstyle='none') 
            if LN_flag == 1:
                 plt.plot(Z[:,0][i], Z[:,1][i], 'r<')
                 if preds[i] != y_test[i]:
                     #print ('LN1 incorrect', (Z[:,0][i], Z[:,1][i]))
                     plt.plot( Z[:,0][i], Z[:,1][i], 'ko', markersize=msize,fillstyle='none') 

    plt.xlabel('1st Principle Component')
    plt.ylabel('2nd Principle Component')
    plt.legend()

plt.savefig('main_PCA.png', bbox_inches='tight')

plt.show()
