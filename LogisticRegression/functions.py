'''
INFO

function file for logistic regression machine learning classifier

created: 24.06.2021
'''

import numpy as np
import copy as copy
import sys

#function to normalise parameters?

def feature_scaling(X):
    '''
    INPUT
    X - features array, with column of ones. Each row one data set, each column one feature

    OUTPUT
    Xmod - normalised feature array via 

    '''
    X_new = copy.deepcopy(X)
    means = np.zeros(len(X[0]))
    sds = np.zeros(len(X[0]))

    for i in range(len(X[0])):
        sd = np.std(X[:,i])
        mean = np.mean(X[:,i])
        X_new[:,i] = (X[:,i] - mean)/sd

        means[i] = mean
        sds[i] = sd

    return X_new, means, sds #also return mean and sd to undo?


def polynomial_terms(X, d):
    '''
    INPUT
    X - features array, with column of ones. Each row one data set, each column one feature
    d - order of polynomial expansion

    OUTPUT
    X_poly - as X, but each row is polynomially expanded. e.g. X[0] = [1 2], X_poly[0] = [1 2 2 4]

    NOTES
    only works so far for d=2, otherwise very complex
    '''
    N0 = len(X[0]) #number of original terms per row
    N_new = int(N0**d) # number of new terms in each row

    #final array size
    #X_poly = np.zeros(N_new*len(X[:,0]))
    #X_poly = np.reshape(X_poly, (len(X[:,0]), N_new))

    c = 1 #counts
    X_orig = copy.deepcopy(X)
    X_loop = copy.deepcopy(X)

    while c<d:
        #create tempory array for increasing power by 1
        #print ('c, d, X_loopshape: ', c, d, np.shape(X_loop) )
        X_temp = np.zeros(N0**(c+1)*len(X_loop[:,0]))
        X_temp = np.reshape(X_temp, ( len(X_loop[:,0]), N0**(c+1) ))

        

        for i in range(len(X_orig[0])):
            #temp = X_loop[:,i]*X_loop #this needs to be performed axis wise - this line gives the right shape for d>2
            temp = (X_loop.T*X_orig[:,i]).T #vectorise to do column multiplication efficiently - works for d = 2
            #temp = (X_loop.T*X_orig[:,i]).T

            #print ('temp: ', temp, np.shape(temp)) #this is right
            #print ('X_loop[:,i]: ', X_loop[:,i])
            #print ('X_loop: ', X_loop)
            #a = input(' ')
            i1 = i*len(X_loop[0])
            i2 = (i+1)*len(X_loop[0])
            #print ('fn indices: ', i1, i2 )
            #print ('fn shapes: ', np.shape(temp), np.shape( X_temp[:, i1:i2]), np.shape(X_temp) )
            X_temp[:, i1:i2] = temp    

        X_loop = X_temp #re-define array
        c+=1
    

    return X_loop

    #loop over X doing X[i]*X -> add to array, get poly^2
    #if d higher, repeat the whole thing -> use a while loop

def sigmoid(z):
    '''
    INPUT
    -----
    z - theta^T x, from hypothesis

    OUTPUTS
    -------
    sigmoid function - forces close to 0 or 1

    NOTES
    -----
    tested for numpy arrays as needs to compute this on all elements individually
    '''
    sig = 1.0/(1.0 + np.exp(-z)) 
    return sig


def costFunction_logR(theta, X, y, reg=0):
    '''
    INPUT
    -----
    theta - model parameters as column vector
    X - features. Each row one data set, each column one feature
    y - classifications (all 0 or 1). Column vector
    reg - regularisation parameter. Default zero

    OUTPUTS
    -------
    J - cost function for input values
    grad - gradient at these values

    NOTES
    -----
    If h=0 or h=1, the original code could nan, so there are edits to prevent this.
       However, this should only happen if the learning rate is too high!

    '''
    #print ('X[0:5]: ', X[0:5])
    #print ('theta: ', theta) 

    #print (np.dot(X, theta), np.shape(X), np.shape(theta), np.shape( np.dot(X, theta) ) )

    h = sigmoid( np.dot(X,theta) ) 
    #print ('h: ', h[0:10])#, X[:,0:10], theta[0:10]) #fine

    #if h = 0 -> replace with tiny number else nans later in term 1 below
    h[h == 0] = sys.float_info.min



    m = float(len(y))

    theta_reg = copy.deepcopy(theta) #make a copy of theta
    theta_reg[1] = 0  #set first element to zero as we fo not regularize over theta_1

    #if h = 1 -> can nan so replace with 1 - smallest number
    #h[h == 1] = (h-sys.float_info.min)-1.7976931348623157e+308
    #sort np.log(1-h) so it can't be -inf
    term = np.log(1-h)
    term[term == -1.0*float('inf')] = -1.7976931348623157e+308/float(len(h)) #need to normalise so min not exceeded

    #J = (1/m) * ( -np.dot(y.T,np.log(h)) - np.dot( (1-y).T,np.log(1-h)) ) + (reg/(2*m)) * ( np.dot(theta[1:].T,theta[1:]) ) #original - works fine unless h = 0 or 1 (should work except in extreme examples when gradient descent diverges)
    J = (1/m) * ( -np.dot(y.T,np.log(h)) - np.dot( (1-y).T,term) ) + (reg/(2*m)) * ( np.dot(theta[1:].T,theta[1:]) )
    #grad = (1/m) * (h-y).T*X + (reg/m)*theta_reg.T
    grad = (1/m) * np.dot((h-y).T,X) + (reg/m)*theta_reg.T
    #print (' term 2 J: ', J, ( - np.dot( (1-y).T,term) ),  (1-y).T[:,0:10] , term[0:10], (1-h)[0:10] ) #nan in normal term

    #nan: 1 - h can be zero, which makes log(0) nan
    #print ('term 1 J: ', J, -np.dot(y.T,np.log(h)), y.T[:,0:10], np.log(h)[0:10], np.dot(y.T[:,0:10],np.log(h)[0:10]) ) #old: nans when h=0
    #a = input(' ')

    return J, grad

def costFunction_linR(theta, X, y, reg=0):
    '''
    INPUT
    -----
    theta - model parameters as column vector
    X - features. Each row one data set, each column one feature
    y - classifications (all 0 or 1). Column vector
    reg - regularisation parameter. Default zero

    OUTPUTS
    -------
    J - cost function for input values
    grad - gradient at these values

    '''
    #print ('X[0:5]: ', X[0:5])
    #print ('theta: ', theta) 

    #print (np.dot(X, theta), np.shape(X), np.shape(theta), np.shape( np.dot(X, theta) ) )

    h = ( np.dot(X,theta) ) 



    m = float(len(y))

    theta_reg = copy.deepcopy(theta) #make a copy of theta
    theta_reg[1] = 0  #set first element to zero as we fo not regularize over theta_1

    #J = (1/m) * ( -y.T*log(h) - (1-y).T*log(1-h) ) + (reg/(2*m)) * ( theta[1:].T*theta[1:] ) #untested
    J = (1/m) * ( -np.dot(y.T,np.log(h)) - np.dot( (1-y).T,np.log(1-h)) ) + (reg/(2*m)) * ( np.dot(theta[1:].T,theta[1:]) )
    #grad = (1/m) * (h-y).T*X + (reg/m)*theta_reg.T
    grad = (1/m) * np.dot((h-y).T,X) + (reg/m)*theta_reg.T

    return J, grad

#gradient descent (with regularization option)
def gradientDescent_logR(X, y, theta, reg=0, alpha=0.01, num_iters=500):
    '''
    INPUT
    -----
    X - features. Each row one data set, each column one feature
    y - classifications (all 0 or 1). Column vector
    theta - model parameters as column vector - what is changed by algorithm
    alpha - stepsize. Default 0.01
    num_iters - number of iterations.

    OUTPUTS
    -------
    J - cost function for input values
    grad - gradient at these values

    '''

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters): 

        h = sigmoid( np.dot(X,theta)  )# % hypothesis prediction for examples
        #print (np.dot(X, theta), np.shape(X), np.shape(theta), np.shape( np.dot(X, theta) ) )
        #print ('GD shapes: ',  np.shape(X), np.shape(theta), np.shape( np.dot(X, theta) ) )
        #print ('hyp, y: ', h[0:5], y[0:5])

        Errs = (h - y) #residual term 

        sumterm = np.dot( Errs.T, X)  #Errs has to be transposed to generally work

        theta = theta - (alpha/m)*sumterm.T #re-transpose or shapes are wrong
        #print ( (theta) )
        #a = input(' ')

        # Save the cost J in every iteration    
        J_history[i] = costFunction_logR( theta, X, y, reg)[0]

    return theta, J_history

#gradient descent (with regularization option)
def gradientDescent_linR(X, y, theta, reg=0, alpha=0.01, num_iters=500):
    '''
    INPUT
    -----
    X - features. Each row one data set, each column one feature
    y - classifications (all 0 or 1). Column vector
    theta - model parameters as column vector - what is changed by algorithm
    alpha - stepsize. Default 0.01
    num_iters - number of iterations.

    OUTPUTS
    -------
    J - cost function for input values
    grad - gradient at these values

    '''

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters): 

        h = ( np.dot(X,theta)  )# % hypothesis prediction for examples
        #print (np.dot(X, theta), np.shape(X), np.shape(theta), np.shape( np.dot(X, theta) ) )
        #print ('GD shapes: ',  np.shape(X), np.shape(theta), np.shape( np.dot(X, theta) ) )
        #print ('hyp, y: ', h[0:5], y[0:5])

        Errs = (h - y) #residual term 

        sumterm = np.dot( Errs.T, X)  #Errs has to be transposed to generally work

        theta = theta - (alpha/m)*sumterm.T #re-transpose or shapes are wrong
        #print ( (theta) )
        #a = input(' ')

        # Save the cost J in every iteration    
        J_history[i] = costFunction_logR( theta, X, y, reg)[0]

    return theta, J_history

def predict(theta, X, threshold=0.5):
    '''
    INPUT
    -----
    X - features. Each row one data set, each column one feature
    theta - model parameters as column vector - what is changed by algorithm
    threshold = if h(X*theta) > threshold, predict y = 1

    OUTPUTS
    -------
    preds - 1 and 0 for predicted PDFS
    '''

    #compute probability
    p = sigmoid( np.dot(X,theta) )
    #print ('pred: ', p[0:5])

    #assign to zeros and ones
    #find indices of <= 0.5 and > 0.5
    pos = p >= threshold #this has true for 1 (p>=0.5) and 0 for false (p<0.5)

    #convert to 0 and 1s
    val = pos*1 
    #print (val)
    return val


def learningCurves_gradDec(X, y, X_cv, y_cv, theta, reg=0, alpha=0.01, num_iters=500):
    '''
    INPUT
    -----
    X -  tarining features. Each row one data set, each column one feature
    y -  training classifications
    X_cv -  cv features. Each row one data set, each column one feature
    y_cv -  cv classifications
    theta - initial parameters
    reg - regularisation parameter

    OUTPUTS
    -------
    
    ''' 

    # Number of training examples
    m = (len(X[:,0]))

    #print ('m:', m)
    #a = input(' ')

    error_train = np.zeros(m) # make column vector?
    error_cv = np.zeros(m) # make column vector?

    #print ('error_train: ', error_train, range(m), m )
    #a = input(' ')

    for i in range(m):
        #print (i)
        theta =gradientDescent_logR(X[0:i+1], y[0:i+1], theta, reg, alpha, num_iters)[0] #train theta for each value of lambda on training set
        error_train[i] = costFunction_logR(theta, X[0:i+1], y[0:i+1], reg)[0] #row slices up to current example
        error_cv[i] = costFunction_logR(theta, X_cv, y_cv, reg)[0] #[0] is cost [1] is gradient
        #print ('error_train[i]: ', error_train[i], np.shape(X[0:i+1]) )
        #a = input(' ')
    return error_train, error_cv



def get_covar_matrix(X):
    '''
    INPUT
    -----
    X - each row an example of features, each col a set feature

    OUTPUT
    ------
    Sigma - covariance matrix for each parameter
    '''
    m = float( len(X[:,0]) ) #number of examples

    Sigma = (1.0/m) * np.dot(X.T,X)

    return Sigma



#function to divide data into train, test and cv sets

#function to compute training and cv errors

#rounding function (hypothesis outputs probability)

#precision function

#recall function

#F1 test function
