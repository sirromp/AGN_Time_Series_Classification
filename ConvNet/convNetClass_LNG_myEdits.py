'''

based on:

https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

1 is G, 0 is LN

Here, I play about with models.
-original is 3 conv layers, a global average pooling and softmax - 94%
-tried 3 conv + 2 FC 
    - fits data perfectly but overfits the train set (variance error) if 10% of data used
    - a larger data set gives lower variance: train acc > 99%, val > 97%, test ~97%

'''

# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense

import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import copy as copy

#fraction of data to train, cross-validate and test
train_frac = 0.8
cv_frac = 0.0
test_frac = 0.2
normalise_flag = 1 #1 standarises data with 0 mean and sd 1
cut_flag = 1
cut_val = 0.1 #cut data to speed up for testing

#load the light curves - each row a time series. use generateTestData.py in this dir to make more data
path = '../GenerateSampleData'
lognormLCs = np.loadtxt(path+'/LognormLCs.txt')
GaussianLCs = np.loadtxt(path+'/GaussianLCs.txt')


#add the classifications - ones to G, 0 to LN
lognormLCs = np.insert(lognormLCs, 0, np.zeros(len(lognormLCs[:,0])), axis=1)
GaussianLCs = np.insert(GaussianLCs, 0, np.ones(len(GaussianLCs[:,0])), axis=1)

#concatenate and shuffle
AllData = np.concatenate((GaussianLCs, lognormLCs)) #joins arrays row wise

#randomise the order of AllData
Alldata_shuffled =np.take(AllData,np.random.permutation(AllData.shape[0]),axis=0)

if cut_flag == 1:
    print ('Original data shape: ', Alldata_shuffled.shape)
    el = int(cut_val*float(len(Alldata_shuffled[:,0])))
    Alldata_shuffled = Alldata_shuffled[0:el]
    print ('Simplified data shape: ', Alldata_shuffled.shape)

#slice our classifications so class indices match LC indices
All_classes = Alldata_shuffled[:,0] #this is zeros or ones
AllLCs = Alldata_shuffled[:,1:]

if normalise_flag == 1:
    AllLCs -= np.mean(AllLCs, axis=1, keepdims=True)
    AllLCs /= np.std(AllLCs, axis=1, keepdims=True)

#make cuts for training, test and cv sets
el_train_end = int(train_frac*float(len(AllLCs[:,0])))
el_cv_end = int( (train_frac+cv_frac)*float(len(AllLCs[:,0])))
#print ('lens: ', el_train_end, el_cv_end, train_frac, cv_frac, test_frac)
x_train, y_train = AllLCs[0:el_train_end], All_classes[0:el_train_end]
x_cv, y_cv = AllLCs[el_train_end:el_cv_end], All_classes[el_train_end:el_cv_end]
x_test, y_test = AllLCs[el_cv_end:], All_classes[el_cv_end:]


from tensorflow import keras

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

'''
plt.figure()
plt.title('check standardised?')
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()
'''

#below generalises to multivariate time-series with one channel (channels could be other data)
print ('original input shape: ', x_train.shape, y_train.shape) #each example in a row
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Our vectorized labels
#y_train = y_train.reshape((y_train.shape[0],  1))
#y_test = y_test.reshape((y_test.shape[0],  1))

print ('reshaped input shape: ', x_train.shape, y_train.shape) #should be N_ex, len(LC), 1





num_classes = len(np.unique(y_train))
print ('There are ', num_classes, ' classes')
a = input(' ')

def model1(input_shape):
    '''
    Used: all data
    train acc: >99%, val acc: > 97%, test acc ~97%

    '''

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    #gap = keras.layers.GlobalAveragePooling1D()(conv3) #unsure what this does?

    F = keras.layers.Flatten()(conv3)

    FC1 = keras.layers.Dense(128, activation="ReLU")(F)
    FC2 = keras.layers.Dense(64, activation="ReLU")(FC1)

    

    #output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap) #first is shape - either (0,1) or (1,0) with softmax so 2D
    output_layer = keras.layers.Dense(num_classes-1, activation="sigmoid")(FC2) #a sigmoid returns 0 or 1, so the shape is 1D

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def model2(input_shape):
    '''
    Used: 

    '''

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    #gap = keras.layers.GlobalAveragePooling1D()(conv3) #unsure what this does?

    F = keras.layers.Flatten()(conv3)

    FC1 = keras.layers.Dense(128, activation="ReLU")(F)
    FC2 = keras.layers.Dense(64, activation="ReLU")(FC1)
    FC3 = keras.layers.Dense(32, activation="ReLU")(FC2)
    

    #output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap) #first is shape - either (0,1) or (1,0) with softmax so 2D
    output_layer = keras.layers.Dense(num_classes-1, activation="sigmoid")(FC3) #a sigmoid returns 0 or 1, so the shape is 1D

    return keras.models.Model(inputs=input_layer, outputs=output_layer)




model = model2(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)

#loss_fn = "sparse_categorical_crossentropy"
loss_fn = "binary_crossentropy" #doesn't like using this?
#loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#happy_model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

model.summary()
a = input(' ')

epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1), #patience is no. of epochs with no improvement
]

model.compile(
    optimizer="adam",
    #loss=loss_fn,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), #false for binary
    #metrics=[loss_fn],
    #metrics=[tf.keras.metrics.BinaryCrossentropy()],
    metrics=['accuracy'],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2, #this is automatically taken from the training set
    verbose=1,
)

#it prints the loss and accuracy on training and cv sets

model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)


#metric = loss_fn#"sparse_categorical_accuracy"
#metric = [tf.keras.metrics.BinaryCrossentropy()]
metric = 'accuracy'
plt.figure()
plt.plot(history.history[metric]) #training set
plt.plot(history.history["val_" + metric]) #validation_set
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

