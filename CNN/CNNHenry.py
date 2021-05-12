import math

import tensorflow.keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.random.set_seed(123)
np.random.seed(123)
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation, BatchNormalization
from NNutils import *
import sys
import keras.losses

from sklearn.utils import class_weight



np.set_printoptions(threshold=sys.maxsize)

def load_data():
    print("loading data")
    images = np.load("images.npy", allow_pickle=True)
    windinfo = np.load("windinfo.npy", allow_pickle=True)
    outputs = np.load("outputs.npy", allow_pickle=True)

    outputs = outputs.reshape(outputs.shape[:-3] + (-1, 3))  # reshape from 16x16x3 to 256x3 (convention)


    print("input images: ", images.shape)
    print("wind info: ", windinfo.shape)
    print("outputs: ", outputs.shape)

    out = np.argmax(outputs, axis = 2)
    print(outputs.shape)

    unique, counts = np.unique(out, return_counts=True)
    weights = dict(zip(unique, counts))
    total = np.sum(counts)

    #print(images[0].dtype, windinfo[0].dtype, outputs[0].dtype)
    #exit()

    print("casting npy arrays")
    images = np.cast['float16'](images)
    windinfo = np.cast['float16'](windinfo)
    outputs = np.cast['float16'](outputs)

    #images = tf.cast(images[:1500], tf.float16)
    #windinfo = tf.cast(windinfo[:1500], tf.float16)
    #outputs = tf.cast(outputs[:1500], tf.float16)

    #return images[:20], windinfo[:20], outputs[:20], weights, total
    return images, windinfo, outputs, weights, total

def jaccard(y_true, y_pred, smooth=100):
    """http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def weightedLoss(originalLossFunc, weightsList):
    # https://stackoverflow.com/questions/51793737/custom-loss-function-for-u-net-in-keras-using-class-weights-class-weight-not
    def lossFunc(true, pred):

        axis = -1 #if channels last
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)
        #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        classSelectors = [K.equal(tf.cast(i, tf.int64), classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred)
        loss = loss * weightMultiplier

        return loss

    return lossFunc


def build_model(input1_shape, input2_shape):
    feature_vector_len = 4*4*32
    model1 = Sequential()

    model1.add(Conv2D(input_shape=input1_shape, filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")) # 16x16x8
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))  # trying to go symmetric from here
    #model1.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Flatten())
    model1.add(Dense(feature_vector_len, activation='relu'))                # 1D feature vector

    # CONCATENATE WITH WIND INFO
    inp2 = Input(input2_shape)
    model_concat = concatenate([model1.output, inp2], axis=1)
    #deconv = Dense(16*16*3, activation='relu')(model_concat)
    deconv = Dense(feature_vector_len, activation='relu')(model_concat)
    #deconv = Dense(16*16*3, activation='relu')(model_concat)

    # DECONVOLUTIONS TO OUTPUT IMAGE
    deconv = Reshape((4, 4, 32))(deconv)
    deconv = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    # kind of copied from here: https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
    #deconv = BatchNormalization()(deconv)
    deconv = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    #deconv = BatchNormalization()(deconv)
    deconv = Reshape((16 * 16, 3))(deconv)
    deconv = Activation('softmax')(deconv)

    model = Model(inputs=[model1.input, inp2], outputs=deconv)

    return model


def predict(model=None, data=None, n_examples=10):
    """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""

    images = np.load("images.npy", allow_pickle=True)
    windinfo = np.load("windinfo.npy", allow_pickle=True)
    outputs = np.load("outputs.npy", allow_pickle=True)

    outputs = outputs.reshape(outputs.shape[:-3] + (-1, 3))  # reshape from 16x16x3 to 256x3 (convention)

    #if not data:
    #    images, windinfo, outputs, weights, total = load_data()
    #else:
    #    images, windinfo, outputs = data

    if not model:
        model = load("safetySafe")
    X1 = images[:n_examples]
    X2 = windinfo[:n_examples]

    print("predicting")
    results = model.predict([X1, X2])                        # outputs 16x16x3

    # translate the 5 channel input back to displayable images
    #orig_img = np.zeros((len(X1), 255, 255))
#
    ##print(np.argmax(results[0], axis = 1).shape)
    ##print(np.argmax(results[0], axis = 1))
#
    #for i, image in enumerate(X1):
    #    print("reconstructing image ", i, "/", n_examples)
    #    for y, row in enumerate(image):
    #        for x, cell in enumerate(row):
    #            for idx, item in enumerate(cell):
    #                #print(i, y, x, idx)
    #                if item == 1:
    #                    orig_img[i][y][x] = idx


    # display input images and the 2 waypoint output images (from 2 channels)
    for i in range(len(results)):
        #plt.imshow(orig_img[i])
        #plt.title("input image")
        #plt.show()
        result = np.reshape(results[i], (16,16,3))
        desired_output = np.reshape(outputs[i], (16,16,3))

        # Give 0/1 pixel values base on where values are the highest
        non_wp_res = np.zeros((16, 16))
        dig_img_res = np.zeros((16, 16))
        drive_img_res = np.zeros((16, 16))
        all_max_img = np.zeros((16, 16))

        for j, col in enumerate(result):
            for k, cell in enumerate(col):
                highest_value_idx = np.argmax(cell)
                #print(highest_value_idx)
                if highest_value_idx == 0:
                    non_wp_res[j][k] = 1
                    all_max_img[j][k] = 0
                elif highest_value_idx == 1:
                    dig_img_res[j][k] = 1
                    all_max_img[j][k] = 1
                elif highest_value_idx == 2:
                    drive_img_res[j][k] = 1
                    all_max_img[j][k] = 2
                #print(cell)
        #exit()



        non_wp_res, dig_img_res, drive_img_res = np.dsplit(result, 3)                  # depth split of 2 channel image
        print(np.amax(non_wp_res), np.amax(dig_img_res), np.amax(drive_img_res))
        non_wp_out, dig_img_out, drive_img_out = np.dsplit(desired_output, 3)                  # depth split of 2 channel image
        f, axarr = plt.subplots(2,4)
        axarr[0,0].imshow(np.reshape(non_wp_res, newshape=(16, 16)))
        axarr[0,0].set_title("non-wp NN output")
        axarr[0,1].imshow(np.reshape(dig_img_res, newshape=(16, 16)))
        axarr[0,1].set_title("dig waypoints image NN output")
        axarr[0,2].imshow(np.reshape(drive_img_res, newshape=(16, 16)))
        axarr[0,2].set_title("drive image NN output")
        axarr[0,3].imshow(np.reshape(all_max_img, newshape=(16, 16)))
        axarr[0,3].set_title("decision (non-wp, dig, drive)")


        axarr[1,0].imshow(np.reshape(non_wp_out, newshape=(16, 16)))
        axarr[1,0].set_title("non-wp desired")
        axarr[1,1].imshow(np.reshape(dig_img_out, newshape=(16, 16)))
        axarr[1,1].set_title("dig image desired")
        axarr[1,2].imshow(np.reshape(drive_img_out, newshape=(16, 16)))
        axarr[1,2].set_title("drive image desired")

        plt.show()



if __name__ == "__main__":
    #predict()                          # predict with model loaded from file
    #exit()

    images, windinfo, outputs, weights, total = load_data()

    class_weights = np.zeros(3)
    class_weights[0] += (1 / weights[0]) * total / 2.0
    class_weights[1] += (1 / weights[1]) * total / 2.0
    class_weights[2] += (1 / weights[2]) * total / 2.0 # 5 waypoints but 250 non-wp, also, slightly more driving wp than digging, 61:41 ratio
    print("class weights: ", class_weights)

    model = build_model(images[0].shape, windinfo[0].shape)

    #model.compile(loss=jaccard,
    #              optimizer='adam',
    #              metrics=[jaccard, 'accuracy'])

    model.compile(loss=weightedLoss(tensorflow.keras.losses.categorical_crossentropy, class_weights),
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

    history = model.fit([images, windinfo],                   # list of 2 inputs to model
              outputs,
              batch_size=64,
              epochs=100,
              shuffle=True,
              validation_split=0.2,
              #class_weight=class_weights,
              callbacks=[callback])



    save(model, "safetySafe")                       # utils
    plot_history(history)
    predict(model=model)