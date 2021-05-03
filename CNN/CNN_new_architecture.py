import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *


def load_data():
    print("loading data")
    images = np.load("imagesNEW.npy", allow_pickle=True)
    windinfo = np.load("concatNEW.npy", allow_pickle=True)
    outputs = np.load("outputsNEW.npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("wind info + agents: ", windinfo.shape)
    print("outputs: ", outputs.shape)
    return images, windinfo, outputs

def build_model(input1_shape, input2_shape):
    # idea/"tutorial" from:
    # https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
    feature_vector_len = 128

    # GOING TO 64x64x3
    downscaleInput = Input(shape=input1_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), activation="relu", padding="same")(downscaleInput)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)

    # GOING TO FEATURE VECTOR
    encoder = Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    encoder = Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(encoder)
    encoder = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(feature_vector_len, activation='relu')(encoder)



    # CONCATENATE WITH WIND INFO
    inp2 = Input(input2_shape)
    model_concat = concatenate([encoder, inp2], axis=1)
    out = Dense(feature_vector_len, activation='relu')(model_concat)
    out = Dense(32, activation='relu')(out)
    out = Dense(3, activation='relu')(out)

    model = Model(inputs=[downscaleInput, inp2], outputs=out)

    model.compile(loss='mse',
                  optimizer='adam')


    return model



def predict(model=None, data=None, n_examples=5):
    """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""

    if not data:
        images, concat, desired_outputs = load_data()
    else:
        images, concat, desired_outputs = data

    if not model:
        model = tf.keras.models.load_model("saved_models\\safetySafe")
    #X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
    indeces = random.sample(range(len(images)), n_examples)
    X1 = images[indeces]                                        # more clever way to write it down
    X2 = concat[indeces]
    desired = desired_outputs[indeces]

    NN_output = model.predict([X1, X2])                        # outputs 61x61x2

    # translate the 5 channel input back to displayable images
    orig_img = np.zeros((len(X1), 256, 256))
    for i, image in enumerate(X1):
        print("reconstructing image ", i+1, "/", n_examples)
        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                for idx, item in enumerate(cell):
                    if item == 1:
                        orig_img[i][y][x] = idx

    #outputs = np.zeros((len(X1), 256, 256))
    #for img, point in outputs, NN_output:


    # display input images and the 2 waypoint output images (from 2 channels)
    for i in range(len(NN_output)):
        print("agent pos: ", X2[i][-2], X2[i][-1])
        print("desired: ", desired[i])
        print("NN output: ", NN_output[i])
        plt.imshow(orig_img[i])
        plt.title("input image")
        plt.show()


if __name__ == "__main__":
    #predict()                          # predict with model loaded from file
    #exit()

    images, concat, outputs = load_data()

    model = build_model(images[0].shape, concat[0].shape)
    print(model.summary())
    #exit()

    model.fit([images, concat],  # list of 2 inputs to model
              outputs,
              batch_size=8,
              epochs=20,
              shuffle=True)                         # mix data randomly

    predict(model=model)

    #save(model, "safetySafe")                       # utils
