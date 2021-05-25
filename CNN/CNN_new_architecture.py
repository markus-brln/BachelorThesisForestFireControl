import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *
#tf.random.set_seed(123)
#np.random.seed(123)


def load_data():
    print("loading data")
    images = np.load("imagesEASYNEW.npy", allow_pickle=True)
    windinfo = np.load("concatEASYNEW.npy", allow_pickle=True)
    outputs = np.load("outputsEASYNEW.npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("wind info + agents: ", windinfo.shape)
    print("outputs: ", outputs.shape)
    return images, windinfo, outputs

def build_model(input1_shape, input2_shape):
    # idea/"tutorial" from:
    # https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
    feature_vector_len = 128

    downscaleInput = Input(shape=input1_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaleInput)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=64, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=128, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = Flatten()(downscaled)
    downscaled = Dense(feature_vector_len, activation='relu')(downscaled)

    # CONCATENATE WITH WIND INFO
    inp2 = Input(input2_shape)
    model_concat = concatenate([downscaled, inp2], axis=1)
    out = Dense(feature_vector_len, activation='relu')(model_concat)
    out = Dense(32, activation='relu')(out)
    out = Dense(3, activation='sigmoid')(out)

    model = Model(inputs=[downscaleInput, inp2], outputs=out)

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])


    return model

def predict(model=None, data=None, n_examples=5):
    """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""
    if data:
        n_examples = len(data[0])
    if not data:
        images, concat, desired_outputs = load_data()
    else:
        images, concat, desired_outputs = data

    if not model:
        model = tf.keras.models.load_model("saved_models/safetySafe")
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


def check_performance(test_data=None, model=None):
    """Check average deviation of x,y,dig/drive outputs from desired
    test outputs, make density plot."""
    if not model:
        model = load("CNN")

    images, concat, outputs = test_data
    results = model.predict([images, concat])

    delta_x, delta_y, delta_digdrive = 0, 0, 0
    d_x, d_y, d_digdrive = list(), list(), list()

    for result, desired in zip(outputs, results):
        d_x.append(abs(result[0] - desired[0]))
        d_y.append(abs(result[1] - desired[1]))
        d_digdrive.append(abs(result[2] - desired[2]))
        delta_x += abs(result[0] - desired[0])
        delta_y += abs(result[1] - desired[1])
        delta_digdrive += abs(result[2] - desired[2])

    delta_x, delta_y, delta_digdrive = delta_x / len(outputs), delta_y / len(outputs), delta_digdrive / len(outputs)


    print("average Delta X: ", delta_x)
    print("average Delta Y: ", delta_y)
    print("average Delta DD: ", delta_digdrive)

    from scipy.stats import gaussian_kde
    xs = np.linspace(0, 0.2, 200)
    density1 = gaussian_kde(d_x)
    density2 = gaussian_kde(d_y)
    density3 = gaussian_kde(d_digdrive)

    plt.plot(xs, density1(xs))
    plt.plot(xs, density2(xs))
    plt.plot(xs, density3(xs))
    plt.legend(['delta x', 'delta y', 'delta dig/drive'])
    plt.show()


if __name__ == "__main__":
    # predict()                          # predict with model loaded from file
    # exit()

    images, concat, outputs = load_data()
    test_data = [images[:20], concat[:20], outputs[:20]]
    images, concat, outputs = images[20:], concat[20:], outputs[20:]

    #check_performance(test_data)
    #exit()

    model = build_model(images[0].shape, concat[0].shape)
    print(model.summary())
    #exit()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    class_weight = {0: 0.7,
                    1: 0.9, # why y coords less precise??
                    2: 0.5}

    history = model.fit([images, concat],  # list of 2 inputs to model
              outputs,
              batch_size=64,
              epochs=100,
              shuffle=True,
              validation_split=0.2,
              class_weight=class_weight,
              callbacks=[callback])

    save(model, "CNN")  # utils
    check_performance(test_data, model)
    plot_history(history=history)
    #predict(model=model, data=test_data)

    """
0.7, 0.9, 0.5, patience 1, seeds for tf and np = 123, 8k images
average Delta X:  0.03370976559817791
average Delta Y:  0.03981716945767402
average Delta DD:  0.2610402735508978

0.7, 0.9, 0.5, patience 1, seeds for tf and np = 123, all images (18k), tested on 200
average Delta X:  0.03597772844135761
average Delta Y:  0.037186374217271806
average Delta DD:  0.28269700743257997

with random seeds:
average Delta X:  0.033934826850891116
average Delta Y:  0.049013579189777376
average Delta DD:  0.2592208239249885

average Delta X:  0.03871146939694881
average Delta Y:  0.04940677046775818
average Delta DD:  0.2902473259717226
"""
