import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *
tf.random.set_seed(123)
np.random.seed(123)


def load_data():
    print("loading data")
    images = np.load("imagesNEWall.npy", allow_pickle=True)
    windinfo = np.load("concatNEWall.npy", allow_pickle=True)
    outputs = np.load("outputsNEWall.npy", allow_pickle=True)

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
    test_data = [images[:200], concat[:200], outputs[:200]]
    images, concat, outputs = images[200:], concat[200:], outputs[200:]

    #check_performance(test_data)
    #exit()

    model = build_model(images[0].shape, concat[0].shape)
    print(model.summary())
    #exit()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
    class_weight = {0: 0.7,
                    1: 1.0,
                    2: 0.5}

    history = model.fit([images, concat],  # list of 2 inputs to model
              outputs,
              batch_size=64,
              epochs=100,
              shuffle=True,
              validation_split=0.2,
              class_weight=class_weight,
              callbacks=[callback])

    check_performance(test_data, model)
    save(model, "CNN")                       # utils
    plot_history(history=history)
    #predict(model=model, data=test_data)

    """
1, 1, 0.5
average Delta X:  0.033679325729608536
average Delta Y:  0.03717247754335404
average Delta DD:  0.27392242016270757

1.5,1.5, 0.5
average Delta X:  0.03869596555829048
average Delta Y:  0.05574466273188591
average Delta DD:  0.28211056704632936

1,1,0.2
average Delta X:  0.04165280729532242
average Delta Y:  0.049058937579393384
average Delta DD:  0.3279600426927209

1,1,1
average Delta X:  0.038817690014839173
average Delta Y:  0.040950313434004786
average Delta DD:  0.2549189481884241

0.7, 1, 0.5
average Delta X:  0.0401893462240696
average Delta Y:  0.04023878253996372
average Delta DD:  0.2874618637561798

2 patience
average Delta X:  0.039171589463949205
average Delta Y:  0.04311009913682937
average Delta DD:  0.28043614050373433
"""
