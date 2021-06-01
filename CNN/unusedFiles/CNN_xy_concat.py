import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *
#tf.random.set_seed(123)
#np.random.seed(123)


def load_data(out_variant):
    print("loading data")
    images = np.load("images_" + out_variant + ".npy", allow_pickle=True)
    concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
    outputs = np.load("outputs_" + out_variant + ".npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("outputs: ", outputs.shape)

    return images, concat, outputs


def build_model(input_shape, concat_shape):
    """Architecture for the xy outputs. Takes a 6-channel image of the environment
    and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""

    downscaleInput = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaleInput)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=64, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)

    inp2 = Input(concat_shape)
    model_concat = concatenate([downscaled, inp2], axis=1)
    out = Dense(64, activation='sigmoid')(model_concat)
    out = Dense(32, activation='sigmoid')(out)
    out = Dense(3)(out)                                     # nothing specified, so linear output

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
        model = load("CNNxy")

    images, concat, outputs = test_data
    results = model.predict([images, concat])
    print("results ", results)

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
    architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
    out_variant = architecture_variants[0]

    images, concat, outputs = load_data(out_variant)
    test_data = [images[:20], concat[:20], outputs[:20]]
    images, concat, outputs = images[20:], concat[20:], outputs[20:]

    #check_performance(test_data)
    #exit()

    model = build_model(images[0].shape, concat[0].shape)
    print(model.summary())
    #exit()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    class_weight = {0: 0.7,
                    1: 0.9, # why y coords less precise??
                    2: 0.5}

    history = model.fit([images, concat],  # list of 2 inputs to model
              outputs,
              batch_size=64,
              epochs=50,
              shuffle=True,
              callbacks=[callback],
              #class_weight=class_weight,
              validation_split=0.2)

    save(model, "CNNxy")  # utils
    check_performance(test_data, model)
    plot_history(history=history)
    #predict(model=model, data=test_data)

"""
patience 2:
[[ 2.94984639e-01  2.58510768e-01  1.02607584e+00]
 [ 2.05759406e-02  3.74104083e-01  1.00585532e+00]
 [-4.09858406e-01 -1.03594735e-04  9.90627050e-01]
 [-9.11411922e-03 -3.16003352e-01  1.02364981e+00]
 [ 3.15944940e-01 -1.50187820e-01  1.02305043e+00]
 [ 2.49238849e-01 -2.62882710e-01  1.02715027e+00]
 [ 3.32698286e-01  2.52575278e-01  1.05216086e+00]
 [ 8.10624473e-03  3.95186126e-01  1.02276850e+00]
 [-3.99246097e-01  1.73732147e-01  1.01920116e+00]
 [-1.52711168e-01 -2.24665582e-01  1.03958142e+00]
 [ 3.85340035e-01 -3.53569001e-01  1.00229347e+00]
 [ 3.64671439e-01  2.08834320e-01  1.03540397e+00]
 [ 2.82211035e-01  2.01858431e-01  1.03551030e+00]
 [-6.94707707e-02  3.46116535e-02  1.03225744e+00]
 [-2.62999982e-02 -2.46265203e-01  1.03271818e+00]
 [ 4.39144820e-01 -1.84096932e-01  1.00884044e+00]
 [ 3.66291493e-01  3.28707516e-01  1.01003790e+00]
 [ 8.61206725e-02  1.18945636e-01  1.03751266e+00]
 [-3.72175910e-02  1.60673410e-02  1.00376105e+00]
 [-7.62920603e-02 -2.42700636e-01  1.01043761e+00]]
average Delta X:  0.12584362123161555
average Delta Y:  0.09161575371399522
average Delta DD:  0.022881978750228883


patience 1: 
 [-0.02437374 -0.08945458  0.84981143]
 [-0.02702904 -0.09062058  0.85652924]
 [-0.02282679 -0.09860069  0.8630065 ]
 [-0.01203194 -0.10163944  0.85491157]
 [-0.02104833 -0.09229805  0.88378763]
 [-0.01863575 -0.0889342   0.8753562 ]
 [-0.02928913 -0.08457538  0.88660324]
 [-0.03376792 -0.08535147  0.88313866]
 [-0.02950079 -0.08681776  0.89049035]
 [-0.0161792  -0.09833044  0.88390815]
 [-0.02554017 -0.08899659  0.89110696]
 [-0.02561095 -0.09026257  0.89177775]
 [-0.02749812 -0.09125054  0.887328  ]
 [-0.02537356 -0.0918282   0.89215195]
 [-0.01617044 -0.09172588  0.8877018 ]
 [-0.02136821 -0.09032801  0.87859863]
 [-0.02055001 -0.09108231  0.88884383]
 [-0.02111759 -0.08844857  0.89646983]
 [-0.01944978 -0.0940548   0.8880652 ]]
average Delta X:  0.23166612423956395
average Delta Y:  0.24804759360849857
average Delta DD:  0.12086288034915924

Process finished with exit code 0
"""