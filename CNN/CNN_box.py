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
    #concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
    outputs = np.load("outputs_" + out_variant + ".npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("CNN box - outputs: ", outputs.shape)

    return images, outputs


def build_model(input_shape):
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
    # out = Flatten()(out) ## do we need flatten layer before dense layer
    out = Dense(64, activation='sigmoid')(downscaled)
    out = Dense(32, activation='sigmoid')(out)
    box = Dense(221, activation='softmax')(out)
    dig_drive = Dense(1, activation='sigmoid')(out)

    # out = Dense(3)(out)                                     # nothing specified, so linear output
    # out = Dense(1, activation='softmax')(out)       ## complete guess (let's see what happens)

    model = Model(inputs=downscaleInput, outputs=[box, dig_drive])

    model.compile(loss=['categorical_crossentropy', 'mse'],
                  optimizer='adam',
                  metrics=['accuracy']) ##not sure if metric is correct (maybe try mse)

    return model

def predict(model=None, data=None, n_examples=5):
    """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""
    print("attempting prediction!")
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
        model = load("CNNbox")

    images, outputs = test_data
    results = model.predict([images])
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
    out_variant = architecture_variants[2]

    images, outputs = load_data(out_variant)
    box = [x[:-1] for x in outputs if len(outputs) > 2]
    box = np.asarray(box, dtype=np.float16)
    # print("shapes:", box.shape) ##841
    dig_drive = [x[-1] for x in outputs if len(outputs) > 2]
    dig_drive = np.asarray(dig_drive, dtype=np.float16)
    ## to finish for two things
    test_data = [images[:20], box[:20], dig_drive[:20]]
    images, box, dig_drive = images[20:], box[20:], dig_drive[20:]

    #check_performance(test_data)
    #exit()
    ## https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    print("shapes:", images[0].shape) ##(256, 256, 7)

    model = build_model(images[0].shape)
    print(model.summary())
    #exit()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    class_weight = {0: 0.7,
                    1: 0.9, # why y coords less precise??
                    2: 0.5}

    history = model.fit(images,  # used to be list of 2 inputs to model
              [box, dig_drive],
              batch_size=64,
              epochs=50,
              shuffle=True,
              callbacks=[callback],
              #class_weight=class_weight,
              validation_split=0.2)

    save(model, "CNNbox")  # utils
    check_performance(test_data, model)
    plot_history(history=history)
    #predict(model=model, data=test_data)