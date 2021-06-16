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

    input_layer = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(input_layer)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)
    out = Dense(64, activation='sigmoid')(downscaled)
    box = Dense(61, activation='softmax', name="box")(out)
    dig_drive = Dense(1, activation='sigmoid', name="dd")(out)


    # out = Dense(3)(out)                                     # nothing specified, so linear output
    # out = Dense(1, activation='softmax')(out)       ## complete guess (let's see what happens)

    model = Model(inputs=input_layer, outputs=[box, dig_drive])
    adam = tf.keras.optimizers.Adam(learning_rate=0.002)    # higher learning rate
    model.compile(loss=['categorical_crossentropy', 'mse'], # kullback_leibler_divergence ## categorical_crossentropy
                  optimizer=adam,
                  metrics=['categorical_accuracy', 'mse'])

    return model


if __name__ == "__main__":
    # predict()                          # predict with model loaded from file
    # exit()
    architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
    out_variant = architecture_variants[2]

    images, outputs = load_data(out_variant)
    #images, outputs = images[:500], outputs[:500]
    box = []
    dig_drive = []
    boxArr = []

    for out in outputs:
        box = out[:-1]
        dig_drive.append(out[-1])
        boxArr.append(box)

    boxes = np.asarray(boxArr, dtype=np.float16)
    dig_drive = np.asarray(dig_drive, dtype=np.float16)
    print(boxes.shape, dig_drive.shape)
    ## to finish for two things
    test_data = [images[:20], boxes[:20], dig_drive[:20]]
    images, box, dig_drive = images[20:], boxes[20:], dig_drive[20:]

    #check_performance(test_data)
    #exit()
    ## https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    print("shapes:", images[0].shape) ##(256, 256, 7)

    model = build_model(images[0].shape)
    print(model.summary())
    #exit()

    callback = tf.keras.callbacks.EarlyStopping(monitor='box_categorical_accuracy', patience=3)
    class_weight = {0: 0.7,
                    1: 0.9, # why y coords less precise??
                    2: 0.5}

    history = model.fit(images,  # used to be list of 2 inputs to model
              [box, dig_drive],
              batch_size=64,
              epochs=200,
              shuffle=True,
              callbacks=[callback],
              #class_weight=class_weight,
              validation_split=0.2) #0.2

    save(model, "CNNbox")  # utils
    # check_performance(test_data, model)
    plot_history_box(history=history)