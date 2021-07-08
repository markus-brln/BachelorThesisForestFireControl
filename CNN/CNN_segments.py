import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *

from getpass import getuser


def load_data(out_variant, experiment):
    directory = "/home/f118885/data/thesis/" if getuser() == "f118885" else ""
    print("loading data:" + directory + out_variant + experiment + ".npy")
    images = np.load(directory + "images_" + out_variant + experiment + ".npy", allow_pickle=True)
    # concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
    outputs = np.load(directory + "outputs_" + out_variant + experiment + ".npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("outputs: ", outputs.shape)

    return images, outputs


def build_model(input_shape, size = 16):
    """
    and outputs [cos(pos_angle), sin(pos_angle), radius, drive/dig] with x,y relative to the active agent's position."""

    downscaleInput = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1,1), activation="relu", padding="same")(downscaleInput)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)
    out = Dense(48, activation='sigmoid')(downscaled)
    seg_out = Dense(32, activation='sigmoid')(out)
    seg_out = Dense(size, name='seg', activation='sigmoid')(seg_out)                                     # nothing specified, so linear output
    dig_out = Dense(4, activation='sigmoid')(out)
    dig_out = Dense(1, name='dig', activation='sigmoid')(dig_out)

    model = Model(inputs=downscaleInput, outputs=[seg_out, dig_out])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)    # initial learning rate faster

    model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
                  optimizer=adam,
                  metrics=['categorical_accuracy'])

    return model

if __name__ == "__main__":
    size = 16 ## Size of segments

    ## experiment type
    out_variant = "segments"
    experiments =  ["BASIC", "STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
    experiment = experiments[0]                             # dictates model name

    if len(sys.argv) > 1 and int(sys.argv[1]) < len(experiments):
      experiment = experiments[int(sys.argv[1])]

    ## Data processing
    images, outputs = load_data(out_variant, experiment)
    test_data = [images[:20], outputs[:20]]
    images = images[20:]
    outputs = outputs[20:]
    segments = np.asarray([x[:size] for x in outputs])
    dig = np.asarray([[x[size]] for x in outputs], dtype=np.float16)


    ## Class weights, currently unused
    class_weights = {idx: 0 for idx in range(size)}
    for segment in segments:
      class_weights[segment.argmax()] += 1
    for idx in range(size):
      class_weights[idx] /= len(segments)

    ## Model initialization and training
    model = build_model(images[0].shape)
    print(model.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_seg_categorical_accuracy', restore_best_weights=True, patience=8)
    history = model.fit([images],  # list of 2 inputs to model
              [segments, dig],
              batch_size=64,
              epochs=100,
              shuffle=True,
              callbacks=[callback],
              # class_weight=class_weights,
              validation_split=0.2)

    save(model, "CNNsegments" + experiment)  # utils
    print(f"saving as CNNsegments{experiment}")
    plot_history(history=history)
    # predict()                          # predict with model loaded from file
    # exit()

