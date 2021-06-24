import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, \
    Reshape, Activation
from NNutils import *


# tf.random.set_seed(923)
# np.random.seed(923)


def load_data(out_variant, experiment):
    print("loading data")
    images = np.load("images_" + out_variant + experiment + ".npy", allow_pickle=True)
    # concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
    outputs = np.load("outputs_" + out_variant + experiment + ".npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("outputs: ", outputs.shape)

    return images, outputs


def build_model_xy(input_shape):
    """Architecture for the xy outputs. Takes a 6-channel image of the environment
    and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""

    downscaleInput = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(
        downscaleInput)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)
    out = Dense(48, activation='sigmoid')(downscaled)
    out = Dense(32, activation='sigmoid')(out)
    out = Dense(3)(out)  # nothing specified, so linear output

    model = Model(inputs=downscaleInput, outputs=out)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # initial learning rate faster

    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse'])

    return model


def build_model_angle(input_shape):
    return model


def check_performance(test_data, model):
    """Check average deviation of x,y,dig/drive outputs from desired
    test outputs, make density plot."""

    images, outputs = test_data
    results = model.predict([images])
    # print("results ", results)

    delta_0, delta_1, delta_2, delta_3 = 0, 0, 0, 0

    for result, desired in zip(outputs, results):
        d_x.append(abs(result[0] - desired[0]))
        d_y.append(abs(result[1] - desired[1]))
        d_digdrive.append(abs(result[2] - desired[2]))
        delta_0 += abs(result[0] - desired[0])
        delta_1 += abs(result[1] - desired[1])
        delta_2 += abs(result[2] - desired[2])
        if len(result) == 4:
            delta_3 += abs(result[3] - desired[3])

    delta_0, delta_1, delta_2 = delta_0 / len(outputs), delta_1 / len(outputs), delta_2 / len(outputs), delta_3 / len(
        outputs)
    return delta_0, delta_1, delta_2, delta_3


def run_experiments():
    """Choose the architecture variant from the list below, make sure
    you have translated all experiment data files according to your
    architecture:
    - 4 * images_architecture_experiment.npy
    - 4 * outputs_architecture_experiment.npy"""
    import time
    start = time.time()

    n_runs = 12
    architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
    architecture_variant = architecture_variants[0]
    experiments = ["UNCERTAINONLY", "UNCERTAIN+WIND"]  # "STOCHASTIC", "WINDONLY",

    for exp, experiment in enumerate(experiments):

        performances = open("performance_data/performance" + architecture_variant + experiment + ".txt", mode='w')
        performances.write("Experiment" + experiment + "\n")

        images, outputs = load_data(architecture_variant, experiment)

        for run in range(0, n_runs):
            print(experiment, "run:", run)
            images, outputs = unison_shuffled_copies(images, outputs)
            test_data = [images[:100], outputs[:100]]  # take random test data away from dataset
            images, outputs = images[100:], outputs[100:]

            ##model = build_model_xy(images[0].shape)
            model = build_model_angle(images[0].shape)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(images, outputs,
                      batch_size=64, epochs=100, shuffle=True,
                      callbacks=[callback],
                      validation_split=0.2,
                      verbose=2)

            save(model, "CNN" + architecture_variant + experiment + str(run))
            performances.write(str(check_performance(test_data, model)) + "\n")

            print(f"model {exp * n_runs + run + 1}/{len(experiments) * n_runs}")
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("time elapsed:")
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            time_left = ((len(experiments) * n_runs) / (exp * n_runs + run + 1)) * (end - start) - (end - start)

            hours, rem = divmod(time_left, 3600)
            minutes, seconds = divmod(rem, 60)
            print("estimated time left:")
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n\n")

        performances.close()


if __name__ == "__main__":
    run_experiments()
    exit()