import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *
#tf.random.set_seed(923)
#np.random.seed(923)


def load_data(out_variant, experiment):
    print("loading data")
    images = np.load("images_" + out_variant +  experiment +".npy", allow_pickle=True)
    #concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
    outputs = np.load("outputs_" + out_variant +  experiment +".npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("outputs: ", outputs.shape)

    return images, outputs


def build_model(input_shape):
    """Architecture for the xy outputs. Takes a 6-channel image of the environment
    and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""

    downscaleInput = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1,1), activation="relu", padding="same")(downscaleInput)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)
    out = Dense(48, activation='sigmoid')(downscaled)
    out = Dense(32, activation='sigmoid')(out)
    out = Dense(3)(out)                                     # nothing specified, so linear output

    model = Model(inputs=downscaleInput, outputs=out)

    adam = tf.keras.optimizers.Adam(learning_rate=0.003)    # initial learning rate faster

    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse'])

    return model

def check_performance(test_data=None, model=None):
    """Check average deviation of x,y,dig/drive outputs from desired
    test outputs, make density plot."""
    if not model:
        model = load("CNNxyWIND")

    images, outputs = test_data
    results = model.predict([images])
    #print("results ", results)

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
    # predict()                                             # predict with model loaded from file
    # exit()
    architecture_variants = ["xy", "angle", "box"]          # our 3 individual network output variants
    out_variant = architecture_variants[0]
    experiments = ["BASIC", "STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
    experiment = experiments[3]                             # dictates model name

    images, outputs = load_data(out_variant, experiment)
    images, outputs = unison_shuffled_copies(images, outputs)
    test_data = [images[:100], outputs[:100]]               # take random test data away from dataset
    images, outputs = images[100:], outputs[100:]
    #for image, output in zip(images, outputs):
    #    print(output)
    #    print("x,y active: ", image[0][0][5], image[0][0][6])
    #    plot_np_image(image)

    #check_performance(test_data)
    #exit()

    model = build_model(images[0].shape)
    print(model.summary())
    #exit()


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    class_weight = {0: 0.9,
                    1: 0.9, # why y coords less precise??
                    2: 0.5}

    history = model.fit([images],  # list of 2 inputs to model
              outputs,
              batch_size=64,
              epochs=100,
              shuffle=True,
              callbacks=[callback],
              #class_weight=class_weight,
              validation_split=0.2)

    save(model, "CNNxy" + experiment)  # utils
    check_performance(test_data, model)
    plot_history(history=history)
    #predict(model=model, data=test_data)

"""
windfive8 (from peregrine)
average Delta X:  0.06951058655977249
average Delta Y:  0.07161892522126437
average Delta DD:  0.00711326296441257

windfive7
average Delta X:  0.09615891709923745
average Delta Y:  0.09676042325794697
average Delta DD:  0.0067756188474595545

wind
average Delta X:  0.10400249533355237
average Delta Y:  0.12826617445796729
average Delta DD:  0.022055730298161505


standard architecture
0.0147
average Delta X:  0.0739567045122385
average Delta Y:  0.050198440253734586
average Delta DD:  0.013081759214401245


paths on:
23 epochs
average Delta X:  0.06107376478612423
average Delta Y:  0.06356262788176537
average Delta DD:  0.01568102240562439

paths off:
26 epochs
average Delta X:  0.054342984408140185
average Delta Y:  0.05875470265746117
average Delta DD:  0.014626950025558472

thicker paths:
26 epochs
average Delta X:  0.05476411022245884
average Delta Y:  0.053901639953255656
average Delta DD:  0.0066058248281478885


mXYEASYFIVE5
average Delta X:  0.1311199590563774
average Delta Y:  0.08875736445188523
average Delta DD:  0.022097966494038702

5 with smaller architecture
average Delta X:  0.16767661944031714
average Delta Y:  0.17266307808458806
average Delta DD:  0.03222956005483866

mXYEASYFIVE3
average Delta X:  0.12525362521409988
average Delta Y:  0.19226661119610072
average Delta DD:  0.0808281959965825

both 
average Delta X:  0.18563791997730733
average Delta Y:  0.22454358264803886
average Delta DD:  0.2907822608947754

"""