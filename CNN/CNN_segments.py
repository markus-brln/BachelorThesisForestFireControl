import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *

def build_model(input_shape):
    """
    and outputs [cos(pos_angle), sin(pos_angle), radius, drive/dig] with x,y relative to the active agent's position."""

    downscaleInput = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1,1), activation="relu", padding="same")(downscaleInput)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1,1), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(3, 3), strides=(2,2), activation="relu", padding="same")(downscaled) # 
    downscaled = Conv2D(filters=64, kernel_size=(3, 3), strides=(2,2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)
    out = Dense(48, activation='sigmoid')(downscaled)
    out = Dense(32, activation='sigmoid')(out)
    seg_out = Dense(16, name='seg')(out)                                     # nothing specified, so linear output
    dig_out = Dense(1, name='dig')(out)

    model = Model(inputs=downscaleInput, outputs=[seg_out, dig_out])
    adam = tf.keras.optimizers.Adam(learning_rate=0.003)    # initial learning rate faster

    model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
                  optimizer=adam,
                  # metrics='mse'
                  )

    return model

if __name__ == "__main__":
    # predict()                          # predict with model loaded from file
    # exit()
    out_variant = "segments"
    experiments =  ["BASIC", "STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
    experiment = experiments[1]                             # dictates model name

    images, outputs = load_data(out_variant)
    test_data = [images[:20], outputs[:20]]
    images = images[20:]
    outputs = outputs[20:]
    segments = np.asarray([x[:16] for x in outputs])
    dig = np.asarray([[x[16]] for x in outputs], dtype=np.float16)
    print(segments.shape)
    print(dig.shape)

    model = build_model(images[0].shape)
    print(model.summary())

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


    history = model.fit([images],  # list of 2 inputs to model
              [segments, dig],
              batch_size=64,
              epochs=100,
              shuffle=True,
              # callbacks=[callback],
              #class_weight=class_weight,
              validation_split=0.2)

    save(model, "CNNangle" + experiment)  # utils
    check_performance(test_data, model)
    plot_history(history=history)
    #predict(model=model, data=test_data)
