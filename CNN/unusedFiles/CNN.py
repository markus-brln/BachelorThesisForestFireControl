import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *


def load_data():
    print("loading data")
    images = np.load("../images.npy", allow_pickle=True)
    windinfo = np.load("../windinfo.npy", allow_pickle=True)
    outputs = np.load("../outputs.npy", allow_pickle=True)

    print("input images: ", images.shape)
    print("wind info: ", windinfo.shape)
    print("outputs: ", outputs.shape)
    return images, windinfo, outputs

def build_model_backup(input1_shape, input2_shape):
    # idea/"tutorial" from:
    # https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
    feature_vector_len = 32
    model1 = Sequential()

    # GOING TO 64x64x3
    model1.add(Conv2D(input_shape=input1_shape, filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.25))

    model1.add(Conv2D(filters=3, kernel_size=(2, 2), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))                              # we're at the output shape (None, 64, 64, 3), from now symmetric!
    model1.add(Dropout(0.25))

    # GOING TO FEATURE VECTOR
    model1.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(Conv2D(filters=2, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(Flatten())
    model1.add(Dense(feature_vector_len, activation='relu'))                # 1D feature vector

    # CONCATENATE WITH WIND INFO
    inp2 = Input(input2_shape)
    model_concat = concatenate([model1.output, inp2], axis=1)
    deconv = Dense(feature_vector_len, activation='relu')(model_concat)

    # DECONVOLUTIONS TO OUTPUT IMAGE
    deconv = Reshape((4, 4, 2))(deconv)
    deconv = Conv2DTranspose(filters=2, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2,2), activation="softmax", padding="same")(deconv)
    deconv = Reshape((64, 64, 3))(deconv)
    deconv = Activation('softmax')(deconv)                      # perform softmax on pixels
    # pixel wise softmax https://www.reddit.com/r/deeplearning/comments/4se210/keras_pixelwise_softmax_from_output_of_a/
    model = Model(inputs=[model1.input, inp2], outputs=deconv)

    model.compile(loss='binary_crossentropy',               # because output pixels can have 0s or 1s
                  optimizer='adam')                         # standard


    return model


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
    encoder = Dense(feature_vector_len, activation='relu')(encoder)
    encoder = Flatten()(encoder)


    # CONCATENATE WITH WIND INFO
    inp2 = Input(input2_shape)
    model_concat = concatenate([encoder, inp2], axis=1)
    decoder = Dense(feature_vector_len, activation='relu')(model_concat)

    # DECONVOLUTIONS TO OUTPUT IMAGE
    decoder = Reshape((8, 8, 2))(decoder)
    decoder = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(filters=12, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(decoder)
    decoder = Reshape((64, 64, 3))(decoder)
    decoder = Activation('softmax')(decoder)                      # perform softmax on pixels
    # pixel wise softmax https://www.reddit.com/r/deeplearning/comments/4se210/keras_pixelwise_softmax_from_output_of_a/
    model = Model(inputs=[downscaleInput, inp2], outputs=decoder)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')


    return model

def build_model2(input1_shape, input2_shape):
    """Built by Henry"""
    # idea/"tutorial" from:
    # https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
    feature_vector_len = 32
    model1 = Sequential()

    # GOING TO 64x64x3
    model1.add(Conv2D(input_shape=input1_shape, filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.25))
    model1.add(Conv2D(filters=3, kernel_size=(2, 2), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))                              # we're at the output shape (None, 64, 64, 3), from now symmetric!
    model1.add(Dropout(0.25))


    # GOING TO FEATURE VECTOR
    model1.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(Conv2D(filters=2, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same"))
    model1.add(Flatten())
    model1.add(Dense(feature_vector_len, activation='relu'))                # 1D feature vector


    # CONCATENATE WITH WIND INFO
    inp2 = Input(input2_shape)
    model_concat = concatenate([model1.output, inp2], axis=1)
    deconv = Dense(feature_vector_len, activation='relu')(model_concat)


    # DECONVOLUTIONS TO OUTPUT IMAGE
    deconv = Reshape((4, 4, 2))(deconv)
    deconv = Conv2DTranspose(filters=2, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Reshape((64, 64, 3))(deconv)
    deconv = Activation('softmax')(deconv)

    model = Model(inputs=[model1.input, inp2], outputs=deconv)

    model.compile(loss='categorical_crossentropy',              # because output pixels can have 0s or 1s
                  optimizer='adam')                             # standard
    return model


def predict(model=None, data=None, n_examples=5):
    """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""

    if not data:
        images, windinfo, desired_outputs = load_data()
    else:
        images, windinfo, desired_outputs = data

    if not model:
        model = tf.keras.models.load_model("saved_models\\safetySafe")
    #X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
    X1 = images[:n_examples]                                        # more clever way to write it down
    X2 = windinfo[:n_examples]
    NN_output = model.predict([X1, X2])                        # outputs 61x61x2

    # translate the 5 channel input back to displayable images
    orig_img = np.zeros((len(X1), 256, 256))
    for i, image in enumerate(X1):
        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                for idx, item in enumerate(cell):
                    if item == 1:
                        orig_img[i][y][x] = idx


    # display input images and the 2 waypoint output images (from 2 channels)
    for i in range(len(NN_output)):
        plt.imshow(orig_img[i])
        plt.title("input image")
        plt.show()

        non_wp_NN, dig_img_NN, drive_img_NN = np.dsplit(NN_output[i], 3)
        non_wp_des, dig_img_des, drive_img_des = np.dsplit(desired_outputs[i], 3)
        f, axarr = plt.subplots(2,3)

        #
        axarr[0,0].imshow(np.reshape(dig_img_NN, newshape=(64, 64)))
        axarr[0,0].set_title("dig waypoints image NN output")
        axarr[0,1].imshow(np.reshape(drive_img_NN, newshape=(64, 64)))
        axarr[0,1].set_title("drive image NN output")
        axarr[0,2].imshow(np.reshape(non_wp_NN, newshape=(64, 64)))
        axarr[0,2].set_title("non-wp NN output")

        axarr[1,0].imshow(np.reshape(dig_img_des, newshape=(64, 64)))
        axarr[1,0].set_title("dig image desired")
        axarr[1,1].imshow(np.reshape(drive_img_des, newshape=(64, 64)))
        axarr[1,1].set_title("drive image desired")
        axarr[1,2].imshow(np.reshape(non_wp_des, newshape=(64, 64)))
        axarr[1,2].set_title("non-wp desired")
        plt.show()


if __name__ == "__main__":
    #predict()                          # predict with model loaded from file
    #exit()

    images, windinfo, outputs = load_data()

    model = build_model2(images[0].shape, windinfo[0].shape)
    print(model.summary())
    #exit()

    model.fit([images, windinfo],                   # list of 2 inputs to model
              outputs,
              batch_size=16,
              epochs=2,
              shuffle=True,
              validation_split=0.2)                         # mix data randomly

    predict(model=model)

    #save(model, "safetySafe")                       # utils