import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation
from NNutils import *
import sys

np.set_printoptions(threshold=sys.maxsize)

def load_data():
    print("loading data")
    images = np.load("images.npy", allow_pickle=True)
    windinfo = np.load("windinfo.npy", allow_pickle=True)
    outputs = np.load("outputs.npy", allow_pickle=True)

    outputs = outputs.reshape(outputs.shape[:-3] + (-1, 3))


    print("input images: ", images.shape)
    print("wind info: ", windinfo.shape)
    print("outputs: ", outputs.shape)

    out = np.argmax(outputs, axis = 2)
    print(outputs.shape)

    unique, counts = np.unique(out, return_counts=True)
    weights = dict(zip(unique, counts))
    total = np.sum(counts)


    #images = tf.cast(images, tf.float16)
    #windinfo = tf.cast(windinfo, tf.float16)
    #outputs = tf.cast(outputs, tf.float16)


    return images, windinfo, outputs, weights, total


def build_model(input1_shape, input2_shape):
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
    deconv = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2,2), activation="relu", padding="same")(deconv)
    deconv = Reshape((16 * 16, 3))(deconv)
    deconv = Activation('softmax')(deconv)

    model = Model(inputs=[model1.input, inp2], outputs=deconv)

    model.compile(loss='categorical_crossentropy',               # because output pixels can have 0s or 1s
                  optimizer='adam')                         # standard


    return model


def predict(model=None, data=None, n_examples=5):
    """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""

    if not data:
        images, windinfo, outputs, weights, total = load_data()
    else:
        images, windinfo, outputs = data

    if not model:
        model = tf.keras.models.load_model("saved_models\\safetySafe")
    #X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
    X1 = images[:n_examples]                                        # more clever way to write it down
    X2 = windinfo[:n_examples]

    results = model.predict([X1, X2])                        # outputs 61x61x2

    # translate the 5 channel input back to displayable images
    orig_img = np.zeros((len(X1), 255, 255))

    #print(np.argmax(results[0], axis = 1).shape)
    #print(np.argmax(results[0], axis = 1))

    for i, image in enumerate(X1):
        print("reconstructing image ", i, "/", n_examples)
        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                for idx, item in enumerate(cell):
                    #print(i, y, x, idx)
                    if item == 1:
                        orig_img[i][y][x] = idx


    # display input images and the 2 waypoint output images (from 2 channels)
    for i in range(len(results)):
        plt.imshow(orig_img[i])
        plt.title("input image")
        plt.show()


        non_wp_res, dig_img_res, drive_img_res = np.dsplit(results[i], 3)                  # depth split of 2 channel image
        non_wp_out, dig_img_out, drive_img_out = np.dsplit(outputs[i], 3)                  # depth split of 2 channel image
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(np.reshape(dig_img_res, newshape=(16, 16)))
        axarr[0,0].set_title("dig waypoints image NN output")
        axarr[0,1].imshow(np.reshape(drive_img_res, newshape=(16, 16)))
        axarr[0,1].set_title("drive image NN output")
        axarr[1,0].imshow(np.reshape(dig_img_out, newshape=(16, 16)))
        axarr[1,0].set_title("dig image desired")
        axarr[1,1].imshow(np.reshape(drive_img_out, newshape=(16, 16)))
        axarr[1,1].set_title("drive image desired")
        plt.show()



if __name__ == "__main__":
    #predict()                          # predict with model loaded from file
    #exit()

    images, windinfo, outputs, weights, total = load_data()

    model = build_model(images[0].shape, windinfo[0].shape)
    print(model.summary())
    #exit()

    class_weights = np.zeros((outputs.shape[1], 3))
    class_weights[:, 0] += (1 / weights[0]) * (total) / 2.0
    class_weights[:, 1] += (1 / weights[1]) * (total) / 2.0
    class_weights[:, 2] += (1 / weights[2]) * (total) / 2.0

    model.fit([images, windinfo],                   # list of 2 inputs to model
              outputs,
              batch_size=16,
              epochs=2,
              shuffle=True,
              validation_split=0.2,
              class_weight = class_weights)                         # mix data randomly

    #predict(model=model)

    save(model, "safetySafe")                       # utils
