import keras.backend
import numpy as np
from keras import Input, Model, Sequential
from keras.layers import concatenate, Dense, Embedding, GlobalAveragePooling1D, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, Deconvolution2D
from NNutils import *

def load_data():
    images = np.load("images.npy", allow_pickle=True)
    windinfo = np.load("windinfo.npy", allow_pickle=True)
    outputs = np.load("outputs.npy", allow_pickle=True)

    return images, windinfo, outputs

def build_model(input1_shape, input2_shape):
    # idea/"tutorial" from:
    # https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
    model1 = Sequential()
    model1.add(Conv2D(input_shape=input1_shape, filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.25))

    model1.add(Conv2D(filters=32, kernel_size=(2, 2), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.25))
    model1.add(Dense(32, activation='relu'))                # 1D feature vector
    model1.add(Flatten())

    inp2 = Input(input2_shape)

    model_concat = concatenate([model1.output, inp2], axis=1)
    print(model_concat.shape)
    deconv = Dense(8 * 8 * 2, activation='relu')(model_concat)     # re-compress feature vector
    deconv = keras.layers.Reshape((8, 8, 2))(deconv)

    # tutorial on deconv layer:
    # https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
    deconv = Conv2DTranspose(32, kernel_size=(2, 2), strides=(1, 1), padding='same')(deconv)
    deconv = Conv2DTranspose(64, kernel_size=(2, 2), strides=(1, 1), padding='same')(deconv)
    deconv = Conv2DTranspose(128, kernel_size=(2, 2), strides=(1, 1), padding='same')(deconv)

    deconv = keras.layers.Reshape((64, 64, 2))(deconv)
    model = Model(inputs=[model1.input, inp2], outputs=deconv)

    model.compile(loss='binary_crossentropy',               # because output pixels can have 0s or 1s
                  optimizer='adam')                         # standard

    return model



def load_model_and_predict():
    images, windinfo, outputs = load_data()
    model = keras.models.load_model("saved_models\\safetySafe")
    #X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
    X1 = images[:9]                                        # more clever way to write it down
    X2 = windinfo[:9]
    results = model.predict([X1, X2])                        # outputs 61x61x2

    # translate the 5 channel input back to displayable images
    orig_img = np.zeros((len(X1), 255, 255))
    for i, image in enumerate(X1):
        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                for idx, item in enumerate(cell):
                    if item == 1:
                        orig_img[i][y][x] = idx


    # display input images and the 2 waypoint output images (from 2 channels)
    for i in range(len(results)):
        plt.imshow(orig_img[i])
        plt.show()
        dig_img, drive_img = np.dsplit(results[i], 2)                  # depth split of 2 channel image
        dig_img_real, drive_img_real = np.dsplit(outputs[i], 2)                  # depth split of 2 channel image
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(np.reshape(dig_img, newshape=(64, 64)))
        axarr[0,0].set_title("dig waypoints image NN output")
        axarr[0,1].imshow(np.reshape(drive_img, newshape=(64, 64)))
        axarr[0,1].set_title("drive image NN output")
        axarr[1,0].imshow(np.reshape(dig_img_real, newshape=(64, 64)))
        axarr[1,0].set_title("dig image desired")
        axarr[1,1].imshow(np.reshape(drive_img_real, newshape=(64, 64)))
        axarr[1,1].set_title("drive image desired")
        plt.show()


if __name__ == "__main__":
    # TODO what should happen if we clicked on the same spot twice, i.e. only one waypoint recorded?
    #load_model_and_predict()
    #exit()

    images, windinfo, outputs = load_data()

    model = build_model(images[0].shape, windinfo[0].shape)
    print(model.summary())


    model.fit([images, windinfo],                   # list of 2 inputs to model
              outputs,
              batch_size=1,
              epochs=500,
              shuffle=True)                         # mix data randomly

    #save(model, "safetySafe")                       # utils
