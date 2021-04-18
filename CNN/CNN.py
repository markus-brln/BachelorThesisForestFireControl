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
    model1.add(MaxPooling2D(pool_size=(3, 3)))
    model1.add(Flatten())
    model1.add(Dense(24, activation='sigmoid'))             # 1D feature vector

    # TODO can we insert the vector in a more raw form? (well, with length 13 it's maybe not that nice)
    model2 = Sequential()                                   # TODO verify that this is how it should be done
    model2.add(Dense(8, input_shape=input2_shape))          # just encode the vector of 13 in a shorter one

    model_concat = concatenate([model1.output, model2.output], axis=1)
    deconv = Dense(64, activation='relu')(model_concat)
    deconv = keras.layers.Reshape((8, 8, 1))(deconv)        # TODO smoother way to upscale?
    deconv = Conv2DTranspose(64, (2, 2), padding='same')(deconv)
    deconv = keras.layers.Reshape((64, 64, 1))(deconv)      # 64x64 output, has to be translated to 255x255 again?

    model = Model(inputs=[model1.input, model2.input], outputs=deconv)

    model.compile(loss='binary_crossentropy',               # because output pixels can have 0s or 1s
                  optimizer='adam')                         # standard

    return model


def load_model_and_predict():
    images, windinfo, outputs = load_data()
    model = keras.models.load_model("safetySafe")
    #X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
    X1 = images[0:1]                                        # more clever way to write it down
    X2 = windinfo[0:1]
    result = model.predict([X1, X2])                        # outputs 61x61x1 for now

    plt.imshow(np.reshape(result, (64, 64)))
    plt.show()


if __name__ == "__main__":
    #load_model_and_predict()
    #exit()
    images, windinfo, outputs = load_data()

    print(outputs[0].shape)
    model = build_model(images[0].shape, windinfo[0].shape)
    print(model.summary())

    X_train_1 = images
    X_train_2 = windinfo
    Y_train = np.random.randint(0, 2, (129, 64, 64, 1)) # TODO how to use the 255x255 image?

    model.fit([images, windinfo],                   # list of 2 inputs to model
              Y_train,
              batch_size=16,
              epochs=20,
              shuffle=True)                         # mix data randomly

    save(model, "safetySafe")                       # utils
