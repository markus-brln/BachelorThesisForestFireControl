import numpy as np
from keras import Input, Model, Sequential
from keras.layers import concatenate, Dense, Embedding, GlobalAveragePooling1D, Conv2D, Flatten, MaxPooling2D


def load_data():
    images = np.load("images.npy", allow_pickle=True)
    windinfo = np.load("windinfo.npy", allow_pickle=True)
    outputs = np.load("outputs.npy", allow_pickle=True)

    return images, windinfo, outputs

def build_model(input1_shape, input2_shape):
    # idea/"tutorial" from:
    # https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
    model1 = Sequential()
    model1.add(Conv2D(input_shape=input1_shape, filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Flatten())
    model1.add(Dense(16, activation='sigmoid'))

    model2 = Sequential()
    model2.add(Embedding(20, 10, trainable=True))
    model2.add(GlobalAveragePooling1D())
    model2.add(Dense(8, activation='sigmoid'))

    model_concat = concatenate([model1.output, model2.output], axis=-1)
    model_concat = Dense(1, activation='softmax')(model_concat)
    model = Model(inputs=[model1.input, model2.input], outputs=model_concat)

    model.compile(loss='binary_crossentropy',               # because output pixels can have 0s or 1s
                  optimizer='adam')                         # standard

    return model




if __name__ == "__main__":
    images, windinfo, outputs = load_data()


    model = build_model(images[0].shape, windinfo[0].shape)
    print(model.summary())

    #X_train_1 = np.random.randint(0, 20, (200, input1_shape[0],input1_shape[1],input1_shape[2]))
    X_train_1 = images
    X_train_2 = windinfo
    #X_train_2 = np.random.randint(0, 20, (129, 256))
    Y_train = np.random.randint(0, 2, 129)

    model.fit([X_train_1, X_train_2],
              Y_train,
              batch_size=16,
              epochs=20,
              verbose=True,
              shuffle=True)




    #checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc',
    #                             save_best_only=True, verbose=2)
    #early_stopping = EarlyStopping(monitor="val_loss", patience=5)

    #merged_model.fit([x1, x2], y=y, batch_size=384, epochs=200,
    #                 verbose=1, validation_split=0.1, shuffle=True)#,
    #callbacks=[early_stopping, checkpoint])