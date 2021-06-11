from keras import Model, Sequential
from keras.layers import concatenate, Dense, Conv2D, Flatten, Conv2DTranspose, Input, Reshape
import numpy as np

def build_model(input1_shape, input2_shape, output_shape):
    inp1 = Sequential()
    inp1.add(Conv2D(input_shape=input1_shape, filters=16, kernel_size=(3, 3),
                      activation="relu", padding="same"))
    inp1.add(Flatten())
    inp1.add(Dense(24, activation='sigmoid'))             # feature vector

    inp2 = Input(input2_shape)

    model_concat = concatenate([inp1.output, inp2], axis=1)
    print(model_concat.shape)
    print(type(model_concat))
    deconv = Dense(64, activation='relu')(model_concat)
    deconv = Reshape((8, 8, 1))(deconv)        # TODO smoother way to upscale to param output_shape?
    deconv = Conv2DTranspose(64, (2, 2), padding='same')(deconv)
    deconv = Reshape((64, 64, 1))(deconv)      # 64x64 output

    model = Model(inputs=[inp1.input, inp2], outputs=deconv)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')

    return model

input1 = np.random.randint(0, 2, (100, 255, 255, 5))        # 5 channel image
input2 = np.random.randint(0, 2, (100, 13))                 # simple vector
output = np.random.randint(0, 2, (100, 64, 64, 1))          # 1 channel



model = build_model(input1[0].shape, input2[0].shape, output[0].shape)
print(model.summary())


model.fit([input1, input2],
          output,
          batch_size=16,
          epochs=20,
          shuffle=True)