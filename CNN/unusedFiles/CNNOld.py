import glob

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations, models, optimizers, losses, Input, Sequential, metrics, Model
from tensorflow.keras.layers import Convolution1D, MaxPool1D, Dropout, GlobalMaxPool1D, Dense, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from npy_512_files import data_handling


def plot_history(history):
    # function taken from https://github.com/musikalkemist/DeepLearningForAudioWithPython
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


# def create_model(input_shape, output_shape, batch_size):
def create_model(n_timesteps, n_features, n_outputs):
    print("creating model")
    # setting all layers to 16 bit by default since it is the precision
    keras.backend.set_floatx('float32')
    dtype = 'float32'

    # regularization l2 & dropout
    # https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-2-a5bd7a4e7e5a

    # more complicated model, copied from https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_mitbih.py
    '''model = Sequential([
        Convolution1D(filters=16, input_shape=(n_timesteps, n_features), kernel_size=5, activation=activations.relu, padding="valid"),
        Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid"),
        MaxPool1D(pool_size=2),
        Dropout(rate=0.1),
        Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
        Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
        MaxPool1D(pool_size=2),
        Dropout(rate=0.1),
        Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
        Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
        MaxPool1D(pool_size=2),
        Dropout(rate=0.1),
        Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid"),
        Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid"),
        GlobalMaxPool1D(),
        Dropout(rate=0.2),
        Dense(64, activation=activations.relu, name="dense_1"),
        Dense(64, activation=activations.relu, name="dense_2"),
        Dense(n_outputs, activation=activations.softmax, name="dense_3_mitbih")
    ])'''
    # basic model
    model = Sequential([
        Convolution1D(input_shape=(n_timesteps, n_features), kernel_size=3, filters=64, activation='relu',
                      kernel_regularizer='l2', name="conv1"),
        Convolution1D(kernel_size=3, filters=64, activation='relu', kernel_regularizer='l2', name="conv2"),
        MaxPool1D(pool_size=4, strides=4),  # reduce size of the input by 4
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(), name="dense1"),
        Dropout(0.2),
        Dense(n_outputs, activation='softmax', name="dense2")
    ])

    return model


# https://github.com/musikalkemist/DeepLearningForAudioWithPython
# https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf
def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add dimensions to input data for sample - model.predict() expects a 3d array in this case
    X = X[np.newaxis, ..., np.newaxis]

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("prediction values: ", prediction)
    print("Target: {}, Predicted label: {}".format(y, predicted_index))

    X = np.reshape(X, (828))
    plt.plot(X)
    plt.show()


#### Saving a Network
def save(model, filename):
    print("saving model " + filename)
    model.save('saved_models\\' + filename)


#### Loading a Network
def load(filename):
    print("loading model " + filename)
    model = tf.keras.models.load_model('saved_models\\' + filename)
    return model


def save_history(file_name, history):
    np.save(file_name, history.history)


def load_history(filename):
    return np.load(filename, allow_pickle=True).item()


def retrain(model_name="cnn_general_5_groups", patient_number=None, saving=True):
    # https://keras.io/guides/transfer_learning/
    if not patient_number:
        print("Patient number needed for retraining")
        exit(-1)

    # load data from .npy files all_512_data_part1/2 and all_512_ann_vec_part1/2
    X, y = data_handling.load_data(patient_number=patient_number)

    # shuffle to avoid uneven distribution of conditions (this is a constraint in the real world, where you cannot
    # shuffle time... just thought it would be better here
    X, y = data_handling.unison_shuffled_copies(X, y)

    # 20% training, 80% test data, so only the first 6 minutes of the recording can be used for retraining
    # split here must be the same as in train_on_patient_only()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = load(model_name)
    print(model.summary())

    retrain_model = tf.keras.Sequential()
    for layer in model.layers[:-3]:
        layer.trainable = False
        retrain_model.add(layer)
    # add a dense layer(s) that should be retrained
    retrain_model.add(Dense(32, activation=activations.relu, name="dense_1"))
    retrain_model.add(Dense(len(y_train[0]), activation=activations.softmax, name="dense_2"))

    # retrain_model.add(Dense(len(y_train[0]), activation='sigmoid'))
    retrain_model.compile(optimizer='adam', loss='mean_squared_error',
                          metrics=['accuracy'])
    print(retrain_model.summary())

    batch_size = 128
    history = retrain_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=batch_size,
                                verbose=1, metrics=[metrics.FalseNegatives(), metrics.FalsePositives()])

    # function to plot the history dict that model.fit() produces based on the param metrics in model.compile()
    # plot_history(history)

    print("evaluating on all test hearbeats:")
    test_loss, test_acc = retrain_model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest accuracy:', test_acc)

    if saving:
        save(model=model, filename="cnn_retrained_patient_" + str(patient_number))

    return test_acc

    exit()


def train_general_model(saving=True):
    # load data from .npy files all_512_data_part1/2 and all_512_ann_vec_part1/2
    X, y = data_handling.load_data()

    # data_handling.visualize_data_balance(y)

    # balancing data to a certain amount of samples, using over/undersampling
    X, y = data_handling.balance_data(X=X, y=y, samples_per_class=2000)

    # shuffle input and output in the same way
    X, y = data_handling.unison_shuffled_copies(X, y)

    # extract 20% validation and test data, test data will be used for a final test after training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)  # (0.8 % 0.25 = 0.2)

    # add axis to the inputs since 1D CNN wants 3D input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(np.shape(X_train))
    print(np.shape(y_train))

    # OPTION: load and play around with a model that has been trained already
    # use_existing_model("general_cnn", X, y)

    # OPTION: retrain the last layer(s) of the model on other data, in our case
    #         transfer learning on one patient
    # retrain((X_train, y_train, X_test,  y_test, X_val, y_val))

    # create the 1D CNN Model
    # from https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = create_model(n_timesteps, n_features, n_outputs)

    # optimizer = tf.keras.optimizers.Adam(lr=.0001)
    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    # short form:
    model.compile(optimizer='adam', loss='mean_squared_error',
                  # we need to know why we use a certain loss function
                  metrics=[metrics.FalseNegatives(), metrics.FalsePositives()])  # also an interesting option

    print(model.summary())

    batch_size = 128
    # 25 epochs, after that there was no improvement seen
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=batch_size, verbose=1)
    save_history("histories\\general_cnn_5_groups", history)

    # function to plot the history dict that model.fit() produces based on the param metrics in model.compile()
    plot_history(history)

    print("evaluating on all test hearbeats:")

    # final evaluation with test data separate, to further avoid overfitting (may be overkill for what we are doing here)
    model.evaluate(X_test, y_test, verbose=2)

    if saving:
        save(model=model, filename="cnn_general_5_groups")

    exit()


def train_on_patient_only(patient_number, saving=True):
    # load data from .npy files all_512_data_part1/2 and all_512_ann_vec_part1/2
    X, y = data_handling.load_data(patient_number=patient_number)

    # shuffle to avoid uneven distribution of conditions (this is a constraint in the real world, where you cannot
    # shuffle time... just thought it would be better here
    X, y = data_handling.unison_shuffled_copies(X, y)

    # 20% training, 80% test data, so only the first 6 minutes of the recording can be used for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)  # (0.8 % 0.25 = 0.2)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = create_model(n_timesteps, n_features, n_outputs)

    # optimizer = tf.keras.optimizers.Adam(lr=.0001)
    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    # short form:
    model.compile(optimizer='adam', loss='mean_squared_error',
                  # we need to know why we use a certain loss function
                  metrics=['accuracy'])  # metrics.FalseNegatives, metrics.FalsePositives also an interesting option

    print(model.summary())

    batch_size = 128
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=batch_size,
                        verbose=1, metrics=[metrics.FalseNegatives(), metrics.FalsePositives()])

    # function to plot the history dict that model.fit() produces based on the param metrics in model.compile()
    # plot_history(history)

    print("evaluating on all test hearbeats:")

    # final evaluation with test data separate, to further avoid overfitting (may be overkill for what we are doing here)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)

    if saving:
        save(model=model, filename="cnn_patient_" + str(patient_number))

    return test_acc


def train_retrain_and_single_all_patients():
    """trains single patient model and retrain model for all patients"""
    patients = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124,
                200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230,
                231, 232, 233, 234]

    retrain_performance = []
    single_performance = []
    for pnum in patients:
        retrain_performance.append(retrain(model_name="cnn_general_5_groups", patient_number=pnum))
        single_performance.append(train_on_patient_only(patient_number=pnum))

    tfile = open("retrain_vs_single_acc.txt", mode="w")

    tfile.write("patients: \n" + str(patients) + "\n")
    tfile.write("retrain performance: \n" + str(retrain_performance) + "\n")
    tfile.write("single performance: \n" + str(single_performance) + "\n")


def retrain_vs_single_fnfpacc():
    """only loads models trained by retrain_vs_single(), recompiles them to output false positives,
    false negatives and accuracy and then tests them
    outputs txt file with obtained performance values"""

    tfile = open("retrain_vs_single_fnfpacc.txt", mode="w")
    tfile.write(
        "mean squared error, rates of false positives and false negatives per patient and accuracy for\n1) single\n2) retrain\n\n")
    tfile.write("patient  mse               fp                  fn                 acc\n")

    for single_name, retrain_name in zip(glob.glob("saved_models/cnn_patient*"),
                                         glob.glob("saved_models/cnn_retrained*")):  # exclude general model
        print(single_name[-3:])
        print(retrain_name[-3:])
        X, y = data_handling.load_data(patient_number=int(single_name[-3:]))
        X, y = data_handling.unison_shuffled_copies(X, y)
        X = X[..., np.newaxis]  # still necessary to make 3D
        single_model = tf.keras.models.load_model(single_name)
        retrain_model = tf.keras.models.load_model(retrain_name)

        # recompile
        print("recompiling")
        single_model.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[metrics.FalsePositives(), metrics.FalseNegatives(), 'acc'])
        retrain_model.compile(optimizer='adam', loss='mean_squared_error',
                              metrics=[metrics.FalsePositives(), metrics.FalseNegatives(), 'acc'])

        # evaluate on all data of the patient
        perf_single = single_model.evaluate(X, y, verbose=2)
        perf_retrain = retrain_model.evaluate(X, y, verbose=2)

        print(perf_single)
        print(perf_retrain)

        tfile.write(str(single_name[-3:]) + " " + str(perf_single[0]) + " " + str(perf_single[1] / len(y)) + " " + str(
            perf_single[2] / len(y)) + " " + str(perf_single[3]) + "\n")
        tfile.write(
            str(retrain_name[-3:]) + " " + str(perf_retrain[0]) + " " + str(perf_retrain[1] / len(y)) + " " + str(
                perf_retrain[2] / len(y)) + " " + str(perf_retrain[3]) + "\n\n")


if __name__ == "__main__":
    retrain_vs_single_fnfpacc()
    # X, y = data_handling.load_data(patient_number=106)
    # data_handling.visualize_data_balance(y)
    # train_on_patient_only(patient_number=106)
    # train_general_model()
    # retrain(model_name="cnn_general_5_groups", patient_number=106)
    # X, y = data_handling.load_data(patient_number=106)
    # X = X[..., np.newaxis]
    # test_model("cnn_patient_106", X, y)
