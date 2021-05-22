import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.engine.saving import model_from_json
from tensorflow.keras import layers, models

def unison_shuffled_copies(a, b):
    # from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def plot_history(history):
    # function taken from https://github.com/musikalkemist/DeepLearningForAudioWithPython
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    #axs[0].plot(history.history["accuracy"], label="train accuracy")
    #axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    #axs[0].set_ylabel("Accuracy")
    #axs[0].legend(loc="lower right")
    #axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()



# https://github.com/musikalkemist/DeepLearningForAudioWithPython
# https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf
def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add dimensions to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ..., np.newaxis]

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    #predicted_index = np.argmax(prediction, axis=1)
    plt.imshow(X)
    plt.show()

    plt.imshow(prediction)
    plt.show()


#### Saving a Network
def save(model, filename):
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    model_json = model.to_json()
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    with open('saved_models\\' + filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('saved_models\\' + filename + ".h5")
    print("saving model " + filename)
    #model.save('saved_models\\' + filename)


#### Loading a Network
def load(filename):
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("loading model " + filename)
    # load json and create model
    json_file = open('saved_models\\' + filename + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights('saved_models\\' + filename + ".h5")
    print("Loaded model from disk")

    #json_model_file = open(os.path.join(self.model_path, name + '.json'), "r").read()
    #model = model_from_json(open('saved_models/' + filename).read())
    #model.load_weights(os.path.join(os.path.dirname('saved_models/' + filename), 'model_weights.h5'))

    #model = tf.keras.models.load_model('saved_models/' + filename)
    return model


def save_history(file_name, history):
    np.save(file_name, history.history)


def load_history(filename):
    return np.load(filename, allow_pickle=True).item()

