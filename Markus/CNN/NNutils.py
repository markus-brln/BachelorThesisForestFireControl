import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

    #X = np.reshape(X, (828))
    #plt.plot(X)
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

