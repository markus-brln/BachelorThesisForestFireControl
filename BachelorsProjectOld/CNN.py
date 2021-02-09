
import os
import pickle
import random
import os.path
import csv
import sys
import time

import pygame
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from os import path
from __main__ import args
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical

EXTENDED = args.extended
HIST_LEN = args.history_length
FPS = args.fire_propagation_speed

'''
Makes a function 'getch()' which gets a char from user without waiting for enter
'''
try:
    # Windows
    from msvcrt import getch
except ImportError:
    # UNIX
    def getch():
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

'''
Extract memory from folder
'''


def extract_from_memory(path):
    with open(path + random.choice(os.listdir(path)), "rb") as pf:
        (state, action, done) = pickle.load(pf)
        pf.close()
    return state, action, done


class CNN:
    def __init__(self, sim, name="no_name", environment="forest", size=10, extended=False, length=0, verbose=True):
        # Constants and such
        self.sim = sim
        self.name = name
        self.METADATA = sim.METADATA
        self.action_size = self.sim.n_actions
        self.DEBUG = sim.DEBUG
        self.verbose = verbose
        self.environment = environment
        self.size = size
        self.depth = self.get_depth()
        self.path = "FPS=" + str(FPS) + "/" + str(self.environment) + "/" + "size" + str(self.sim.W.WIDTH) + "/" + str(
            len(self.sim.W.agents)) + "/"
        self.path_logs = "FPS=" + str(FPS) + "/" + "Extended=" + str(EXTENDED) + "/" + "History_length=" + str(HIST_LEN)\
                         + "/" + str(self.environment) + "/" + "size" + str(self.sim.W.WIDTH) + "/" + str(
            len(self.sim.W.agents)) + "/"

        # Information about number of fires that are contained, saved to a file
        self.logs = 0

        # Information about the time it takes ot isolate a fire
        self.time = 0

        # CNN Parameter alpha (learning rate)
        self.alpha = self.METADATA['alpha']  # learning rate

        # Creating the CNN with function .make_network()
        self.model = self.make_network()

        # Print Constants
        if self.verbose:
            width, height = self.METADATA['width'], self.METADATA['height']
            print("\n\t[Parameters]")
            print("[size]", f"{width}x{height}")

    '''
    Performing runs in which a Human has to provide training input (delivering data)
    '''

    def create_memories(self, train_memories, validation_memories):
        iso = 0
        # Loops over amount of episodes which is specified in input arguments
        for episode in range(train_memories + validation_memories):
            # Print some information about the episode
            print(f"[Episode {episode + 1}]")
            # Initialize the done flag, in order that the system knows when the episode is finished
            done = False
            # Initialize the state
            state = self.sim.reset()
            rot90_1, rot90_2, rot90_3 = self.sim.W.get_augmented_state()
            # Maps keyboard to number for a certain action
            key_map = {'w': 0, 's': 1, 'd': 2, 'a': 3, ' ': 4, 'n': 5}
            while not done and not self.sim.W.FIRE_ISOLATED:
                for agent in self.sim.W.agents:
                    agent.active = True
                    self.sim.W.active = agent
                    self.sim.render()
                    print("W A S D to move, Space to dig,")
                    print("'n' to wait, 'q' to quit.\n")
                    # Get action from keyboard
                    char = getch()
                    if char == 'q':
                        return "Cancelled"
                    elif self is not None and len(self.sim.W.border_points) == 0:
                        done = True
                    elif char in key_map:
                        action = key_map[char]
                        # Do action, observe environment
                        sprime, score, done, _ = self.sim.step(action)

                        # Store experience in memory if agent is not dead and the performed action is not waiting!
                        if not done and not agent.dead:
                            if episode < train_memories:
                                self.remember(state, action, True, done, agent.save_move)
                                self.remember_augmented_data(rot90_1, rot90_2, rot90_3, action, True, done, agent.save_move)
                            else:
                                self.remember(state, action, False, done, agent.save_move)
                                self.remember_augmented_data(rot90_1, rot90_2, rot90_3, action, False, done, agent.save_move)
                        # Current state is now next state
                        state = sprime
                        rot90_1, rot90_2, rot90_3 = self.sim.W.get_augmented_state()
                        if not self.sim.W.RUNNING or self.sim.W.FIRE_ISOLATED or agent.dead:
                            break
                    else:
                        print("Invalid action, not good for collecting memories.\n")
                    agent.active = False
                    # if not self.sim.W.RUNNING or self.sim.W.FIRE_ISOLATED:
                    #     break
            if self.sim.W.FIRE_ISOLATED:
                iso += 1
            print("ISO: ", iso, "/", episode+1)

            # # Store the observed experience for last move before episode is done
            if not action == 5 and not done:
                if episode < train_memories:
                    self.remember(state, action, True, done, agent.save_move)
                else:
                    self.remember(state, action, False, done, agent.save_move)

    '''
    Train the Keras CNN model with samples taken from the memory
    '''

    def train(self):
        percent_of_data = 1
        # Train variables
        train_path = "train/" + self.path
        train_states_batch = list()
        train_action_batch = list()
        train_batch = list()
        # Validation variables
        validation_path = "validate/" + self.path
        validation_states_batch = list()
        validation_action_batch = list()
        validation_batch = list()

        # Amount of available memories for current environment and size
        amount_of_memories = round(len(
            [name for name in os.listdir(train_path) if
             os.path.isfile(os.path.join(train_path, name))])) * percent_of_data
        print("\n[train_memories]", amount_of_memories)
        validation_memories = round(len(
            [name for name in os.listdir(validation_path) if
             os.path.isfile(os.path.join(validation_path, name))]) * percent_of_data)
        print("[validation_memories]", validation_memories)

        i = 0
        # Extract train memories
        while i < amount_of_memories:
            x = extract_from_memory(train_path)
            train_batch.append(x)
            i = i + 1

        i = 0
        # Extract validation memories
        while i < validation_memories:
            x = extract_from_memory(validation_path)
            validation_batch.append(x)
            i = i + 1

        # Store the states and the corresponding action for this batch
        for state, action, done in train_batch:
            state = self.configure_state(state, EXTENDED, HIST_LEN)
            train_states_batch.append(state[0])
            output = to_categorical(action, num_classes=6, dtype='float32')
            train_action_batch.append(output)

        for state, action, done in validation_batch:
            state = self.configure_state(state, EXTENDED, HIST_LEN)
            validation_states_batch.append(state[0])
            output = to_categorical(action, num_classes=6, dtype='float32')
            validation_action_batch.append(output)

        # Convert the batch into numpy arrays for Keras and fit the model
        data_input = np.array(train_states_batch)
        data_output = np.array(train_action_batch)
        val_data_input = np.array(validation_states_batch)
        val_data_output = np.array(validation_action_batch)

        # To prevent overfitting and choose best model according to data on which is only evaluated
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        mc = ModelCheckpoint('best_model', monitor='val_accuracy', mode='max', save_best_only=True)
        tensorboard = TensorBoard(log_dir='TB_log' + self.path_logs)
        # Final command to train model
        self.model.fit(data_input, data_output, validation_data=(val_data_input, val_data_output), epochs=1000,
                       callbacks=[es, mc], verbose=True)

    '''
    Test the behavior learned from training on completely new situations
    without supervision and see whether agent will isolate the fire.
    '''

    def test(self, n_rounds, load, view):
        if load:
            model = tf.keras.models.load_model("Saved_Models/" + self.path + 'Best_Model.h5')
        else:
            model = self.model
        self.save_model()
        state_old = None
        # Start n episodes to see how the model behaves
        for episode in range(n_rounds):
            t = 0
            # Initialize the done flag, in order that the system knows when the episode is finished
            done = False
            # Initialize the state, and reshape because Keras expects the first dimension to be the batch size
            state = self.sim.reset()
            save = state
            state = self.configure_state(state, EXTENDED, HIST_LEN)
            if self.sim.W.board is not None:
                pygame.event.get()
            self.sim.W.performance.episode = episode
            while not done:
                self.sim.render(view, False)
                for agent in self.sim.W.agents:
                    agent.active = True
                    self.sim.W.active = agent
                    if not self.sim.W.RUNNING:
                        done = True
                        self.sim.render(view, False)
                        break
                    softmax_char = model.predict(state)[0]
                    action = np.argmax(softmax_char)
                    sprime, score, done, _ = self.sim.step(action)
                    state_old = save
                    save = sprime
                    sprime = self.configure_state(sprime, EXTENDED, HIST_LEN)
                    # Current state is now next state
                    state = sprime
                    agent.active = False
                t += 1
            self.sim.render(view, False)
            # input("Press Enter to continue...")

            # Keep track of sucesfull(means fire is isolated) episodes
            if self.sim.W.FIRE_ISOLATED:
                self.sim.W.performance.cumulative_burnt += self.sim.W.get_percent_burnt()
                self.sim.W.performance.amount_fires_isolated += 1
        self.logs = self.sim.W.performance.amount_fires_isolated
        self.time = self.sim.W.performance.cumulative_burnt / self.sim.W.performance.amount_fires_isolated
        self.write_logs()
        print(f"Amount of times the agent isolated the fire : {self.logs}")
        print(f"Average time needed to isolate fire: {self.sim.W.get_percent_burnt()}")
        if self.sim.W.performance.amount_fires_isolated == 100:
            self.save_model()
        self.sim.W.performance.clear()

        return self.logs

    '''
    Store an experience in /train/environment/sizeX.
    '''

    def remember(self, state, action, train, done, save):
        if save:
            # create folder names
            name = self.sim.get_name(self.sim.W.WIDTH, str(self.environment))

            # TRAIN DATA
            if train:
                # makes sure correct folder is used for storing examples
                if not os.path.exists("train/" + self.path):
                    os.makedirs("train/" + self.path)
                # add counter to the end of the name of files
                counter = 0
                while os.path.isfile("train/" + self.path + name):
                    if counter > 0:
                        n_digits_to_delete = len(str(counter))
                        name = name[:-n_digits_to_delete]
                    name = name + str(counter)
                    counter += 1
                # store object with current state, action and whether agent has died in a file
                with open("train/" + self.path + name, "wb") as pf:
                    pickle.dump((state, action, done), pf)
                    pf.close()

            # VALIDATION DATA
            if not train:
                if not os.path.exists("validate/" + self.path):
                    os.makedirs("validate/" + self.path)
                # add counter to the end of the name of files
                counter = 0
                while os.path.isfile("validate/" + self.path + name):
                    if counter > 0:
                        n_digits_to_delete = len(str(counter))
                        name = name[:-n_digits_to_delete]
                    name = name + str(counter)
                    counter += 1
                with open("validate/" + self.path + name, "wb") as pf:
                    pickle.dump((state, action, done), pf)
                    pf.close()

    '''
    Remembers the augmented data. Which is just flipped states + according action. 
    '''

    def remember_augmented_data(self, rot90_1, rot90_2, rot90_3, action, train, done, save):
        key_map_rot90_1 = {0: 2, 1: 3, 2: 1, 3: 0, 4: 4, 5: 5}
        key_map_rot90_2 = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 5: 5}
        key_map_rot90_3 = {0: 3, 1: 2, 2: 0, 3: 1, 4: 4, 5: 5}

        self.remember(rot90_1, key_map_rot90_1[action], train, done, save)
        self.remember(rot90_2, key_map_rot90_2[action], train, done, save)
        self.remember(rot90_3, key_map_rot90_3[action], train, done, save)

    '''
    Create the Deep Convolutional Neural Network
    '''

    def make_network(self):
        input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT, self.depth)
        layers = [
            Conv2D(100, kernel_size=2, activation='relu', input_shape=input_shape),

            MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid', data_format=None),

            Conv2D(100, kernel_size=2, activation='relu'),

            Conv2D(25, kernel_size=2, activation='relu'),

            Flatten(),

            Dense(units=self.action_size, activation='softmax'),

        ]
        adam = Adam(lr=self.alpha, clipvalue=1)
        model = Sequential(layers)
        # Compile model with categorical crossentropy error loss, the Adam optimizer
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      # And DEPTHan Adam optimizer with gradient clipping
                      optimizer=adam)
        if self.verbose:
            model.summary()
        return model

    '''
    Write the amount of contained fires and time needed to isolate to a log file
    '''

    def write_logs(self):
        # Get name for Log file
        name = str(self.sim.W.WIDTH) + "-" + str(self.environment) + "-" + str(len(self.sim.W.agents)) + ".csv"
        exists = True
        # If the folder doesn't exist, create it
        if not os.path.exists("Logs/" + self.path_logs):
            exists = False
            os.makedirs("Logs/" + self.path_logs)
        # field names
        fields = ['Accuracy', 'Percent Burnt']
        # data rows of csv file
        rows = [[self.logs, round(self.time, 5)]]
        # writing to csv file
        with open("Logs/" + self.path_logs + name, 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            if not exists:
                # writing the fields
                csvwriter.writerow(fields)
            # writing the data rows
            csvwriter.writerows(rows)

    '''
    Saves the model in the designated folder except if one already exists. 
    '''

    def save_model(self):
        if not os.path.exists("Saved_Models/" + self.path):
            os.makedirs("Saved_Models/" + self.path)
        if not path.exists("Saved_Models/" + self.path + 'Best_Model.h5'):
            self.model.save("Saved_Models/" + self.path + 'Best_Model.h5')

    '''
       Returns next active agent
    '''

    def next_agent(self):
        if self.sim.W.agents.index(self.sim.W.active) + 2 > len(self.sim.W.agents):
            return 0
        else:
            return self.sim.W.agents.index(self.sim.W.active) + 1

    '''
    returns the correct state given the agent history settings
    '''

    def configure_state(self, state, other_agent, history_length):
        # return np.reshape(state, [1] + list(state.shape))
        state = np.dstack((np.dstack((state))))
        if history_length == 0:
            state = self.make_stack(state, None, None)
            return np.reshape(state, [1] + list(state.shape))
        if history_length == 3 and not other_agent:
            state = self.make_stack(state, 2, None)
            return np.reshape(state, [1] + list(state.shape))
        if history_length == 3 and other_agent:
            state = self.make_stack(state, 2, 5)
            return np.reshape(state, [1] + list(state.shape))
        if history_length == 5 and not other_agent:
            state = self.make_stack(state, 3, None)
            return np.reshape(state, [1] + list(state.shape))
        if history_length == 5 and other_agent:
            state = self.make_stack(state, 3, 6)
            return np.reshape(state, [1] + list(state.shape))
        if history_length == 10 and not other_agent:
            state = self.make_stack(state, 4, None)
            return np.reshape(state, [1] + list(state.shape))
        if history_length == 10 and other_agent:
            state = self.make_stack(state, 4, 7)
            return np.reshape(state, [1] + list(state.shape))

    '''
    creates the state stack
    The stack can be a configuration of 
    -active agent location
    -other agents location
    -active agent history (3,5 or all previous steps) (optional)
    -other agent histroy (optional)
    -fire
    -dirt
    '''

    def make_stack(self, state, h1, h2):
        if h1 is None and h2 is None:
            return np.dstack((
                state[0],
                state[1],
                state[8],
                state[9],
                state[10],
            ))
        if h1 is not None and h2 is None:
            return np.dstack((
                state[0],
                state[1],
                state[h1],
                state[8],
                state[9],
                state[10],
            ))
        if h1 is not None and h2 is not None:
            return np.dstack((
                state[0],
                state[1],
                state[h1],
                state[h2],
                state[8],
                state[9],
                state[10],
            ))

    '''
    Returns the depth of the input i.e. the amount of layers
    '''
    def get_depth(self):
        # return self.sim.W.get_state().shape[2]
        if HIST_LEN == 0:
            return 5
        if EXTENDED:
            return 7
        if not EXTENDED:
            return 6
