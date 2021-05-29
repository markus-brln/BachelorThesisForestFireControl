import pygame
import time
from Model.model import Model
from View.view import View
from Model.utils import *
import os
import numpy as np
from matplotlib import pyplot as plt


class Controller:
  def __init__(self, model: Model, view: View, NN_control = False, variant="xy"):
    self.model = model
    self.view = view

    # Initialization
    self.mouse_button_pressed = False                       # Mouse button assumed not to be pressed initially
    self.collecting_waypoints = False
    self.agent_no = 0
    self.last_timestep_waypoint_collection = -1

    # NN INTEGRATION
    self.NN_variant = variant                               # xy, angle, box
    self.NN_control = NN_control
    self.NN = None
    if self.NN_control:
      self.NN = self.load_NN("CNN"+self.NN_variant)                              # from json and h5 file
    self.digging_threshold = digging_threshold


  def update(self, event):
    """High-level of distribution of input events. Quit, mouse, keyboard events."""
    if event.type == pygame.QUIT:                           # Exit the program
      self.shut_down(event)

    if self.collecting_waypoints:
      self.collect_waypoints(event)
      return

    if self.model.reset_necessary:                          # update view when resetting env (hacky way)
      self.view.update()
      self.model.reset_necessary = False
    elif event.type == pygame.MOUSEBUTTONDOWN:              # Mouse stationary and mouse button pressed
      self.mouse_button_pressed = event.button
    elif event.type == pygame.MOUSEBUTTONUP:                # Mouse button released
      self.mouse_button_pressed = False
      self.last_clicked = (-1, 0)                           # Reset to allow clicking a square twice in a row
    elif event.type == pygame.KEYDOWN:
      # Keyboard button pressed
      self.key_press(event)


  def shut_down(self, event):
    self.model.shut_down()                                  # Ensure proper shutdown of the model
    exit(0)                                                 # Exit program


  def select(self, event):
    position = self.view.pixel_belongs_to_block(event.pos)  # Determine the block the mouse is covering
    self.model.select_square(position)                      # Select or deselect that block


  def prepare_collecting_waypoints(self):
    """Clear old waypoints from view, order by angle, highlight first agent."""
    print("Start collecting waypoints")
    self.view.clear_waypoints([self.model.find_node(pos) for pos in self.model.waypoints])
    self.model.waypoints.clear()                            # clear the actual waypoint positions after
                                                            # deleting them on the view!
    self.collecting_waypoints = True
    self.model.sort_agents_by_angle()
    self.agent_no = 0
    self.model.highlight_agent(0)


  def collect_waypoints(self, event):
    """@:param event, left or right mouse button
    Selects the square where button was pressed.
    Left Mouse Button  -> dig
    Right Mouse Button -> drive waypoint"""

    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_LEFT:
        if self.agent_no != 0:
          self.model.undo_selection(self.agent_no - 1)
          self.agent_no -= 1

    if event.type != pygame.MOUSEBUTTONDOWN:
      return

    position = self.view.pixel_belongs_to_block(event.pos)

    if event.button == 1:                                   # left mouse button, digging waypoint
      self.model.select_square(position, digging=True)
    elif event.button == 3:                                 # right mouse button, walking waypoint
      self.model.select_square(position, digging=False)
    else:
      print("use left(digging) or right (walking) mouse button")
      return

    self.agent_no += 1
    if self.agent_no >= len(self.model.agents):
      self.collecting_waypoints = False
      self.model.highlight_agent(None)
      return

    self.model.highlight_agent(self.agent_no)


  def key_press(self, event):
    """Governs data collection.
    :param event: keyboard button press
    SPACE     -> start next waypoint collection
    RETURN    -> save data collection episode
    BACKSPACE -> discard data collection episode (e.g. when agent drove into fire)
    """
    if event.key == pygame.K_ESCAPE:
      self.model.DataSaver.save_training_run()

    if event.key == pygame.K_SPACE:
      if self.model.time % timeframe == 0:
        if self.last_timestep_waypoint_collection != self.model.time:
          self.prepare_collecting_waypoints()
          self.last_timestep_waypoint_collection = self.model.time
        else:
          self.model.append_datapoint()                     # only start after first 'timeframe' timesteps
          #start = time.time()                              # measure model progression
          for _ in range(timeframe):
            self.model.time_step()                          # Space to go to next timestep

          #print("time: ", time.time()-start)

    if event.key == pygame.K_RETURN:
      self.model.append_episode()
      self.model.start_episode()                            # Return / ENTER to go to next episode
      self.model.reset_wind()
      self.last_timestep_waypoint_collection = -1           # first get new waypoints when restarting episode

    if event.key == pygame.K_BACKSPACE:
      self.model.discard_episode()
      self.model.start_episode()                            # Return / ENTER to go to next episode
      self.model.reset_wind()
      self.last_timestep_waypoint_collection = -1


  # ALL NN STUFF STARTS HERE
  def update_NN(self, event):
    """Using SPACE, let NN assign waypoints to agents and progress the simulation."""
    if event.type == pygame.QUIT:
      # TODO save data about how often fire was contained
      exit()

    if (not self.collecting_waypoints and event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE) or len(self.model.agents) != 5:
      self.model.discard_episode()
      self.model.start_episode()                            # BACKSPACE to go to next episode
      self.model.reset_wind()
      self.last_timestep_waypoint_collection = -1

    if self.collecting_waypoints and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
      outputs = self.predict_NN()                           # waypoints from CNN
      #print("outputs: ", outputs)
      self.set_waypoints_NN(outputs)

      for _ in range(timeframe):
        self.model.time_step()
      return

    if self.model.reset_necessary:                          # update view when resetting env (hacky way)
      self.view.update()
      self.model.reset_necessary = False

    elif not self.collecting_waypoints and event.type == pygame.KEYDOWN:
      self.prepare_collecting_waypoints()


  def set_waypoints_NN(self, outputs):
    """Emulates the manual setting of waypoints through mouse clicks
       by using the NN output."""

    for output in outputs:
      new_wp, digging = self.postprocess_output_NN_xy(output, self.model.agents[self.agent_no])
      print("pos: ", new_wp, "dig: ", digging)
      self.model.highlight_agent(self.agent_no)
      self.model.select_square(new_wp, digging=digging)
      self.agent_no += 1
      time.sleep(0.5)                           # TODO get rid of this when collecting data

    self.collecting_waypoints = False
    self.model.highlight_agent(None)


  def postprocess_output_NN_xy(self, output, agent):
    """All operations needed to transform the raw normalized NN output
    to pixel coords of the waypoints and a drive/dig (0/1) decision.
    """
    digging = output[2] > self.digging_threshold

    if digging:
      delta_x = output[0] * timeframe
      delta_y = output[1] * timeframe
    else:
      delta_x = output[0] * timeframe * 2                   # twice as fast driving
      delta_y = output[1] * timeframe * 2

    output = int(agent.position[0] + delta_x), int(agent.position[1] + delta_y)

    return output, digging


  def postprocess_output_NN_xy_full_env(self, output, agent):
    """
    NOT USED in 3 different architectures
    All operations needed to transform the raw normalized NN output
    to pixel coords of the waypoints and a drive/dig (0/1) decision.
    Scales the waypoints to fit the maximum ('timeframe') distance
    they can travel between waypoint assignments."""
    new_wp = (int(output[0] * 255), int(output[1] * 255))
    digging = output[2] > self.digging_threshold

    wanted_len = timeframe                                  # agents can dig 1 step per timestep
    if not digging:
      wanted_len *= 2                                       # driving twice as fast

    delta_x = new_wp[0] - agent.position[0]
    delta_y = new_wp[1] - agent.position[1]

    scale = wanted_len / (abs(delta_x) + abs(delta_y))
    output = agent.position[0] + int(scale * delta_x), agent.position[1] + int(scale * delta_y)

    return output, digging

  @staticmethod
  def plot_np_image(image):
    channels = np.dsplit(image.astype(dtype=np.float32), len(image[0][0]))
    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(np.reshape(channels[0], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[0, 0].set_title("active fire")
    axarr[0, 1].imshow(np.reshape(channels[1], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[0, 1].set_title("fire breaks")
    axarr[0, 2].imshow(np.reshape(channels[2], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[0, 2].set_title("wind dir (uniform)")
    axarr[1, 0].imshow(np.reshape(channels[3], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[1, 0].set_title("wind speed (uniform)")
    axarr[1, 1].imshow(np.reshape(channels[4], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[1, 1].set_title("other agents")
    # axarr[1, 2].imshow(np.reshape(channels[5], newshape=(256, 256)), vmin=0, vmax=1)
    # axarr[1, 2].set_title("active agent")
    plt.show()

  def produce_input_NN(self):
    """
    construct inputs images like in the data_translator, plus active agent positions to concatenate

    - [0] active fire (no burned cell or tree channels)
    - [1] fire breaks
    - [2] wind direction
    - [3] wind speed
    - [4] other agents
    """
    concat = [agent.position for agent in self.model.agents]  # the only info concatenated at the
                                                              # moment is the positions of the active agents
    for pos in concat:
      pos = [pos[0] / size, pos[1] / size]

    shape = (256, 256, 5)                                     # see doc comment
    single_image = np.zeros(shape)

    for fire_pixel in self.model.firepos:
      single_image[fire_pixel[0]][fire_pixel[1]][0] = 1
    for firebreak_pixel in self.model.firebreaks:
      single_image[firebreak_pixel[0]][firebreak_pixel[1]][1] = 1


    single_image[:, :, 2] = self.model.get_wind_dir_idx() / (n_wind_dirs - 1)
    single_image[:, :, 3] = self.model.wind_speed / (n_wind_speed_levels - 1)

    all_images = []
    apd = 10                                                # agent_point_diameter
    for active_agent in self.model.agents:
      agent_image = np.copy(single_image)
      for other_agent in self.model.agents:
        if other_agent != active_agent:
          x, y = other_agent.position
          agent_image[x - apd: x + apd, y - apd : y + apd, 4] = 1

      #self.plot_np_image(agent_image)
      all_images.append(agent_image)                        # 1 picture per agent


    return [np.asarray(all_images), np.asarray(concat)]


  def produce_input_full_env_xy(self):
    """NOT USED
    not for the 3 newest architectures

    INPUT:
    - on-the-fly updated 5-channel image of environment
    - concat vector (wind dir + wind speed + agent pos)

    OUTPUT:
    - n_agents * [x, y, 0-1 value for drive/dig]"""
    env_images = 5 * [self.model.array_np]
    wind_info = list(self.model.wind_info_vector)
    agent_positions = [agent.position for agent in self.model.agents]
    print("agent positions: ", agent_positions)
    concat_vector = list()
    for pos in agent_positions:
      concat_vector.append(wind_info + [pos[0] / size, pos[1] / size])

    print(concat_vector)
    concat_vector = np.asarray(concat_vector)

    return env_images, concat_vector

  def predict_NN(self):
    """Use the pre-loaded CNN to generate waypoints for the agents.
    """
    if len(self.model.agents) != 5:
      print("Agent(s) must have died")
      exit()

    NN_input = self.produce_input_NN()
    print(NN_input)

    print("predicting")
    output = self.NN.predict(NN_input)                      # needs to be a list of [images, concat], see
    return output


  @staticmethod
  def plot_env_img(img):
    """translate the 5 channel input back to displayable images"""
    orig_img = np.zeros((256, 256))

    for y, row in enumerate(img):
      for x, cell in enumerate(row):
        for idx, item in enumerate(cell):
          if item == 1:
            orig_img[y][x] = idx

    from matplotlib import pyplot as plt
    plt.imshow(orig_img)
    plt.show()


  @staticmethod
  def load_NN(filename):
    """Load a Keras model from json file and weights (.h5). Same as in
    CNN/NNutils.py"""
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("loading model " + filename)
    import tensorflow
    # load json and create model
    json_file = open('saved_models' + os.sep + filename + '.json', 'r')
    model_json = json_file.read()
    json_file.close()

    model = tensorflow.keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights('saved_models' + os.sep + filename + ".h5")
    print("Loaded model from disk")

    return model
