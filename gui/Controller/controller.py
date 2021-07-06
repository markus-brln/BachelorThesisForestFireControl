import time
from Model.model import Model
from Model import utils
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import statistics
from numba import jit
import tensorflow.keras.backend as K


timeframe = 20
class Controller:
  def __init__(self, model: Model, NN_control = False, variant="xy", NN_number=None, n_runs_per_NN=None):
    self.model = model

    # Initialization
    self.mouse_button_pressed = False                       # Mouse button assumed not to be pressed initially
    self.collecting_waypoints = False
    self.agent_no = 0
    self.last_timestep_waypoint_collection = -1

    # NN INTEGRATION
    if NN_control:
      self.NN_variant = variant                               # xy, angle, box
      self.NN_number = NN_number
      self.NN_control = NN_control
      self.NN = self.load_NN("CNN"+self.NN_variant + utils.experiment+str(NN_number))  # from json and h5 file
      self.digging_threshold = utils.digging_threshold
      self.n_failed = 0
      self.n_success = 0
      self.n_burned_cells = []
      self.n_assignments = 0
      self.fail_bit = 0
      self.next_model = False
      self.n_runs_per_NN = n_runs_per_NN


  """def update(self, event):
    #High-level of distribution of input events. Quit, mouse, keyboard events.
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
      self.key_press(event)"""


  def shut_down(self, event):
    self.model.shut_down()                                  # Ensure proper shutdown of the model
    exit(0)                                                 # Exit program


  def select(self, event):
    position = self.view.pixel_belongs_to_block(event.pos)  # Determine the block the mouse is covering
    self.model.select_square(position)                      # Select or deselect that block


  def prepare_collecting_waypoints(self):
    """Clear old waypoints from view, order by angle, highlight first agent."""
    # print("Start collecting waypoints")
    #self.view.clear_waypoints([self.model.find_node(pos) for pos in self.model.waypoints])
    self.model.waypoints.clear()                            # clear the actual waypoint positions after
                                                            # deleting them on the view!
    self.collecting_waypoints = True
    self.model.sort_agents_by_angle()
    self.agent_no = 0
    self.model.highlight_agent(0)


  """def collect_waypoints(self, event):
    @:param event, left or right mouse button
    Selects the square where button was pressed.
    Left Mouse Button  -> dig
    Right Mouse Button -> drive waypoint

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

    self.model.highlight_agent(self.agent_no)"""


  """def key_press(self, event):
    Governs data collection.
    :param event: keyboard button press
    SPACE     -> start next waypoint collection
    RETURN    -> save data collection episode
    BACKSPACE -> discard data collection episode (e.g. when agent drove into fire)
    
    if event.key == pygame.K_ESCAPE:
      self.model.DataSaver.save_training_run()
    if event.key == pygame.K_p:
      print("hi")
      env = self.produce_input_NN()
      print(env.shape)
      self.plot_binmap(env[0])
    if event.key == pygame.K_SPACE:
      if self.model.time % utils.timeframe == 0:
        if self.last_timestep_waypoint_collection != self.model.time:
          self.prepare_collecting_waypoints()
          self.last_timestep_waypoint_collection = self.model.time
        else:
          self.model.append_datapoint()                     # only start after first 'timeframe' timesteps
          #start = time.time()                              # measure model progression
          for _ in range(utils.timeframe):
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
      self.last_timestep_waypoint_collection = -1"""


  # ALL NN STUFF STARTS HERE
  def print_results_NN(self):
    print("\n\nEND OF TESTING")
    print("successfully contained fires: ", self.n_success)
    print("failed attempts: ", self.n_failed)
    print("total: ", self.n_failed + self.n_success)
    print("amounts of burned cells:", self.n_burned_cells)
    if len(self.n_burned_cells) > 2:
      print("average: ", sum(self.n_burned_cells) / len(self.n_burned_cells))
      print("SD, SE: ", statistics.stdev(self.n_burned_cells),
            statistics.stdev(self.n_burned_cells) / math.sqrt(len(self.n_burned_cells)))

  def fail(self):
    self.n_failed += 1
    print("model", self.NN_number, "fail", self.n_success, "/", self.n_success+self.n_failed, " successful")
    self.n_assignments = 0
    self.fail_bit = 1
    self.model.discard_episode()
    self.model.start_episode()
    self.model.reset_wind()
    self.last_timestep_waypoint_collection = -1


  def success(self, burned_cells):
    self.n_success += 1
    print("model", self.NN_number, "success", self.n_success, "/", self.n_success+self.n_failed, " successful")
    self.n_burned_cells.append(burned_cells)
    self.fail_bit = 0
    self.n_assignments = 0
    self.model.discard_episode()
    self.model.start_episode()
    self.model.reset_wind()
    self.last_timestep_waypoint_collection = -1

  def append_results_to_file(self):
    print("writing to file")
    file = open("results" + os.sep + self.NN_variant + os.sep + self.NN_variant + utils.experiment + ".txt", mode='a')
    file.write(str(self.n_runs_per_NN) + "\n")
    file.write(str(self.n_burned_cells) + "\n")
    print(self.n_burned_cells)
    #file.write(self.NN_variant + utils.experiment+ " model nr:" + str(self.NN_number)+"\n")
    #file.write(f"successfully contained fires: {self.n_success}\n")
    #file.write(f"failed attempts: {self.n_failed}\n")
    #file.write(f"total #runs: {self.n_failed + self.n_success}\n")
    #file.write(f"amounts of burned cells: {self.n_burned_cells}\n")
    #if len(self.n_burned_cells) > 2:
      #file.write(f"average: {sum(self.n_burned_cells) / len(self.n_burned_cells)}\n")
      #file.write(f"SD, SE: {statistics.stdev(self.n_burned_cells)} {statistics.stdev(self.n_burned_cells) / math.sqrt(len(self.n_burned_cells))}")

    #file.write("\n\n")


  def update_NN_no_gui(self):#, event):
    """Using SPACE, let NN assign waypoints to agents and progress the simulation."""
    #if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:

    if self.n_success + self.n_failed == self.n_runs_per_NN:
      self.append_results_to_file()
      self.next_model = True
      return

    # check whether amount of waypoint assignments is over 15 (failed)
    if self.n_assignments == 15:
      print("nr assignments fail")
      self.fail()
    self.n_assignments += 1

    self.fail_bit = 0                                     # reset for this iteration
    self.prepare_collecting_waypoints()
    outputs = self.predict_NN()                           # waypoints from CNN
    if self.set_waypoints_NN(outputs) == -1:
      print("waypoint fail")
      self.fail()
    if not self.fail_bit:
      for _ in range(utils.timeframe):
        self.model.time_step()

    # 1. check if agents are burned
    if len(self.model.agents) != utils.nr_of_agents:
      print("nr agents fail")
      self.fail()

    # 2. check whether there is a way out for the fire
    if not self.fail_bit:
      if self.model.firebreaks:                                 # do not compute anything when no firebreaks exist (yet)
        burned_cells = count_containment(self.model.firebreaks, self.model.size)
        if burned_cells != -1:                                  # -1 -> a way out was found for the fire
          self.success(burned_cells)
          self.fail_bit = 1                                     # still set this to 1 such that no new wp are generated
      #else:
        #print("way out found, continuing...")

    # 0. print the current results for every run, just to be sure
    #self.print_results_NN()

    if self.model.reset_necessary:                          # update view when resetting env (hacky way)
      #self.view.update()
      self.model.reset_necessary = False


  def set_waypoints_NN(self, outputs):
    """Emulates the manual setting of waypoints through mouse clicks
       by using the NN output."""
    if self.NN_variant == "box":
      #print("outputs[0]", outputs[0])
      bad_wp = 0
      for agent in range(0, 5):
        positions, dig_drive = outputs
        # print("len", len(positions), "agent no.", self.agent_no)
        new_wp, digging = self.postprocess_output_NN_box(positions[agent], self.model.agents[agent], dig_drive[agent])
        #print("pos: ", new_wp, "dig: ", digging)
        #print(" ")
        #self.model.highlight_agent(agent)
        print(new_wp)
        if not 0 < new_wp[0] < utils.size or not 0 < new_wp[1] < utils.size:
          print("Waypoint was outside the environment! Press backspace to discard episode!")
          bad_wp = 1  # waypoint outside of environment, FAIL!

        self.model.select_square(new_wp, digging=digging)
        self.agent_no += 1

      if bad_wp:
        print("HELLOOO")
        return -1
    else:
      for output in outputs:
        new_wp, digging = None, None
        if self.NN_variant == "xy":
          new_wp, digging = self.postprocess_output_NN_xy(output, self.model.agents[self.agent_no])
        elif self.NN_variant == "angle":
          new_wp, digging = self.postprocess_output_NN_angle(output, self.model.agents[self.agent_no])
        else:
          print("implement postprocess_output_NN_...() for your variant")
          exit()

        if not 0 < new_wp[0] < utils.size or not 0 < new_wp[1] < utils.size:
          print("Waypoint was outside the environment! Press backspace to discard episode!")
          return -1                                         # waypoint outside of environment, FAIL!

        # print("pos: ", new_wp, "dig: ", digging)
        self.model.highlight_agent(self.agent_no)
        self.model.select_square(new_wp, digging=digging)
        self.agent_no += 1

    self.collecting_waypoints = False
    self.model.highlight_agent(None)

  def arrayIndex2WaypointPos(self, idx):
    size = 5
    x = 0
    cnt = 0
    wp = (0, 0)
    for y in range(-size, size + 1):
      for x in range(-x, x + 1):
        cnt += 1
        if cnt == idx:
          wp = (x, y)
          #print(cnt, "wp", wp)
          if (abs(x) + abs(y)) != 0:
            scale = utils.timeframe / (abs(x) + abs(y))
          else:
            scale = 1
          x = round(scale * x)
          y = round(scale * y)
          wp = (x, y)
      x += 1 if y < 0 else -1

    return wp

  def waypointValid(self, wp, agent):
    # if (wp[0] >= -20 and wp[0] <= 20):
    # if(wp[1] >= -20 and wp[1] <= 20):
    if 0 < agent.position[0] + wp[0] < 255:
      if 0 < agent.position[1] + wp[1] < 255:
        return 1
    return 0

  def postprocess_output_NN_box(self, output, agent, dig_drive):
    """All operations needed to transform the raw normalized NN output
    to pixel coords of the waypoints and a drive/dig (0/1) decision.
    """

    #print("pay attention", max(output), "digging:", output[1])
    digging = dig_drive > self.digging_threshold
    # digging = 0
    waypointIdx = 0
    for idx in range(len(output)):
      if output[idx] == max(output):
        waypointIdx = idx

    #print("agent pos", agent.position[0], agent.position[1])
    wp = self.arrayIndex2WaypointPos(waypointIdx)

    if not digging:  # double range for driving
      wp = (wp[0] * 2, wp[1] * 2)

    # print("indx:", waypointIdx, "wp", wp)
    if self.waypointValid(wp, agent):
      delta_x = wp[0]
      delta_y = wp[1]

      #print("x", delta_x, "y", delta_y)
      # if (abs(delta_x) + abs(delta_y)) > 15:
      #   digging = 1
      #print("agent moves to", agent.position[0] + delta_x, agent.position[1] + delta_y)
      output = agent.position[0] + int(delta_x), agent.position[1] + int(delta_y)
    else:
      #print("invalid waypoint position agent stops")
      output = agent.position[0], agent.position[1]

    return output, digging


  def postprocess_output_NN_xy(self, output, agent):
    """All operations needed to transform the raw normalized NN output
    to pixel coords of the waypoints and a drive/dig (0/1) decision.
    """
    digging = output[2] > self.digging_threshold

    if digging:
      delta_x = output[0] * utils.timeframe
      delta_y = output[1] * utils.timeframe
    else:
      delta_x = output[0] * utils.timeframe * 2                   # twice as fast driving
      delta_y = output[1] * utils.timeframe * 2

    wanted_len = utils.timeframe                                  # agents can dig 1 step per timestep
    if not digging:
      wanted_len *= 2                                       # driving twice as fast

    scale = wanted_len / (abs(delta_x) + abs(delta_y))
    output = agent.position[0] + int(scale * delta_x), agent.position[1] + int(scale * delta_y)

    return output, digging


  def postprocess_output_NN_angle(self, output, agent):
    """All operations needed to transform the raw normalized NN output
    to pixel coords of the waypoints and a drive/dig (0/1) decision.
    """

    # print(f"output: {output}")
    cos_x = output[0]
    sin_x = output[1]
    radius = output[2]
    digging = output[3] > self.digging_threshold

    delta_x = cos_x * radius * utils.timeframe
    delta_y = sin_x * radius * utils.timeframe

    wanted_len = utils.timeframe                            # agents can dig 1 cell per timestep
    if not digging:
      wanted_len *= 2                                       # driving twice as fast

    scale = wanted_len / (abs(delta_x) + abs(delta_y))
    output = agent.position[0] + int(scale * delta_x), agent.position[1] + int(scale * delta_y)

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

    wanted_len = utils.timeframe                                  # agents can dig 1 step per timestep
    if not digging:
      wanted_len *= 2                                       # driving twice as fast

    delta_x = new_wp[0] - agent.position[0]
    delta_y = new_wp[1] - agent.position[1]

    scale = wanted_len / (abs(delta_x) + abs(delta_y))
    output = agent.position[0] + int(scale * delta_x), agent.position[1] + int(scale * delta_y)

    return output, digging

  @staticmethod
  def plot_binmap(image):
    channels = np.dsplit(image.astype(dtype=np.float32), len(image[0][0]))
    f, axarr = plt.subplots(2, 4)
    axarr[0, 0].imshow(np.reshape(channels[0], newshape=(256, 256)).T, cmap='Greys', vmin=0, vmax=1)
    axarr[0, 0].set_title("active fire")
    axarr[0, 1].imshow(np.reshape(channels[1], newshape=(256, 256)).T, cmap='Greys',vmin=0, vmax=1)
    axarr[0, 1].set_title("fire breaks")
    axarr[0, 2].imshow(np.reshape(channels[2], newshape=(256, 256)).T, cmap='Greys',vmin=0, vmax=1)
    axarr[0, 2].set_title("wind dir (uniform)")
    axarr[1, 0].imshow(np.reshape(channels[3], newshape=(256, 256)).T, cmap='Greys',vmin=0, vmax=1)
    axarr[1, 0].set_title("wind speed (uniform)")
    axarr[1, 1].imshow(np.reshape(channels[4], newshape=(256, 256)).T, cmap='Greys',vmin=0, vmax=1)
    axarr[1, 1].set_title("other agents")
    axarr[1, 2].imshow(np.reshape(channels[5], newshape=(256, 256)).T, cmap='Greys',vmin=0, vmax=1)
    axarr[1, 2].set_title("active agent")
    axarr[1, 3].imshow(np.reshape(channels[6], newshape=(256, 256)).T, cmap='Greys',vmin=0, vmax=1)
    axarr[1, 3].set_title("active agent y")
    print("max", np.max(channels[5]), np.max(channels[6]))
    plt.show()

  @staticmethod
  def plot_np_image(image):
    channels = np.dsplit(image.astype(dtype=np.float32), len(image[0][0]))
    f, axarr = plt.subplots(2, 4)
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
    axarr[1, 2].imshow(np.reshape(channels[5], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[1, 2].set_title("active agent")
    axarr[1, 3].imshow(np.reshape(channels[6], newshape=(256, 256)), vmin=0, vmax=1)
    axarr[1, 3].set_title("active agent y")
    print("max", np.max(channels[5]), np.max(channels[6]))
    plt.show()


  def produce_input_NN_old(self):
    """
    construct inputs images like in the data_translator, plus active agent positions to concatenate
    - [0] active fire (no burned cell or tree channels)
    - [1] fire breaks
    - [2] wind direction
    - [3] wind speed
    - [4] other agents
    """
    shape = (256, 256, 7)  # see doc comment
    single_image = np.zeros(shape)

    for fire_pixel in self.model.firepos:
      single_image[fire_pixel[0]][fire_pixel[1]][0] = 1
    for firebreak_pixel in self.model.firebreaks:
      single_image[firebreak_pixel[0]][firebreak_pixel[1]][1] = 1

    single_image[:, :, 2] = self.model.get_wind_dir_idx() / (utils.n_wind_dirs - 1)
    single_image[:, :, 3] = self.model.wind_speed / (utils.n_wind_speed_levels - 1)

    all_images = []
    apd = 10  # agent_point_diameter
    for active_agent in self.model.agents:
      agent_image = np.copy(single_image)
      agent_image[:, :, 5] = active_agent.position[0] / 255  # x, y position of active agent on channel 5,6
      agent_image[:, :, 6] = active_agent.position[1] / 255

      for other_agent in self.model.agents:
        if other_agent != active_agent:
          x, y = other_agent.position
          agent_image[y - apd: y + apd, x - apd: x + apd, 4] = 1

      print("current agent: ", active_agent.position)

      # self.plot_np_image(agent_image)
      all_images.append(agent_image)  # 1 picture per agent

    return np.asarray(all_images)
    # return [np.asarray(all_images), np.asarray(agent_positions)]   concat version


  def produce_input_NN(self):
    """
    construct inputs images like in the data_translator, plus active agent positions to concatenate

    - [0] active fire (no burned cell or tree channels)
    - [1] fire breaks
    - [2] wind direction
    - [3] wind speed
    - [4] other agents
    """
    shape = (256, 256, 7)                                     # see doc comment
    single_image = np.zeros(shape)

    for fire_pixel in self.model.firepos:
      single_image[fire_pixel[0]][fire_pixel[1]][0] = 1
    for firebreak_pixel in self.model.firebreaks:
      single_image[firebreak_pixel[0]][firebreak_pixel[1]][1] = 1

    single_image[:, :, 2] = self.model.get_wind_dir_idx() / (utils.n_wind_dirs - 1)
    single_image[:, :, 3] = self.model.wind_speed / (utils.n_wind_speed_levels - 1)

    all_images = []
    apd = 10                                                # agent_point_diameter
    for active_agent in self.model.agents:
      agent_image = np.copy(single_image)
      agent_image[:, :, 5] = active_agent.position[0] / utils.size  # x, y position of active agent on channel 5,6
      agent_image[:, :, 6] = active_agent.position[1] / utils.size

      for other_agent in self.model.agents:
        if other_agent != active_agent:
          x, y = other_agent.position
          agent_image[y - apd : y + apd, x - apd: x + apd, 4] = 1

      # print("current agent: ", active_agent.position)

      #self.plot_np_image(agent_image)
      all_images.append(agent_image)                        # 1 picture per agent

    return np.asarray(all_images)
    #return [np.asarray(all_images), np.asarray(agent_positions)]   concat version


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
      concat_vector.append(wind_info + [pos[0] / utils.size, pos[1] / utils.size])

    print(concat_vector)
    concat_vector = np.asarray(concat_vector)

    return env_images, concat_vector

  def predict_NN(self):
    """Use the pre-loaded CNN to generate waypoints for the agents.
    """
    if len(self.model.agents) != utils.nr_of_agents:
      print("Agent(s) must have died")
      exit()

    NN_input = self.produce_input_NN()
    #print(NN_input)

    #print("predicting")
    output = self.NN.predict(NN_input)                      # needs to be a list of [images, concat], see
    #print(output)
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
  def box_loss(y_true, y_pred):
    # scale predictions so that the class probabilities of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clipping to remove divide by zero errors
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # results of loss func
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, -1)
    return loss

  @staticmethod
  def build_model_box(input_shape):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

    downscaleInput = Input(shape=input_shape)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(
      downscaleInput)
    downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
    downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
    downscaled = Flatten()(downscaled)
    downscaled = Dropout(0.03)(downscaled)
    out = Dense(8, activation='sigmoid')(downscaled)
    dig_drive = Dense(1, activation='sigmoid', name='dig')(out)
    box = Dense(64, activation='relu')(downscaled)
    box = Dense(61, activation='softmax', name='box')(box)

    model = Model(inputs=downscaleInput, outputs=[box, dig_drive])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # 0.0005
    # loss1 = weighted_loss(weights=weights)
    model.compile(loss=[Controller.box_loss, tf.keras.losses.BinaryCrossentropy()],
                  ## categorical_crossentropy  ## tf.keras.losses.BinaryCrossentropy()
                  optimizer=adam,
                  # metrics=['categorical_accuracy', 'binary_crossentropy']
                  )
    return model


  @staticmethod
  def load_NN(filename):
    """Load a Keras model from json file and weights (.h5). Same as in
    CNN/NNutils.py"""
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("loading model " + filename)
    import tensorflow

    if "box" in filename:
      model = Controller.build_model_box((256, 256, 7))
      model.load_weights('..' + os.sep + 'CNN' + os.sep + 'saved_models' + os.sep + filename + ".h5")
    else:
      # load json and create model
      json_file = open('..' + os.sep +'CNN' + os.sep + 'saved_models' + os.sep + filename + '.json', 'r')
      model_json = json_file.read()
      json_file.close()

      model = tensorflow.keras.models.model_from_json(model_json)
      # load weights into new model
      model.load_weights('..' + os.sep +'CNN' + os.sep + 'saved_models' + os.sep + filename + ".h5")
      print("Loaded model from disk")

    return model


@jit(nopython=True)
def count_containment(firebreaks, size):
  """Important for testing, counts the amount of potentially
  burned cells when fire was contained."""
  burned = [(int(size/2), int(size/2))]         # starting from the middle like the fire
  new_burned = burned.copy()
  previous_n = 0                                          # previous amount of potentially burned cells
  last_check = 0

  while len(burned) > previous_n:
    if len(burned) - last_check > 200:                    # check regularly if the fire has an escape to
      #start = time.time_ns()
      if check_escape(firebreaks, new_burned):  # the bounds of the environment
        #print("escape:", time.time_ns() - start)
        return -1
      #print("escape:", time.time_ns() - start)
      last_check = len(burned)

    previous_n = len(burned)
    new_new_burned = []
    for pos in new_burned:
      neighbours = [(pos[0] + 1, pos[1]),(pos[0], pos[1] + 1),(pos[0] - 1, pos[1]),(pos[0], pos[1] - 1)]
      for neighbour in neighbours:
        if neighbour not in firebreaks and neighbour not in burned and neighbour not in new_new_burned:
          new_new_burned.append(neighbour)
    new_burned = new_new_burned
    burned.extend(new_burned)

  return len(burned) + len(firebreaks)


@jit(nopython=True)
def check_escape(firebreaks, new_burned):
  #print("check escape")
  for burning_cell in new_burned:
    blocked_up = 0
    blocked_down = 0
    blocked_right = 0
    blocked_left = 0
    for firebreak in firebreaks:
      if firebreak[1] == burning_cell[1]:                 # if y value is the same
        if 0 < firebreak[0] < burning_cell[0]:            # firebreak between cell and left bound
          blocked_left = 1
        else:                                             # between cell and right bound
          blocked_right = 1
      if firebreak[0] == burning_cell[0]:
        if 0 < firebreak[1] < burning_cell[1]:
          blocked_up = 1
        else:
          blocked_down = 1
                                                          # stop trying if a cell is "surrounded"
      if blocked_up and blocked_down and blocked_left and blocked_right:
        break

    if not (blocked_up and blocked_down and blocked_left and blocked_right):
      return 1                                            # escape found!

  return 0