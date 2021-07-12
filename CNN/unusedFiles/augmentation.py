
# ROTATION TRYS
def rot_pos(pos):
    size = 255
    x, y = pos
    x -= size / 2
    y -= size / 2

    return (-y + size / 2, x + size / 2)


def rotate_wind(wind):
    list = wind.tolist()
    idx = list.index(1)
    list[idx] = 0
    if idx == 0:  # nn
        idx = 2  # ee
    elif idx == 1:  # ss
        idx = 3  # ww
    if idx == 2:  # ee
        idx = 1  # ss
    elif idx == 3:  # ww
        idx = 0  # nn
    if idx == 4:  # ne
        idx = 6  # se
    elif idx == 5:  # nw
        idx = 4  # ne
    if idx == 6:  # se
        idx = 7  # sw
    elif idx == 7:  # sw
        idx = 5  # nw
    list[idx] = 1
    wind = np.array(list)
    return wind


def rotate(datapoint):
    environment = np.rot90(datapoint[0])
    wind = rotate_wind(datapoint[1])
    # Wind speed
    windspeed = datapoint[2]

    new_waypoints = []
    for idx in range(len(datapoint[3])):
        first_entry = rot_pos(datapoint[3][idx][0])
        second_entry = rot_pos(datapoint[3][idx][1])
        digging = datapoint[3][idx][2]
        new_waypoints.append([first_entry, second_entry, digging])

    return [environment, wind, windspeed, new_waypoints]


def augment_datapoint(datapoint):
    augmented_data = [datapoint]
    augmented_data.append(rotate(augmented_data[-1]))
    augmented_data.append(rotate(augmented_data[-1]))
    augmented_data.append(rotate(augmented_data[-1]))

    return augmented_data


def shift_img(img, x_shift, y_shift):
  #  NORMAL = 0
  #  FIREBREAK = 1
  #  ON_FIRE = 2
  #  BURNED_OUT = 3
  #  AGENT = 4
  #  waypoint_digging = 5 (based on agent.waypoint)
  #  waypoint_walking = 6
  img_copy = img.copy()
  for y, row in enumerate(img):
    for x, cell in enumerate(row):
      if cell == 0 or cell == 2 or cell == 3:
        img_copy[y][x] = cell
      else:
        img_copy[y + y_shift][x + x_shift] = cell
        img_copy[y][x] = -3

  return img_copy

def shift_out(out, x_shift, y_shift):
  final = []
  for elem in out:
    final.append([elem[0][0] + x_shift, elem[0][1] + y_shift,
                  elem[1][0] + x_shift, elem[1][1] + y_shift,
                  elem[2]])
  return final


def shift_augment(data):
  """Shifts agent positions, waypoints and firebreaks by 1 in every direction."""
  final = []

  for i, dat in enumerate(data):
    print(f"shift augment {i} /{len(data)}")
    img, wind_dir, wind_speed, out = dat
    final.append([shift_img(img.copy(), 1, 1), wind_dir, wind_speed, shift_out(out, 1, 1)])
    final.append([shift_img(img.copy(), 1, -1), wind_dir, wind_speed, shift_out(out, 1, -1)])
    final.append([shift_img(img.copy(), -1, 1), wind_dir, wind_speed, shift_out(out, -1, 1)])
    final.append([shift_img(img.copy(), -1, -1), wind_dir, wind_speed, shift_out(out, -1, -1)])
    final.append(dat)

  return final
