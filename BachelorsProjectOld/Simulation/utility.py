from colour import Color
import numpy as np
import random

'''
The mid-point circle drawing algorithm returns the points on a discrete 2D map
that should be filled to form a circle of radius r around the point (midx, midy)
'''


def circle_points(midx, midy, r):
    coords = list()
    x = r
    y = 0

    coords.append((x + midx, y + midy))

    # When radius is zero there is only a single point
    if (r > 0):
        coords.append((-x + midx, -y + midy))
        coords.append((y + midx, -x + midy))
        coords.append((-y + midx, x + midy))

    # Initialising the value of P
    P = 1 - r
    while (x > y):
        y += 1

        # Mid-points inside or on the perimeter
        if (P <= 0):
            P = P + 2 * y + 1
        # Mid-points outside the perimeter
        else:
            x -= 1
            P = P + 2 * y - 2 * x + 1

        # All the perimeter points have already been printed
        if (x < y):
            break

        # Get the point and its reflection in the other octants
        coords.append((x + midx, y + midy))
        coords.append((-x + midx, y + midy))
        coords.append((x + midx, -y + midy))
        coords.append((-x + midx, -y + midy))

        # If the generated points on the line x = y then
        # the perimeter points have already been printed
        if (x != y):
            coords.append((y + midx, x + midy))
            coords.append((-y + midx, x + midy))
            coords.append((y + midx, -x + midy))
            coords.append((-y + midx, -x + midy))

    return coords


'''
Find the middle point of the map to start the fire

This function has to stay deterministic because we call it multiple times
throughout the code and we rely on the fact that it always returns the same
position every time.
'''


def get_fire_location(width, height):
    midx = int(width / 2)
    midy = int(height / 2)
    return (midx, midy)


def get_agent_location(width, height, all_locations):
    # Minimum map size is 10x10
    assert width >= 10 and height >= 10

    # Agent distance from fire varies but is never too far away
    radius = random.randint(1, int((width / 2)) - 1)
    # Don't draw circles larger than the map
    midx, midy = get_fire_location(width, height)
    locations = circle_points(midx, midy, radius)
    # Choose a random point on the circle
    random_idx = np.random.choice(np.array(range(len(locations))))
    random_loc = locations[random_idx]
    while (int(random_loc[0]), int(random_loc[1])) in all_locations:
        random_idx = np.random.choice(np.array(range(len(locations))))
        random_loc = locations[random_idx]
    all_locations.append((int(random_loc[0]), int(random_loc[1])))
    x, y = get_faraway_loc(int(random_loc[0]), int(random_loc[1]), height, False)
    # Convert to int because otherwise not JSON serializable
    return x, y

def get_faraway_loc(x, y, width, b):
    middle = width / 2
    left = round(middle / 2)
    right = middle + left
    lits = []
    for i in range(0, left):
        lits.append(i)
    for i in range(int(right), width):
        lits.append(i)

    if b and (x > left or x < right) and (y > left or y < right):
        x = random.choice(lits)
        lits.remove(x)
        y = random.choice(lits)
        return x, y
    else:
        return x, y


# Generate a name for a certain environment, based on size and complexity of the environment
def get_name(size, environment):
    import time
    return (
        f"""{size}s-{environment}-"""
    )


# Convert a color to grayscale with the grayscale formula from Wikipedia
def grayscale(color):
    r, g, b = color.red, color.green, color.blue
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# The parameters for grass
grass = {
    "gray": grayscale(Color("Green")),
    "gray_burning": grayscale(Color("Red")),
    "gray_burnt": grayscale(Color("Black")),
    "heat": 0.3,
    "fuel": 20,
    "threshold": 3,
    "radius": 1,
}

# The parameters for dirt
dirt = {
    "gray": grayscale(Color("Brown")),
}

# The parameters for water
water = {
    "gray": grayscale(Color("Blue")),
}

# The parameters for houses
house = {
    "gray": grayscale(Color("Orange")),
    "gray_house_burning": grayscale(Color("Firebrick")),
    "gray_house_burnt": grayscale(Color("Whitesmoke")),
    "heat": 0.3,
    "fuel": 50,
    "threshold": 3,
    "radius": 1,
}

# The (depth) layers of the map, which correspondsagent_other_hist to cell attributes
layer = {
    "type": 0,
    "gray": 1,
    "temp": 2,
    "heat": 3,
    "fuel": 4,
    "threshold": 5,
    "agent_pos": 6,
    "other_pos": 7,
    "agent_hist_3": 8,
    "agent_hist_5": 9,
    "agent_hist_all": 10,
    "agent_other_hist_3": 11,
    "agent_other_hist_5": 12,
    "agent_other_hist_all": 13,
    "fire_mobility": 14,
    "agent_mobility": 15,
}

# Which cell type (from the type layer) corresponds to which value
types = {
    0: "grass",
    1: "fire",
    2: "burnt",
    3: "dirt",
    4: "water",
    5: "house ",

    "grass": 0,
    "fire": 1,
    "burnt": 2,
    "dirt": 3,
    "water": 4,
    "house": 5,
}

# Convert grayscale to ascii for rendering
color2ascii = {
    grayscale(Color("Green")): '+',
    grayscale(Color("Red")): '@',
    grayscale(Color("Black")): '#',
    grayscale(Color("Brown")): '0',
    grayscale(Color("Blue")): 'x',
    grayscale(Color("Orange")): '^',
    grayscale(Color("Firebrick")): '*',
    grayscale(Color("Whitesmoke")): '!',

}
