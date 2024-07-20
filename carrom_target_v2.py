import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Load the target image
target_image = cv2.imread('assets/carromboard2.jpeg')
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Load multiple template images
template_paths = ['assets/coin_black1.jpeg', 'assets/coin_white1.jpeg']
striker_gray = cv2.imread('assets/coin_striker.jpeg', 0)

def distance(x1, y1, x2, y2):
    return sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

def is_far_enough(pt, existing_pts, min_dist=50):
    return all(distance(pt[0], pt[1], ept[0], ept[1]) >= min_dist for ept in existing_pts)

def generate_points_on_line(x1, y1, x2, y2):
    points = []
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        for i in range(-25, 26,1):
            k = b + i
            if abs(x2 - x1) >= abs(y2 - y1):
                points.extend((x, round(m * x + k)) for x in range(min(x1, x2), max(x1, x2) + 1))
            else:
                points.extend((round((y - k) / m), y) for y in range(min(y1, y2), max(y1, y2) + 1))
    else:
        points.extend((x1, y) for y in range(min(y1, y2), max(y1, y2) + 1))
    return points

def choosing_corners(striker_loc):
    corners = [(30, 30), (30, target_image.shape[0] - 30), (target_image.shape[1] - 30, 30), (target_image.shape[1] - 30, target_image.shape[0] - 30)]
    return [corner for corner in corners if distance(corner[0], corner[1], striker_loc[0], striker_loc[1]) >= 1000]

def coin_on_track(optimal_coin, optimal_corner, coins_loc):
    points = generate_points_on_line(optimal_coin[0], optimal_coin[1], optimal_corner[0], optimal_corner[1])
    coins_loc.remove(optimal_coin)
    return any(coin in points for coin in coins_loc)

def extend_board_image(image):
    h, w = image.shape[:2]
    extended_image = np.zeros((3 * h, 3 * w, 3), dtype=np.uint8)
    extended_image[h:2 * h, w:2 * w] = image
    extended_image[:h, w:2 * w] = cv2.flip(image, 0)
    extended_image[2 * h:, w:2 * w] = cv2.flip(image, 0)
    extended_image[h:2 * h, :w] = cv2.flip(image, 1)
    extended_image[h:2 * h, 2 * w:] = cv2.flip(image, 1)
    extended_image[:h, :w] = cv2.flip(cv2.flip(image, 0), 1)
    extended_image[:h, 2 * w:] = cv2.flip(cv2.flip(image, 0), 1)
    extended_image[2 * h:, :w] = cv2.flip(cv2.flip(image, 0), 1)
    extended_image[2 * h:, 2 * w:] = cv2.flip(cv2.flip(image, 0), 1)
    return extended_image

def find_reflection_paths(striker_loc, target_corners):
    board_width, board_height = target_image.shape[1], target_image.shape[0]
    reflections = [
        (2 * board_width - striker_loc[0], striker_loc[1]),  # Right wall
        (striker_loc[0], 2 * board_height - striker_loc[1]),  # Bottom wall
        (striker_loc[0], -striker_loc[1]),  # Top wall
        (-striker_loc[0], striker_loc[1])  # Left wall
    ]
    
    paths = [(striker_loc, corner) for corner in target_corners]
    for corner in target_corners:
        for reflection in reflections:
            paths.append((reflection, corner))
    
    return paths

def choose_optimal_coin_and_path(striker_loc, coins_loc, target_corners):
    paths = find_reflection_paths(striker_loc, target_corners)
    d_dict = {}

    for path in paths:
        for coin in coins_loc:
            d = distance(path[0][0], path[0][1], coin[0], coin[1]) + distance(coin[0], coin[1], path[1][0], path[1][1])
            d_dict[d] = (coin, path[1])

    dict_keys = list(d_dict.keys())
    dict_keys.sort()

    for key in dict_keys:
        optimal_coin, optimal_corner = d_dict[key]
        if not coin_on_track(optimal_coin, optimal_corner, coins_loc.copy()):
            return optimal_coin, optimal_corner

# Extend the board image
extended_image = extend_board_image(target_image)

# Template matching and processing
corners = [[30, 30, target_image.shape[1] - 30, target_image.shape[1] - 30], [30, target_image.shape[0] - 30, 30, target_image.shape[0] - 30]]
x_loc = []
y_loc = []
coins_loc = []

for template_path in template_paths:
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(target_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        if is_far_enough(pt, coins_loc):
            cv2.rectangle(target_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 5)
            x_loc.append(pt[0])
            y_loc.append(pt[1])
            coins_loc.append((pt[0], pt[1]))

result = cv2.matchTemplate(target_gray, striker_gray, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)
striker_box = max_loc
cv2.rectangle(target_image, striker_box, (striker_box[0] + striker_gray.shape[1], striker_box[1] + striker_gray.shape[0]), 100, 5)
striker_loc = [striker_box[0] + (striker_gray.shape[1] // 2), striker_box[1] + (striker_gray.shape[0] // 2)]
plt.scatter(striker_loc[0], striker_loc[1], color='darkblue', s=100, marker='+')

target_corners = choosing_corners(striker_loc)
optimal_coin, optimal_corner = choose_optimal_coin_and_path(striker_loc, coins_loc, target_corners)

# Pyplot representation
plt.xlim(0, target_image.shape[1])
plt.ylim(0, target_image.shape[0])
plt.scatter(x_loc, y_loc, marker='o')
plt.scatter(corners[0], corners[1], color='r', marker='s')
plt.scatter(optimal_coin[0], optimal_coin[1], color='green', marker='o', s=100)
plt.scatter(optimal_corner[0], optimal_corner[1], color='green', marker='s', s=100)
plt.plot([striker_loc[0], optimal_coin[0]], [striker_loc[1], optimal_coin[1]], color='orange')
plt.gca().invert_yaxis()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Carrom Board Plot')
plt.grid(True)
plt.show()

# Display the result
cv2.rectangle(target_image, optimal_coin, (optimal_coin[0] + w, optimal_coin[1] + h), (0, 255, 0), 10)
cv2.line(target_image,striker_loc,(optimal_coin[0]+w//2,optimal_coin[1]+h//2),(0, 165, 255),10)
output_image = cv2.resize(target_image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Detected Objects', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

