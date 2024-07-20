import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
# Load the target image
target_image = cv2.imread('assets/carromboard2.jpeg')
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Load multiple template images
template_paths = ['assets/coin_black1.jpeg', 'assets/coin_white1.jpeg']
striker_gray=cv2.imread('assets/coin_striker.jpeg',0)

def distance(x1,y1,x2,y2):
    d=sqrt(((x2-x1)**2) + ((y2-y1)**2))
    return d

def is_far_enough(pt, existing_pts):
    min_dist=50
    for ept in existing_pts:
        if distance(pt[0], pt[1], ept[0], ept[1]) < min_dist:
            return False
    return True

def generate_points_on_line(x1, y1, x2, y2):
    points = []

    # Calculate the slope
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        # for width of striker
        for i in range(-25,26,1):
            k=b+i
            # Generate points
            if abs(x2 - x1) >= abs(y2 - y1):  # More horizontal line
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    y = m * x + k
                    points.append((x, round(y)))
            else:  # More vertical line
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    x = (y - k) / m
                    points.append((round(x), y))
    else:  # Vertical line
        for y in range(min(y1, y2), max(y1, y2) + 1):
            points.append((x1, y))

    return points

def choosing_corners(striker_loc):
    corners=[(30,30),(30,target_image.shape[0]-30),(target_image.shape[1]-30,30),(target_image.shape[1]-30,target_image.shape[0]-30)]
    choosen_corners=[]
    for corner in corners:
        if (distance(corner[0],corner[1],striker_loc[0],striker_loc[1])) >=1000:
            choosen_corners.append(corner)
    return choosen_corners

def coin_on_track(optimal_coin,optimal_corner,coins_loc):
    points=generate_points_on_line(optimal_coin[0],optimal_coin[1],optimal_corner[0],optimal_corner[1])
    coins_loc.remove(optimal_coin)
    for coin in coins_loc:
        if coin in points:
            return True
    else:
        return False

def optimal_coin_choosing(coins_loc,target_corners):
    d_dict={}
    for target_corner in target_corners:
        for coin in coins_loc:
            d_dict[distance(coin[0],coin[1],target_corner[0],target_corner[1])] = (coin,target_corner)
    dict_keys=list(d_dict.keys())
    dict_keys.sort()
    optimal_coin,optimal_corner = d_dict[dict_keys[0]][0],d_dict[dict_keys[0]][1]
    if coin_on_track(optimal_coin,optimal_corner,coins_loc):
        dict_keys.remove(dict_keys[0])
    else:
        return optimal_coin,optimal_corner


# pyplot
corners=[[30,30,target_image.shape[1]-30,target_image.shape[1]-30],[30,target_image.shape[0]-30,30,target_image.shape[0]-30]]
x_loc=[]
y_loc=[] 
coins_loc=[] # this coin loc is for giveing touple location for optimal coin choosing

# Iterate over each template
for template_path in template_paths:
    # Load template image and convert to grayscale
    template = cv2.imread(template_path, 0)  # Read as grayscale directly
    w, h = template.shape[::-1]
    
    # Perform template matching
    res = cv2.matchTemplate(target_gray, template, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.5
    loc = np.where(res >= threshold)
    
    # Draw rectangles around the matches
    for pt in zip(*loc[::-1]):
        if is_far_enough(pt, coins_loc):
            cv2.rectangle(target_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 5)
            x_loc.append(pt[0])
            y_loc.append(pt[1])
            coins_loc.append((pt[0], pt[1]))

### striker identification 
result = cv2.matchTemplate(target_gray,striker_gray, cv2.TM_CCOEFF_NORMED) 
_, _,_,max_loc = cv2.minMaxLoc(result)  # min_val, max_val,min_loc,max_loc = cv2.minMaxLoc(result)
striker_box = max_loc
cv2.rectangle(target_image,striker_box,(striker_box[0] + striker_gray.shape[1], striker_box[1] + striker_gray.shape[0]),100,5)
striker_loc=[striker_box[0] + (striker_gray.shape[1]//2),striker_box[1] + (striker_gray.shape[0]//2)]
plt.scatter(striker_loc[0],striker_loc[1],color='darkblue',s=100,marker='+')

target_corners=choosing_corners(striker_loc)
optimal_coin,optimal_corner = optimal_coin_choosing(coins_loc,target_corners)


# pyplot representation
# Set limits for the plot (optional)
plt.xlim(0, target_image.shape[1])  # Adjust according to image width
plt.ylim(0, target_image.shape[0])  # Adjust according to image height
plt.scatter(x_loc,y_loc,marker='o')
plt.scatter(corners[0],corners[1],color='r',marker='s')
plt.scatter(optimal_coin[0],optimal_coin[1],color='green',marker='o',s=100)
plt.scatter(optimal_corner[0],optimal_corner[1],color='green',marker='s',s=100)
plt.plot([striker_loc[0],optimal_coin[0]],[striker_loc[1],optimal_coin[1]],color='orange')
plt.gca().invert_yaxis()
# Customize plot labels and grid if needed
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Carrom Board Plot')
plt.grid(True)
plt.show()


# Display the result
cv2.rectangle(target_image,optimal_coin,(optimal_coin[0] + w, optimal_coin[1] + h),(0,255,0),10)
cv2.line(target_image,striker_loc,(optimal_coin[0]+w//2,optimal_coin[1]+h//2),(0, 165, 255),10)
output_image = cv2.resize(target_image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Detected Objects', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()