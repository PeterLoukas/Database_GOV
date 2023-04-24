import cv2 # OpenCV για τον μετασχηματισμό προοπτικής
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc # Για αποθήκευση εικόνων
import glob  # Για διάβασμα εικόνων
import imageio
import math


#διαβάζει και εμφανίζει μια τυχαία εικόνα από το φάκελο «test_dataset».
path = './test_dataset/IMG/*'
img_list = glob.glob(path)
#Επιλογή μιάς τυχαίας εικόνας
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
plt.imshow(image)


#Αυτή την τυχαία εικόνα την κάνει display image σε RGB
print(image.dtype, image.shape, np.min(image), np.max(image))

red_channel = np.copy(image)
red_channel[:,:,[1, 2]] = 0 # Μηδενίζουμε τα κανάλια G και B

green_channel = np.copy(image)
green_channel[:,:,[0, 2]] = 0 # Μηδενίζουμε τα κανάλια R και B

blue_channel = np.copy(image)
blue_channel[:,:,[0, 1]] = 0 # Μηδενίζουμε τα κανάλια G και R

fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(red_channel)
plt.subplot(132)
plt.imshow(green_channel)
plt.subplot(133)
plt.imshow(blue_channel)
plt.show()


#Κάνει display grid_image and rock_image (βαθμονομημένες)
example_grid = './calibration_images/example_grid1.jpg'
example_rock = './calibration_images/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)
plt.show()


# Ορίστε μια συνάρτηση για να πραγματοποιήσετε μετασχηματισμό προοπτικής
# εικόνα σε κάτοψη!
def perspect_transform(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

# Define the source and destination points for perspective transform
src = np.float32([[15.2, 140.6], [303.2, 131.8], [198.6, 107.8], [118.8, 96.1]])
dst_size = 5
bottom_offset = 6
dst = np.float32([
    [grid_img.shape[1]/2 - dst_size, grid_img.shape[0] - bottom_offset],
    [grid_img.shape[1]/2 + dst_size, grid_img.shape[0] - bottom_offset],
    [grid_img.shape[1]/2 + dst_size, grid_img.shape[0] - 2*dst_size - bottom_offset],
    [grid_img.shape[1]/2 - dst_size, grid_img.shape[0] - 2*dst_size - bottom_offset]])

# Perform perspective transform
warped = perspect_transform(grid_img, src, dst)

# Display warped image
plt.imshow(warped)
plt.show()


#Κάνω display το rock_image σε RGB, ώστε να πάρω τα lower-upper limits, για τη συνάρτηση "rock_thresh", σύμφωνα με την οδηγία:
"""
Για τα πετρώματα, σκεφτείτε να επιβάλλετε ένα κατώτερο και ανώτερο όριο στην επιλογή χρωμάτων για να είστε πιο συγκεκριμένοι
σχετικά με την επιλογή χρωμάτων. Μπορείτε να διερευνήσετε τα χρώματα των πετρωμάτων (τιμές pixel RGB)
σε ένα διαδραστικό παράθυρο matplotlib για να αποκτήσετε μια αίσθηση για το κατάλληλο εύρος κατωφλίου
(λάβετε υπόψη ότι μπορεί να θέλετε διαφορετικά εύρη για καθένα από τα R, G και B!).
"""

example_rock = './calibration_images/example_rock1.jpg'
rock_img = mpimg.imread(example_rock)

print(image.dtype, image.shape, np.min(image), np.max(image))

red_channel = np.copy(rock_img)
red_channel[:,:,[1, 2]] = 0 # Μηδενίζουμε τα κανάλια G και B

green_channel = np.copy(rock_img)
green_channel[:,:,[0, 2]] = 0 # Μηδενίζουμε τα κανάλια R και B

blue_channel = np.copy(rock_img)
blue_channel[:,:,[0, 1]] = 0 # Μηδενίζουμε τα κανάλια G και R

fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(red_channel)
plt.subplot(132)
plt.imshow(green_channel)
plt.subplot(133)
plt.imshow(blue_channel)
plt.show()


#Κατωφλιοποίηση μετασχηματισμένης εικόνας προοπτικής. (warped and then thresholded)!ι
# Το κατώφλι για RGB > 160 είναι μια καλή εκτίμηση για να βρούμε τα pixels που θέλουμε, δηλαδή "φωτεινό" λευκό
def ground_thresh(img, rgb_thresh=(160, 160, 160)):

    # Δημιουργία ενός πίνακα με μηδενικά για να αποθηκεύσουμε την εικόνα που θα προκύψει, αλλά με ένα μόνο channel
    # αφού θα έχουμε τιμές 0 ή 1 (ασπρόμαυρο)

    ground_select = np.zeros_like(img[:, :, 0])

    # Απαιτείται κάθε εικονοστοιχείο να είναι πάνω από τις τρεις τιμές κατωφλίου σε RGB
    # Το above_thresh θα περιέχει τώρα έναν δυαδικό πίνακα με "True"
    # όπου ικανοποιήθηκε το όριο

    above_thresh = (img[:, :, 0] > rgb_thresh[0]) & (img[:, :, 1] > rgb_thresh[1]) & (img[:, :, 2] > rgb_thresh[2])

    ground_select[above_thresh] = 1

    return ground_select


def rock_thresh(img, rgb_thresh=(0, 0, 0)):

    # Define thresholds for RGB color channels
    r_thresh, g_thresh, b_thresh = rgb_thresh

    rock_select = np.zeros_like(img[:, :, 0])

    lower_r_thresh = [152, 95]
    lower_g_thresh = [155, 95]
    lower_b_thresh = [153, 93]

    upper_r_thresh = [166, 195]
    upper_g_thresh = [165, 106]
    upper_b_thresh = [166, 104]

    r_thresh = (img[:, :, 0] > lower_r_thresh[0]) & (img[:, :, 0] < upper_r_thresh[0])
    g_thresh = (img[:, :, 1] > lower_g_thresh[0]) & (img[:, :, 1] < upper_g_thresh[0])
    b_thresh = (img[:, :, 2] > lower_b_thresh[0]) & (img[:, :, 2] < upper_b_thresh[0])

    # Combine the thresholds for each channel
    rgb_thresh = (r_thresh & g_thresh & b_thresh)

    rock_select[rgb_thresh] = 1

    return rock_select


def obstacle_thresh(img, rgb_thresh=(100, 100, 100)):

    obstacle_select = np.zeros_like(img[:, :, 0])

    below_thresh = (img[:, :, 0] < rgb_thresh[0]) & (img[:, :, 1] < rgb_thresh[1]) & (img[:, :, 2] < rgb_thresh[2])

    obstacle_select[below_thresh] = 1

    return obstacle_select


threshed = ground_thresh(warped)
plt.imshow(threshed, cmap='gray')
plt.show()


#Μετασχηματισμός συντεταγμένων
# Μετατροπή από τις συντεταγμένες της εικόνας σε συντεταγμένες rover
def rover_coords(binary_img):

    # Βρείτε τα μη μηδενικά pixels
    ypos, xpos = binary_img.nonzero()

    # Υπολογίστε τις θέσεις pixel με τη θέση του rover να βρίσκεται στο
    # κεντρικό κάτω μέρος της εικόνας.

    xpix = -(ypos - binary_img.shape[0]).astype(np.float64)
    ypix = -(xpos - binary_img.shape[1] / 2).astype(np.float64)

    return xpix, ypix


# Μετατροπή σε πολικές συντεταγμένες
def to_polar_coords(xpix, ypix):

    dist = np.sqrt(xpix ** 2 + ypix ** 2)

    angles = np.arctan2(ypix, xpix)

    return dist, angles


# Εφαρμογή περιστροφής
def rotate_pix(xpix, ypix, yaw):

    # Μετατροπή μοιρών σε ακτίνια
    yaw_rad = yaw * math.pi / 180

    # Εφαρμόστε περιστροφή
    xpix_rotated = (xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad))

    return xpix_rotated, ypix_rotated


# Εφαρμογή μετaτόπισης και κλιμάκωσης
def translate_pix(xpix_rotated, ypix_rotated, xpos, ypos, scale):

    # Εφαρμόστε κλιμάκωση και μετατόπιση
    xpix_translated = (xpix_rotated / scale) + xpos
    ypix_translated = (ypix_rotated / scale) + ypos

    return xpix_translated, ypix_translated


# Ορίστε μια συνάρτηση για εφαρμογή περιστροφής και μετάτόπισης (και αποκοπής).
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):

    # Περιστροφή
    xpix_rotated, ypix_rotated = rotate_pix(xpix, ypix, yaw)

    # Μετατόπιση
    xpix_translated, ypix_translated = translate_pix(xpix_rotated, ypix_rotated, xpos, ypos, scale)

    # Αποκοπή των pixels που πέφτουν έξω από τον κόσμο
    x_pix_world = np.clip(np.int_(xpix_translated), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_translated), 0, world_size - 1)

    # επιστροφή των συντεταγμένων κόσμου

    return x_pix_world, y_pix_world


# Επιλογή τυχαίας εικόνας
idx = np.random.randint(0, len(img_list) - 1)
image = mpimg.imread(img_list[idx])
warped = perspect_transform(image, src, dst)
threshed = ground_thresh(warped)

# Υπολογισμός όλων των pixels σε ρομπο-κεντρικές συντεταγμένες και υπολογισμός απόστασης/γωνίας όλων τον pixels
xpix, ypix = rover_coords(threshed)
dist, angles = to_polar_coords(xpix, ypix)
mean_dir = np.mean(angles)

#display
fig = plt.figure(figsize=(12, 9))
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(warped)
plt.subplot(223)
plt.imshow(threshed, cmap='gray')
plt.subplot(224)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_dir)
y_arrow = arrow_length * np.sin(mean_dir)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
plt.show()

#Διαβάζω τα αποθηκευμένα δεδομένα και έναν ground-truth χάρτη
import pandas as pd

df = pd.read_csv('./test_dataset/robot_log.csv', delimiter=';', decimal='.')
csv_img_list = df["Path"].tolist() # Δημιουργία λίστας με τα ονόματα και τις διαδρομές των εικόνων

# Διαβάζουμε τον χάρτη και τον μετατρέπουμε σε εικόνα με 3 κανάλια
ground_truth = mpimg.imread('./calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float64)

#world_size = 200
# Δημιουργία κλάσης για να κρατάμε τα δεδομένα
# Θα διαβάσει αποθηκευμένα δεδομένα από το αρχείο csv και θα συμπληρώσει αυτό το αντικείμενο
# Ο Χάρτης αντιστοιχεί σε ένα πλέγμα 200 x 200
# σε χώρο 200m x 200m (ίδιο μέγεθος με το χάρτη αλήθειας εδάφους: 200 x 200 pixel)
# Αυτό περιλαμβάνει το πλήρες εύρος τιμών θέσης εξόδου σε x και y από το sim

class Databucket():
    def __init__(self):
        self.images = csv_img_list
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        self.count = 0
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float64) #200x200 (hxw), 3=rgb
        self.ground_truth = ground_truth_3d # Ground truth worldmap = a grayscale image with dimensions of 200x200

# Ορίστε ένα Databucket().. θα είναι global μεταβλητή
# όπου θα μπορούμε να την προσπελάσουμε από παντού
data = Databucket()


# Ορίστε μια συνάρτηση για να φορτώσετε αποθηκευμένες εικόνες
# Αυτή η συνάρτηση θα χρησιμοποιηθεί από το moviepy για τη δημιουργία βίντεο εξόδου
#επεξεργασία αποθηκευμένων εικόνων
def process_image(img):

    data = Databucket()
    print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])
    world_size = 200
    scale = 1
    """
    When the scale value is set to 1, each pixel in the image corresponds to one unit in the world map. 
    This means that the output world map has the same size as the input image. 
    When the scale value is greater than 1, each pixel in the image will correspond to a larger unit in the world map, resulting in a smaller output world map. 
    Conversely, when the scale value is less than 1, each pixel in the image will correspond to a smaller unit in the world map, resulting in a larger output world map.
    """

    # TODO:
    # 1) Ορίστε σημεία προέλευσης και προορισμού για μετασχηματισμό προοπτικής
    src = np.float32([[15.2, 140.6], [303.2, 131.8], [198.6, 107.8], [118.8, 96.1]])
    dst_size = 5
    bottom_offset = 6
    dst = np.float32([
        [grid_img.shape[1] / 2 - dst_size, grid_img.shape[0] - bottom_offset],
        [grid_img.shape[1] / 2 + dst_size, grid_img.shape[0] - bottom_offset],
        [grid_img.shape[1] / 2 + dst_size, grid_img.shape[0] - 2 * dst_size - bottom_offset],
        [grid_img.shape[1] / 2 - dst_size, grid_img.shape[0] - 2 * dst_size - bottom_offset]])

    # 2) Εφαρμόστε μετασχηματισμό προοπτικής
    warped = perspect_transform(grid_img, src, dst)

    # 3) Εφαρμόστε κατώφλι χρώματος για να αναγνωρίσετε προσπελάσιμο έδαφος, εμπόδια και πετρώματα
    threshed_navigable = ground_thresh(warped, rgb_thresh=(160, 160, 160))
    threshed_rocks = rock_thresh(warped, rgb_thresh=(0, 0, 0))
    threshed_obstacles = obstacle_thresh(warped, rgb_thresh=(100, 100, 100))

    # 4) Μετατρέψτε τις συντεταγμένες των pixels στην εικόνα σε ρομποκεντρικές
    xpix_navigable, ypix_navigable = rover_coords(threshed_navigable)
    xpix_rocks, ypix_rocks = rover_coords(threshed_rocks)
    xpix_obstacles, ypix_obstacles = rover_coords(threshed_obstacles)

    """
    # Μετατροπή σε πολικές συντεταγμένες
    dist_navigable, angles_navigable = to_polar_coords(xpix_navigable, ypix_navigable) 
    dist_rocks, angles_rocks = to_polar_coords(xpix_rocks, ypix_rocks) 
    dist_obstacles, angles_obstacles = to_polar_coords(xpix_obstacles, ypix_obstacles) 

    # Εφαρμογή περιστροφής
    xpix_navigable_rotated, ypix_navigable_rotated = rotate_pix(xpix_navigable, ypix_navigable, data.yaw[data.count])
    xpix_rocks_rotated, ypix_rocks_rotated = rotate_pix(xpix_rocks, ypix_rocks, data.yaw[data.count])
    xpix_obstacles_rotated, ypix_obstacles_rotated = rotate_pix(xpix_obstacles, ypix_obstacles, data.yaw[data.count])

    #Εφαρμογή μετaτόπισης και κλιμάκωσης
    xpix_navigable_translated, ypix_navigable_translated = translate_pix(xpix_navigable_rotated, ypix_navigable_rotated, data.xpos[data.count], data.ypos[data.count], scale)
    xpix_rocks_translated, ypix_rocks_translated = translate_pix(xpix_rocks_rotated, ypix_rocks_rotated, data.xpos[data.count], data.ypos[data.count], scale)   
    xpix_obstacles_translated, ypix_obstacles_translated = translate_pix(xpix_obstacles_rotated, ypix_obstacles_rotated, data.xpos[data.count], data.ypos[data.count], scale)
    """

    # 5) Μετατρέψτε τις ρομποκεντρικές συντεταγμένες σε συντεταγμένες περιβάλλοντος
    navigable_x_pix_world, navigable_y_pix_world = pix_to_world(xpix_navigable, ypix_navigable, data.xpos[data.count],
                                                                data.ypos[data.count], data.yaw[data.count], world_size, scale)
    rocks_x_world, rocks_y_world = pix_to_world(xpix_rocks, ypix_rocks, data.xpos[data.count], data.ypos[data.count],
                                                data.yaw[data.count], world_size, scale)
    obstacles_x_world, obstacles_y_world = pix_to_world(xpix_obstacles, ypix_obstacles, data.xpos[data.count],
                                                        data.ypos[data.count], data.yaw[data.count], world_size, scale)

    # 6) Ενημερώστε τον χάρτη του κόσμου
    data.worldmap[navigable_y_pix_world, navigable_x_pix_world, 2] += 1
    data.worldmap[rocks_y_world, rocks_x_world, 1] += 1
    data.worldmap[obstacles_y_world, obstacles_x_world, 0] += 1

    # 7) Δημιουργήστε ένα βίντεο mosaic με τα βήματα που ακολουθήσατε
    # Για αρχη δημιουργήστε ένα κενό image
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1] * 2, 3))
    # Έπειτα τοποθετήστε σε κάθε περιοχή του αυτά που θέλετε

    # Πάνω αριστερά την εικόνα
    output_image[0:img.shape[0], 0:img.shape[1]] = img

    # Πάνω δεξία τη μετασχηματισμένη εικόνα
    output_image[0:img.shape[0], img.shape[1]:] = warped

    # Επίστρωνουμε τον χάρτη που έχουμε με τον χάρτη που περιέχει το ground truth
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)

    # Then putting some text over the image
    cv2.putText(output_image, "Populate this image with your analyses to make a video!", (20, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1

    return output_image


#Δημιουργώ ένα βίντεο με τα επεξεργασμένα δεδομένα
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

# Ορίστε ένα όνομα και ένα μονοπάτι αρχείου
output = './output/test_mapping.mp4'
data = Databucket()
clip = ImageSequenceClip(data.images, fps=60)
new_clip = clip.fl_image(process_image)
new_clip.write_videofile(output, audio=False)

#Diplay το βίντεο που δημιούργησα
from IPython.display import HTML
import io
import base64
video = io.open(output, 'r+b').read()
encoded_video = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded_video.decode('ascii')))