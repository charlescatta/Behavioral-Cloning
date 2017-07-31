import csv
import cv2
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Cropping2D, Conv2D

# Parameters
CORRECTION_FACTOR = 0.2
CSV_PATH = "data/driving_log.csv"
IMG_PATH = "data/IMG/"
N_EPOCHS = 5
KEEP_RATE = 0.75


# Load data

# Helper functions
def get_filename(path):
    return path.split("/")[-1]


def get_img(path):
    return cv2.imread(IMG_PATH + get_filename(path))


with open(CSV_PATH, 'r') as f:
    # get info on our data
    reader = csv.reader(f)
    lines = [row for row in reader]
    n_examples = 1 + len(lines) * 3  # We have 3 images per lines
    img_width, img_height, img_depth = get_img(lines[0][0]).shape
    X_test = np.array([get_img(lines[0][0]), get_img(lines[0][1])])
    # Preallocate memory for all our data
    X_train = np.zeros((n_examples, img_width, img_height, img_depth))
    y_train = np.zeros(n_examples)
   
    print("Importing data ...")
    idx = 0
    
    for line in tqdm(lines):
        angle_center = float(line[3])
        angle_left = angle_center + CORRECTION_FACTOR
        angle_right = angle_center - CORRECTION_FACTOR
        
        img_center = get_img(line[0])
        img_left = get_img(line[1])
        img_right = get_img(line[2])
        
        X_train[idx] = img_center
        y_train[idx] = angle_center
        X_train[idx + 1] = img_left
        y_train[idx + 1] = angle_left
        X_train[idx + 2] = img_right
        y_train[idx + 3] = angle_right
        
        idx += 3
 
"""
Define our model, based on a paper from Nvidia
see: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

model = Sequential([

        Cropping2D(cropping=((100, 0), (0, 0)), input_shape=(img_width, img_height, img_depth)),
        Lambda(lambda pix: (pix / 255.0) - 0.5),
        Conv2D(24, (5, 5), activation="relu"),
        Dropout(KEEP_RATE),
        Conv2D(36, (5, 5), activation="relu"),
        Conv2D(48, (5, 5), activation="relu"),
        Dropout(KEEP_RATE),
        Conv2D(64, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        Dropout(KEEP_RATE),
        Flatten(),
        Dense(1164, activation="relu"),
        Dense(100, activation="relu"),
        Dropout(KEEP_RATE),
        Dense(50, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ])

print("\t\t\t MODEL SUMMARY")
print(model.summary())

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, validation_split=0.15, shuffle=True, epochs=N_EPOCHS, verbose = 1)
