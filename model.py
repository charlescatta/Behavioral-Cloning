import csv
import cv2
from tqdm import tqdm
import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Cropping2D, Conv2D

# Parameters
CORRECTION_FACTOR = 0.2
CSV_PATH = "data/driving_log.csv"
IMG_PATH = "data/IMG/"
N_EPOCHS = 5
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.20
KEEP_RATE = 0.75


# Load data

# Helper functions
def get_filename(path):
    return path.split("/")[-1]


def get_img(path, top_path=IMG_PATH):
    return cv2.imread(top_path + get_filename(path))

def split_row(row, append_dict):
    left_img = row[0]
    center_img = row[1]
    right_img = row[2]
    center_angle = float(row[3])
    left_angle = center_angle + CORRECTION_FACTOR 
    right_angle = center_angle - CORRECTION_FACTOR
    append_dict.update(
        {
            left_img: left_angle,
            center_img: center_angle,
            right_img: right_angle
        }
    )

def data_gen(dataset):
    for example in dataset.items():
        yield (get_img(example[0]), example[1])


def parse_csv(input_csv, validation_split=VALIDATION_SPLIT, shuffle_data=True):
    """
    Parse our csv file into a dict so that each row is an image with its associated steering angle
    """
    training_dict = {}
    validation_dict = {}
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        n_examples = len(rows)
        n_validation = int(n_examples * validation_split)
        n_train = n_examples - n_validation
        if shuffle_data:
            shuffle(rows) 
        print("parsing dataset")
        for i in range(n_train):
            split_row(rows[i], training_dict)
        for i in range(n_train, n_examples):
            split_row(rows[i], validation_dict)

    return (training_dict, validation_dict) 

training_data, validation_data = parse_csv(CSV_PATH)
img_shape = get_img(list(training_data.keys())[0]).shape
img_width, img_height, img_depth = img_shape

training_generator = data_gen(training_data)
validation_generator = data_gen(validation_data)
 
"""
Define our model, based on a paper from Nvidia
see: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

model = Sequential([
        Cropping2D(cropping=((100, 0), (0, 0)), input_shape=img_shape)
        Lambda(lambda pix: (pix / 255.0) - 0.5),
        Conv2D(24, 5, 5, activation="relu"),
        Dropout(KEEP_RATE),
        Conv2D(36, 5, 5, activation="relu"),
        Conv2D(48, 5, 5, activation="relu"),
        Dropout(KEEP_RATE),
        Conv2D(64, 3, 3, activation="relu"),
        Conv2D(64, 3, 3, activation="relu"),
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

model.fit_generator(training_generator, BATCH_SIZE, validation_data=validation_generator, validation_steps=int(BATCH_SIZE*VALIDATION_SPLIT), shuffle=True, epochs=N_EPOCHS, verbose=1)
