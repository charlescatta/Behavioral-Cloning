from helper import CSVPreprocessor, CSVImageDataGen
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Dropout, Lambda, Reshape
from keras.layers.convolutional import Cropping2D, Conv2D


# Parameters
CORRECTION_FACTOR = 0.2
CSV_PATH = "data/driving_log.csv"
IMG_PATH = "data/IMG/"

# Model Hyperparameters
N_EPOCHS = 5
LEARNING_RATE = 0.0009
BATCH_SIZE = 32 
VALIDATION_SPLIT = 0.20
KEEP_RATE = 0.75

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

preprocessor = CSVPreprocessor(CSV_PATH, correction_factor=CORRECTION_FACTOR, validation_split=VALIDATION_SPLIT) 
training_csv, validation_csv = preprocessor.preprocess()

generator = CSVImageDataGen(training_csv, validation_csv, img_dir=IMG_PATH)
img_shape = generator.get_img_shape() 

"""
Define our model, based on a paper from Nvidia
see: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

model = Sequential([
        Lambda(lambda pix: (pix / 255.0) - 0.5, input_shape=img_shape),
        Conv2D(10, 5, 5, activation="relu"),
        Dropout(KEEP_RATE),
        Conv2D(15, 5, 5, activation="relu"),
        Conv2D(20, 5, 5, activation="relu"),
        Dropout(KEEP_RATE),
        Conv2D(22, 3, 3, activation="relu"),
        Conv2D(26, 3, 3, activation="relu"),
        Dropout(KEEP_RATE),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dropout(KEEP_RATE),
        Dense(10, activation="relu"),
        Dense(1, activation="relu"),
        Dense(1)
    ])

print("\t\t\t MODEL SUMMARY")
print(model.summary())

optimizer = Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="mse")

model.fit_generator(generator.training_generator(), 
                    BATCH_SIZE, 
                    validation_data=generator.validation_generator(), 
                    nb_val_samples=int(BATCH_SIZE*VALIDATION_SPLIT), 
                    nb_epoch=N_EPOCHS, 
                    verbose=1)
