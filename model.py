import os 
from helper import CSVPreprocessor, CSVImageDataGen
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Dropout, Lambda, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D, Conv2D


# Parameters
CORRECTION_FACTOR = 0.23
CSV_PATH = "data/driving_log.csv"
IMG_PATH = "data/IMG/"

# Model Hyperparameters
N_EPOCHS = 6
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.10
KEEP_RATE = 0.75
BETA = 0.009

MODEL_NAME = "behavior_cloning-EPOCHS-{}-LEARN_RATE_{}".format(N_EPOCHS, LEARNING_RATE)

print("Training {} ...".format(MODEL_NAME))
# Instantiate preprocessor to process our csv data
preprocessor = CSVPreprocessor(CSV_PATH, correction_factor=CORRECTION_FACTOR, validation_split=VALIDATION_SPLIT) 

# Call the preprocessor and get the new training and validation data
training_csv, validation_csv = preprocessor.preprocess()

# Get our image generator object
image_generator = CSVImageDataGen(training_csv, validation_csv, img_dir=IMG_PATH)
img_shape = image_generator.get_img_shape() 
img_height, img_width, img_depth = img_shape
"""
Define our model, based on a paper from Nvidia
see: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

model = Sequential([
        Cropping2D(cropping=((50, 20), (0, 0)), input_shape=img_shape),
        Lambda(lambda pix: (pix / 255.0) - 0.5),
        Conv2D(36, 5, activation="relu", subsample=(2, 2), kernel_regularizer=l2(BETA)),
        Conv2D(36, 5, activation="relu", subsample=(2, 2), kernel_regularizer=l2(BETA)),
        Conv2D(36, 5, activation="relu", subsample=(2, 2), kernel_regularizer=l2(BETA)),
        Conv2D(64, 3, activation="relu", kernel_regularizer=l2(BETA)),
        Conv2D(64, 3, activation="relu", kernel_regularizer=l2(BETA)),
        Flatten(),
        Dense(500, activation="relu", kernel_regularizer=l2(BETA)),
        Dropout(KEEP_RATE),
        Dense(250, activation="relu", kernel_regularizer=l2(BETA)),
        Dropout(KEEP_RATE),
        Dense(50, activation="relu", kernel_regularizer=l2(BETA)),
        Dropout(KEEP_RATE),
        Dense(1)
    ])

print("\t\t\t MODEL SUMMARY")
print(model.summary())

# Define our model optimizer to have a custom learning rate
optimizer = Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="mse")

model.fit_generator(image_generator.training_data_gen(),
                    steps_per_epoch=int(image_generator.num_train_examples / 5),
                    epochs=N_EPOCHS,
                    verbose=1,
                    validation_data=image_generator.validation_data_gen(),
                    validation_steps=int(image_generator.num_train_examples * VALIDATION_SPLIT)
                    )

# Save our model to disk
model.save(MODEL_NAME)
