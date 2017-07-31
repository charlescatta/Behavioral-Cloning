from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, convolutional.Cropping2D, convolutional.Conv2D, Lambda


# HyperParameters
EPOCHS = 5

model = Sequential([
    Cropping2D(cropping=((100, 0), (0, 0)), input_shape=(160, 320, 3))
    Lambda(lambda pix: (pix / 255.0) - 0.5)
    Conv2D(24, (5, 5), activation="relu"),
    Dropout(0.7),
    Conv2D(36, (5, 5), activation="relu"),
    Conv2D(48, (5, 5), activation="relu"),
    Dropout(0.7),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    Dropout(0.7),
    Flatten(),
    Dense(1164, activation="relu"),
    Dense(100, activation="relu"),
    Dropout(0.7),
    Dense(50, activation="relu"),
    Dense(10, activation="relu")
    ]

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, validation_split=0.15, shuffle=True, nb_epochs=EPOCHS)
