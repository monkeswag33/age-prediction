import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf, numpy as np
keras = tf.keras
layers = keras.layers
models = keras.models

tf.get_logger().setLevel('ERROR')

genders = {
    0: "Male",
    1: "Female"
}

checkpoint_path = "training/cp.ckpt"
input_shape = (200, 200, 1)
inputs = layers.Input(input_shape)
conv_1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
mp_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(mp_1)
mp_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(mp_2)
mp_3 = layers.MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = layers.Conv2D(256, kernel_size=(3, 3), activation="relu")(mp_3)
mp_4 = layers.MaxPooling2D(pool_size=(2, 2))(conv_4)
flatten = layers.Flatten()(mp_4)
dense_1 = layers.Dense(256, activation="relu")(flatten)

dropout_1 = layers.Dropout(0.2)(dense_1)

output_1 = layers.Dense(1, activation="relu", name="age_out")(dropout_1)

model = models.Model(inputs=[inputs], outputs=[output_1])
model.compile(loss=["mae"], optimizer="adam", metrics=["accuracy"])