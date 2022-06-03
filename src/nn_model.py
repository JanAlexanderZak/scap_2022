""" Script to build model in tf.keras.
"""

import tensorflow as tf

from tensorflow import keras


def build_model(_input_shape: int) -> keras.engine.sequential.Sequential:
    """ Vanilla tf.Keras model."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape=[_input_shape],),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(64, activation=tf.nn.relu,),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error",],
    )
    return model
