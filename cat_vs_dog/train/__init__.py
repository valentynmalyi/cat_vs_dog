import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from cat_vs_dog import settings


def get_model(size: int = settings.SIZE):
    """Return MobileNetV2 model. It can classify by dog or cat"""
    base_layers = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False)
    base_layers.trainable = False

    model = tf.keras.Sequential([
        base_layers,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model
