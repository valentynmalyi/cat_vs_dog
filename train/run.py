import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from settings import Settings
from train.data import download_data


def sava_model():
    train = download_data()
    base_layers = tf.keras.applications.MobileNetV2(input_shape=(Settings.SIZE, Settings.SIZE, 3), include_top=False)
    base_layers.trainable = False

    model = tf.keras.Sequential([
        base_layers,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(train, epochs=3)
    model.save("model")


if __name__ == '__main__':
    sava_model()
