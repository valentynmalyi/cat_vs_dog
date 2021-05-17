import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from cat_vs_dog import settings
from cat_vs_dog.train.data import get_train_data

_base_layers = tf.keras.applications.MobileNetV2(input_shape=(settings.size, settings.size, 3), include_top=False)
_base_layers.trainable = False

model = tf.keras.Sequential([
    _base_layers,
    GlobalAveragePooling2D(),
    Dropout(settings.dropout),
    Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])


def sava_model():
    train = get_train_data()
    model.fit(train, epochs=settings.epochs)
    model.save(settings.model_path)
