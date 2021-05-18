import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from cat_vs_dog import settings
from cat_vs_dog.train.data import get_train_data


def get_model(size: int = settings.SIZE):
    """Return model"""
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


def train_and_save_model(model_path: str = settings.MODEL_PATH, epochs: int = settings.EPOCHS):
    """Train model and save it in settings.MODEL_PATH"""
    train = get_train_data()
    model = get_model()
    model.fit(train, epochs=epochs)
    model.save(model_path)
