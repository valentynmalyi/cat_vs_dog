from unittest import TestCase

import numpy as np
import tensorflow as tf

from cat_vs_dog import utils


class Test(TestCase):
    def test_resize_image(self):
        image = tf.constant([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
        image = image[..., tf.newaxis]
        first = np.array([[0.625, 0], [0, 0.625]], dtype=np.float32)
        resized_image, _ = utils.resize_image(image, None, 2)
        second = resized_image[..., 0].numpy()
        self.assertTrue(np.array_equal(first, second))

    def test_normalize_image(self):
        image = np.array([255])
        first = np.array([1])
        second, _ = utils.normalize_image(image, None)
        self.assertTrue(np.array_equal(first, second))

    def test_is_valid_image(self):
        self.assertTrue(utils.is_valid_image("1.png"))
        self.assertTrue(utils.is_valid_image("2.jpg"))
        self.assertFalse(utils.is_valid_image("3.py"))


class ImageCategory(TestCase):
    def test_get_from_float(self):
        first = utils.ImageCategory.dog
        second = utils.ImageCategory.get_from_float(10)
        self.assertEqual(first, second)

        first = utils.ImageCategory.cat
        second = utils.ImageCategory.get_from_float(-10)
        self.assertEqual(first, second)

        first = utils.ImageCategory.unknown
        second = utils.ImageCategory.get_from_float(0.5)
        self.assertEqual(first, second)
