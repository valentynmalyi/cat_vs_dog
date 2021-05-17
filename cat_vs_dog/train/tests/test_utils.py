from unittest import TestCase

import numpy as np
import tensorflow as tf

from cat_vs_dog.train import utils


class Test(TestCase):
    def test_rot90(self):
        image = tf.constant([
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ])
        image = image[..., tf.newaxis]
        first = np.array([
            [0, 0, 0, 0, 5],
            [0, 0, 0, 4, 0],
            [0, 0, 3, 0, 0],
            [0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        rotated_image, _ = utils.rot90(image, None)
        second = rotated_image[..., 0].numpy()
        self.assertTrue(np.array_equal(first, second))
