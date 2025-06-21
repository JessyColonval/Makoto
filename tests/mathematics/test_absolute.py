"""
Written by Jessy Colonval.
"""
from unittest import TestCase

import random
import string
import numpy as np

from src.mathematics.method_distance import Absolute


class TestAbsolute(TestCase):
    """
    Unit tests of functions that calculate an absolute distance between two
    points.
    """

    @staticmethod
    def randomword(length: int):
        """
        Generates a random word of a given length.

        Parameters
        ----------
        length: int
            Desired word length.
        """
        letters = string.printable
        return ''.join(random.choice(letters) for i in range(length))

    def test_distance_same_object_coords(self):
        """
        Verifies if the distance calculation between two identical objects is
        equal to zero.
        """
        length = 10
        for n in range(1, 1000):
            coords = np.array(
                [TestAbsolute.randomword(length) for _ in range(0, n)],
                dtype=str
            )
            actual = Absolute.distance(coords, coords)
            self.assertEqual(0, actual)

    def test_distance_all_differents(self):
        """
        Verifies if the distance calculation is correct when all elements are
        different in the two points.
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["rose", "violet", "marron"], dtype=str)
        actual = Absolute.distance(a, b)
        self.assertEqual(3, actual)

    def test_distance_one_different(self):
        """
        Verifies if the distance calculation is equal to one when only one
        element is different in the two points.
        """
        a = np.array(["bleu", "rouge", "noir"])
        b = np.array(["bleu", "rouge", "marron"])
        actual = Absolute.distance(a, b)
        self.assertEqual(1, actual)

    def test_distance_two_different(self):
        """
        Verifies if the distance calculation is equal to two when two elements
        are differents in the two points.
        """
        a = np.array(["bleu", "bleu", "noir"])
        b = np.array(["bleu", "violet", "marron"])
        actual = Absolute.distance(a, b)
        self.assertEqual(2, actual)
