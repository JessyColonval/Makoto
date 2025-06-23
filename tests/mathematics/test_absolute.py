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

    def test_distance_different_length(self):
        """
        Verifies that an exception is throw when points don't have the same
        length.
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["rose", "violet"], dtype=str)
        self.assertRaises(ValueError, Absolute.distance, a, b)

    def test_distance_same_object_coords(self):
        """
        Verifies if the distance calculation between two identical objects is
        equal to zero.
        """
        length = 10
        for n in range(1, 100):
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

    def test_subdistance_unknow_index(self):
        """
        Verifies that an exception is throw when one of the indices aren't in
        the coordinates' interval points.
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["rose", "violet", "marron"], dtype=str)
        indices = np.array([0, 3], dtype=np.int64)
        self.assertRaises(ValueError, Absolute.subdistance, a, b, indices)

    def test_subdistance_all_differents(self):
        """
        Verifies that the absolute distance is equal to two when two indices
        are given.
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["rose", "violet", "marron"], dtype=str)
        indices = np.array([0, 1], dtype=np.int64)
        actual = Absolute.subdistance(a, b, indices)
        self.assertEqual(2, actual)

    def test_subdistance_two_differents_w_same(self):
        """
        Verifies that the absolute distance is equal to one when among the
        indices used, the identical one is present
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["bleu", "violet", "marron"], dtype=str)
        indices = np.array([0, 1], dtype=np.int64)
        actual = Absolute.subdistance(a, b, indices)
        self.assertEqual(1, actual)

    def test_subdistance_two_differents_wo_same(self):
        """
        Verifies that the absolute distance is equal to two when among the
        indices used, the identical one is not present
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["bleu", "violet", "marron"], dtype=str)
        indices = np.array([1, 2], dtype=np.int64)
        actual = Absolute.subdistance(a, b, indices)
        self.assertEqual(2, actual)

    def test_subdistance_empty_arrays(self):
        """
        Verifies that the absolute distance between two points without any
        values is equal to 0.
        """
        a = np.array([], dtype=str)
        b = np.array([], dtype=str)
        indices = np.array([], dtype=np.int64)
        actual = Absolute.subdistance(a, b, indices)
        self.assertEqual(0, actual)

    def test_subdistance_different_length(self):
        """
        Verifies that an exception is throw when the two points don't have
        the same length.
        """
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["bleu", "violet"], dtype=str)
        indices = np.array([1, 2], dtype=np.int64)
        self.assertRaises(ValueError, Absolute.subdistance, a, b, indices)
