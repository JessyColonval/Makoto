"""
Written by Jessy Colonval.
"""
from unittest import TestCase

import random
import string
import numpy as np

from src.mathematics.method_distance import Absolute


class TestAbsolute(TestCase):

    @staticmethod
    def randomword(length):
        letters = string.printable
        return ''.join(random.choice(letters) for i in range(length))

    def test_distance_nD_same_object_coords(self):
        l_str = 10
        for length in range(1, 1000):
            coords = np.array(
                [TestAbsolute.randomword(l_str) for _ in range(0, length)],
                dtype=str
            )
            result = Absolute.distance(coords, coords)
            expected = 0
            self.assertEqual(result, expected)

    def test_distance_nD_example(self):
        a = np.array(["bleu", "rouge", "noir"], dtype=str)
        b = np.array(["rose", "violet", "marron"], dtype=str)
        result = Absolute.distance(a, b)
        expected = 3
        self.assertEqual(result, expected)

    def test_distance_nD_equal_to_1(self):
        a = np.array(["bleu", "rouge", "noir"])
        b = np.array(["bleu", "violet", "marron"])
        result = Absolute.distance(a, b)
        expected = 2
        self.assertEqual(result, expected)

    def test_distance_nD_equal_twice(self):
        a = np.array(["bleu", "bleu", "noir"])
        b = np.array(["bleu", "violet", "marron"])
        result = Absolute.distance(a, b)
        expected = 2
        self.assertEqual(result, expected)
