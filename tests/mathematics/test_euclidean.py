"""
Written by Jessy Colonval.
"""
from unittest import TestCase

from random import random, randint, sample
from numpy import array, float64, int64

from src.mathematics.method_distance import Euclidean


class TestEuclidean(TestCase):
    """
    Unit tests of functions used to calculate a Euclidean distance between two
    points.
    """

    def test_distance_no_python_lists(self):
        """
        Verifies that an exception is thrown when python lists are used.
        """
        p0 = [3, 5, 7]
        p1 = [9, 11, 13]
        self.assertRaises(TypeError, Euclidean.distance, p0, p1)

    def test_distance_no_int_arrays(self):
        """
        Verifies that an exception is thrown when arrays of integers are used.
        """
        p0 = array([3, 7, 11], dtype=int64)
        p1 = array([5, 9, 13], dtype=int64)
        self.assertRaises(TypeError, Euclidean.distance, p0, p1)

    def test_distance_different_dimensions(self):
        """
        Verifies that an exception is thrown when arrays of differents
        dimensions are used.
        """
        p0 = array([3, 7, 11], dtype=float64)
        p1 = array([5, 9], dtype=float64)
        self.assertRaises(ValueError, Euclidean.distance, p0, p1)
        p0 = array([3, 7], dtype=float64)
        p1 = array([5, 9, 11], dtype=float64)
        self.assertRaises(ValueError, Euclidean.distance, p0, p1)

    def test_distance(self):
        """
        Verifies that the calculation of a Euclidean distance between two
        3-dimensional points works correctly.
        """
        p0 = array([3, 7, 11], dtype=float64)
        p1 = array([5, 9, 13], dtype=float64)
        actual = Euclidean.distance(p0, p1)
        self.assertAlmostEqual(3.4641016151377544, actual)

    def test_distance_positive_and_negative_vectors_equals(self):
        """
        Verifies that the Euclidean distance between two points of any
        dimension with identical values, but one only with positive values and
        the other negative, are identical.
        """
        for length in range(0, 100):
            p0_plus = array([random() for _ in range(0, length)],
                            dtype=float64)
            p0_minus = -p0_plus
            p1_plus = array([random() for _ in range(0, length)],
                            dtype=float64)
            p1_minus = -p1_plus
            res_plus = Euclidean.distance(p0_plus, p1_plus)
            res_minus = Euclidean.distance(p0_minus, p1_minus)
            self.assertEqual(res_plus, res_minus)

    def test_distance_empty_arrays(self):
        """
        Verifies that the Euclidean distance is equal to 0 when the coordinates
        of the two points have no values.
        """
        p0 = array([], dtype=float64)
        p1 = array([], dtype=float64)
        actual = Euclidean.distance(p0, p1)
        self.assertEqual(0.0, actual)

    def test_distance_same_positive_coord(self):
        """
        Verifies that the Euclidean distance between two identical points with
        positive values is 0.
        """
        for length in range(1, 100):
            p0 = array([random() for _ in range(0, length)], dtype=float64)
            p1 = p0
            actual = Euclidean.distance(p0, p1)
            self.assertEqual(0.0, actual)

    def test_distance_same_negative_coord(self):
        """
        Verifies that the Euclidean distance between two identical points with
        negative values is 0.
        """
        for length in range(1, 100):
            p0 = array([-random() for _ in range(0, length)], dtype=float64)
            p1 = p0
            result = Euclidean.distance(p0, p1)
            expected = 0.0
            self.assertEqual(result, expected)

    def test_distance_same_mixed_coord(self):
        """
        Verifies that the Euclidean distance between two identical points with
        positive and negative values is 0.
        """
        for length in range(1, 100):
            p0 = array(
                [
                    random() if i % 2 == 0 else -random()
                    for i in range(0, length)
                ],
                dtype=float64
            )
            p1 = p0
            result = Euclidean.distance(p0, p1)
            expected = 0.0
            self.assertEqual(result, expected)

    def test_subdistance_indices_cant_be_python_list(self):
        """
        Verifies that an exception is thrown when the indices of the values to
        be used is a python list.
        """
        p0 = array([3, 7, 11], dtype=float64)
        p1 = array([5, 9, 13], dtype=float64)
        indices = [0, 1]
        self.assertRaises(TypeError, Euclidean.subdistance, p0, p1, indices)

    def test_subdistance_indices_cant_be_tuple(self):
        """
        Verifies that an exception is thrown when the indices of the values to
        be used is a tuple.
        """
        p0 = array([3, 7, 11], dtype=float64)
        p1 = array([5, 9, 13], dtype=float64)
        indices = (0, 1)
        self.assertRaises(TypeError, Euclidean.subdistance, p0, p1, indices)

    def test_subdistance_index_in_coordinates_range(self):
        """
        Verifies that an exception is thrown when an index is outside points
        index range.
        """
        p0 = array([3, 7, 11], dtype=float64)
        p1 = array([5, 9, 13], dtype=float64)
        indices = array([0, 3], dtype=int64)
        self.assertRaises(ValueError, Euclidean.subdistance, p0, p1, indices)

    def test_subdistance_w_all_indices(self):
        """
        Verifies that the Euclidean distance calculation is identical to that
        calculated when all indices are selected.
        """
        for _ in range(0, 1000):
            length = randint(2, 100)
            indices = array(list(range(0, length)), dtype=int64)
            p0 = array([random() for _ in range(0, length)], dtype=float64)
            p1 = array([random() for _ in range(0, length)], dtype=float64)
            result = Euclidean.subdistance(p0, p1, indices)
            expected = Euclidean.distance(p0, p1)
            self.assertAlmostEqual(result, expected)

    def test_subdistance(self):
        """
        Verifies that the Euclidean distance calculation with part of the
        indices works correctly.
        """
        for _ in range(0, 1000):
            # Random length.
            length = randint(2, 100)
            full_indices = list(range(0, length))

            # Sub-indices used for the construction of sub-list.
            sub_length = randint(2, length)
            sub_indices = array(sample(full_indices, sub_length), dtype=int64)

            # Full coordinates.
            p0 = array([random() for _ in range(0, length)], dtype=float64)
            p1 = array([random() for _ in range(0, length)], dtype=float64)

            # Sub-coordinates.
            sub_p0 = array([p0[i] for i in sub_indices], dtype=float64)
            sub_p1 = array([p1[i] for i in sub_indices], dtype=float64)

            # Verification.
            result = Euclidean.subdistance(p0, p1, sub_indices)
            expected = Euclidean.distance(sub_p0, sub_p1)
            self.assertAlmostEqual(result, expected)
