"""
Written by Jessy Colonval.
"""
from unittest import TestCase
from random import random
from numpy import array, float64, int64
from src.mathematics.basic_math import BasicMath


class TestBasicMath(TestCase):
    """
    Unit tests to evaluation all functions in basic_math file.
    """

    def test_dot_positive(self):
        """
        Verifies that the 'dot' function compute the right value with two
        positive vectors.
        """
        x = array([3, 5, 7, 9, 11], dtype=float64)
        y = array([13, 17, 19, 23, 29], dtype=float64)
        actual = BasicMath.dot(x, y)
        self.assertEqual(783, actual)

    def test_dot_negative(self):
        """
        Verifies that the 'dot' function compute the right value with two
        negative vectors.
        """
        x = array([-3, -5, -7, -9, -11], dtype=float64)
        y = array([-13, -17, -19, -23, -29], dtype=float64)
        actual = BasicMath.dot(x, y)
        self.assertEqual(783, actual)

    def test_dot_one_vector_negative(self):
        """
        Verifies that the 'dot' function compute the right value with one
        negative vector and one positive vector.
        """
        x = array([-3, -5, -7, -9, -11], dtype=float64)
        y = array([13, 17, 19, 23, 29], dtype=float64)
        actual = BasicMath.dot(x, y)
        self.assertEqual(-783, actual)

    def test_dot_switch_pos_neg_values(self):
        """
        Verifies that the 'dot' function compute the right value when the two
        vectors contains positive and negative values.
        """
        x = array([-3, 5, -7, 9, -11], dtype=float64)
        y = array([13, -17, 19, -23, 29], dtype=float64)
        result = BasicMath.dot(x, y)
        expected = -783
        self.assertEqual(result, expected)

    def test_dot_id_vector_have_no_influence(self):
        """
        Verifies that the Id vector doesn't influence the result.
        The expected result is alway the sum of the other vector.
        """
        for length in range(1, 100):
            x = array([random() for _ in range(0, length)], dtype=float64)
            y = array([1 for _ in range(0, length)], dtype=float64)
            result = BasicMath.dot(x, y)
            expected = sum(x)
            self.assertAlmostEqual(result, expected)

    def test_dot_empty_array(self):
        """
        Verifies that the result between two empty vector is zero.
        """
        x = array([], dtype=float64)
        y = array([], dtype=float64)
        actual = BasicMath.dot(x, y)
        self.assertEqual(0.0, actual)

    def test_dot_not_numpy_array(self):
        """
        Verifies that an exception is raise when the vectors aren't numpy
        arrays.
        """
        x = [3.0, 5.0]
        y = [7.0, 9.0]
        self.assertRaises(TypeError, BasicMath.dot, x, y)

    def test_dot_not_floating_array(self):
        """
        Verifies that an exception is raise when the vectors values aren't
        float64.
        """
        x = array([3, 5, 7, 9, 11], dtype=int64)
        y = array([13, 17, 19, 23, 29], dtype=int64)
        self.assertRaises(TypeError, BasicMath.dot, x, y)

    def test_dot_different_size(self):
        """
        Verifies that an exception is raise when the vectors aren't the
        same length.
        """
        x = array([3, 5, 7, 9, 11], dtype=float64)
        y = array([13, 17, 19, 23], dtype=float64)
        self.assertRaises(ValueError, BasicMath.dot, x, y)

    def test_positive_vertical_distance(self):
        """
        Verifies that the distance between a point and a segment is always
        equal to the x coordindate of the point when the segment start at
        zero and is vertical.
        """
        start = array([0, 0], dtype=float64)
        for x_end in range(1, 100):
            end = array([x_end, 0])
            for y_pt in range(0, 100):
                point = array([x_end / 2, y_pt], dtype=float64)
                actual = BasicMath.distance_line_nD(start, end, point)
                self.assertEqual(y_pt, actual)

    def test_line_distance_no_numpy_array(self):
        """
        Verifies that an exception is raise when the three points aren't
        numpy arrays.
        """
        start = [3.0, 5.0]
        end = [7.0, 9.0]
        point = [11.0, 13.0]
        self.assertRaises(TypeError, BasicMath.distance_line_nD, start, end,
                          point)

    def test_line_distance_no_floating_array(self):
        """
        Verifies that an exception is raise when the coordinates of the three
        point aren't float.
        """
        start = array([3, 5, 7], dtype=int64)
        end = array([9, 11, 13], dtype=int64)
        point = array([17, 19, 23], dtype=int64)
        self.assertRaises(TypeError, BasicMath.distance_line_nD, start, end,
                          point)

    def test_line_distance_different_dimension_between_line_and_point(self):
        """
        Verifies that an exception is raise when the dimension between the
        points' segment and the point aren't equal.
        """
        start = array([3, 5, 7], dtype=float64)
        end = array([9, 11, 13], dtype=float64)
        point = array([17, 19], dtype=float64)
        self.assertRaises(ValueError, BasicMath.distance_line_nD, start, end,
                          point)

        start = array([3, 5], dtype=float64)
        end = array([9, 11], dtype=float64)
        point = array([17, 19, 23], dtype=float64)
        self.assertRaises(ValueError, BasicMath.distance_line_nD, start, end,
                          point)

    def test_line_distance_not_on_segment(self):
        """
        Verifies that the computation of the distance between a point and a
        segment is correct when the point is not on the segment.
        """
        start = array([0, 0], dtype=float64)
        end = array([0, 4], dtype=float64)
        point = array([1, 3], dtype=float64)
        actual = BasicMath.distance_line_nD(start, end, point)
        self.assertEqual(1.0, actual)
