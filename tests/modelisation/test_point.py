"""
Written by Jessy Colonval.
"""
from unittest import TestCase

from random import random, randint, shuffle
from numba.core.errors import TypingError
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.modelisation.point import Point


class TestPoint(TestCase):

    def test_init_w_float64_and_int16(self):
        """
        Verifies that a point can be create with an array of float64 for
        contextual values and an array of int16 for behavioral values.
        """
        ctxs = np.array([1.3, 0.3], dtype=np.float64)
        bhvs = np.array([0], dtype=np.int16)
        p = Point(ctxs, bhvs)
        self.assertAlmostEqual(p.contextual(0), 1.3)
        self.assertAlmostEqual(p.contextual(1), 0.3)
        self.assertEqual(p.behavior(0), 0)

    def test_init_w_empty_contextual_array(self):
        """
        Verifies that an exception is throw when an empty array is given as
        contextual values.
        """
        self.assertRaises(ValueError, Point, np.empty(0, dtype=np.float64),
                          np.array([0], dtype=np.int16))

    def test_init_w_empty_behavior_array(self):
        """
        Verifies that an exception is throw when an empty array is given as
        behavioral values.
        """
        self.assertRaises(ValueError,
                          Point, np.array([1.3, 0.3], dtype=np.float64),
                          np.empty(0, dtype=np.int16))

    def test_modify_contextuals_dont_affect_point(self):
        """
        Verifies that the modification of the given contextual array after the
        point's creation doesn't affect its own contextual values.
        """
        ctxs = np.array([1.3, 0.3], dtype=np.float64)
        p = Point(ctxs, np.array([0], dtype=np.int16))
        ctxs[0] = 177013.0
        ctxs[1] = 228922.0
        self.assertEqual(p.contextual(0), 1.3)
        self.assertEqual(p.contextual(1), 0.3)

    def test_modify_behaviors_not_affect_point(self):
        """
        Verifies that the modification of the given behavioral array after the
        point's creation doesn't affect its own behavioral values.
        """
        bhvs = np.array([0], dtype=np.int16)
        p = Point(np.array([1.3, 0.3], dtype=np.float64), bhvs)
        bhvs[0] = np.array(177013).astype(np.int16)
        self.assertEqual(p.behavior(0), 0)

    def test_length(self):
        """
        Verifies that the length of a point is always equal to its number of
        contextual and behavioral values.
        """
        for n_ctx in range(1, 100):
            ctxs = np.random.rand(n_ctx)
            for n_bhv in range(1, 100):
                bhvs = np.random.randint(-5, 5, n_bhv, dtype=np.int16)
                point = Point(ctxs, bhvs)
                self.assertEqual(len(point), n_ctx + n_bhv)

    def test_get_item(self):
        """
        Verifies that the get item of a point gives all values (contextual and
        behavioral) in the right order and with contextual first.
        """
        for n_ctx in range(1, 100):
            ctxs = np.random.rand(n_ctx)
            for n_bhv in range(1, 100):
                bhvs = np.random.randint(-5, 5, n_bhv, dtype=np.int16)
                point = Point(ctxs, bhvs)
                for i in range(0, n_ctx):
                    self.assertEqual(point[i], ctxs[i])
                for i in range(n_ctx, n_ctx+n_bhv):
                    self.assertEqual(point[i], bhvs[i-(n_ctx+n_bhv)])

    def test_is_equal_same_value(self):
        """
        Verifies that two points with the same values are equals.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        self.assertEqual(p0, p1)

    def test_is_equal_same_object(self):
        """
        Verifies that the same object Point is equal with itself.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = p0
        self.assertEqual(p0, p1)

    def test_not_equal_to_none(self):
        """
        Verifies that a point isn't equal to None.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        self.assertNotEqual(p0, None)

    def test_not_equal_w_different_behaviors(self):
        """
        Verifies that two points with different behavioral value aren't equals.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([1], dtype=np.int16)
        )
        self.assertNotEqual(p0, p1)

    def test_not_equal_w_different_contextuals(self):
        """
        Verifies that two points with different contextual values aren't
        equals.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = Point(
            np.array([1.7, 0.5], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        self.assertNotEqual(p0, p1)

    def test_not_equal_different_contextual_length(self):
        """
        Verifies that two points with different number of contextual values
        aren't equals.
        """
        p0 = Point(
            np.array([1.4, 0.2, 0.0], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        self.assertNotEqual(p0, p1)

    def test_not_equal_different_behavioral_length(self):
        """
        Verifies that two points with different number of behavioral values
        aren't equals.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0, 1], dtype=np.int16)
        )
        p1 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        self.assertNotEqual(p0, p1)

    def test_copy_point_are_equals(self):
        """
        Verifies that a point is equal to its copy.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = p0.copy()
        self.assertEqual(p0, p1)

    def test_copy_point_have_differents_id(self):
        """
        Verifies that a point and its copy aren't the same object.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        p1 = p0.copy()
        self.assertNotEqual(id(p0), id(p1))

    def test_contextual(self):
        """
        Verifies that the function 'contextual' given the right contextal
        values in the right order.
        """
        bhvs = np.array([0], dtype=np.int16)
        for n_ctx in range(1, 100):
            ctxs = np.random.rand(n_ctx)
            point = Point(ctxs, bhvs)
            for i_ctx in range(0, n_ctx):
                self.assertEqual(point.contextual(i_ctx), ctxs[i_ctx])

    def test_contextual_wrong_indices(self):
        """
        Verifies that an exception is throw when an index greater or equal to
        the number of contextual values is given or when an negative index is
        given.
        """
        bhvs = np.array([0], dtype=np.int16)
        ctxs = np.random.rand(100)
        p = Point(ctxs, bhvs)
        self.assertRaises(IndexError, p.contextual, 100)
        self.assertRaises(IndexError, p.contextual, 101)
        self.assertRaises(IndexError, p.contextual, 1000)
        self.assertRaises(IndexError, p.contextual, -1)
        self.assertRaises(IndexError, p.contextual, -10)

    def test_behavior(self):
        """
        Verifies that the function 'behavior' returns the right behavior value
        in the right order.
        """
        ctxs = np.array([1.4, 0.2], dtype=np.float64)
        for n_bhv in range(1, 100):
            bhvs = np.random.randint(-5, 5, n_bhv, dtype=np.int16)
            point = Point(ctxs, bhvs)
            for i_bhv in range(0, n_bhv):
                self.assertEqual(bhvs[i_bhv], point.behavior(i_bhv))

    def test_behavior_wrong_indices(self):
        """
        Verifies that an exception is throw when an index greater or equal to
        the number of behavioral values is given or when an negative index is
        given.
        """
        bhvs = np.random.randint(-5, 5, 100, dtype=np.int16)
        ctxs = np.random.rand(100)
        p = Point(ctxs, bhvs)
        self.assertRaises(IndexError, p.behavior, 100)
        self.assertRaises(IndexError, p.behavior, 101)
        self.assertRaises(IndexError, p.behavior, 1000)
        self.assertRaises(IndexError, p.behavior, -1)
        self.assertRaises(IndexError, p.behavior, -10)

    def test_number_contextual_values(self):
        """
        Verifies that the number of contextual values is always equal to the
        length of the given contextual array.
        """
        for n_ctx in range(1, 100):
            point = Point(
                np.random.rand(n_ctx),
                np.array([0, 1], dtype=np.int16)
            )
            actual = point.number_contextuals()
            self.assertEqual(actual, n_ctx)

    def test_number_behaviors(self):
        """
        Verifies that the number of behavioral values is always equal to the
        length of the given behavioral array.
        """
        for n_bhv in range(1, 100):
            point = Point(
                np.array([1.4, 0.2], dtype=np.float64),
                np.random.randint(-5, 5, n_bhv, dtype=np.int16)
            )
            actual = point.number_behaviors()
            self.assertEqual(actual, n_bhv)

    def test_shuffle_contextuals(self):
        """
        Verifies that the shuffle of contextual values with a given indices
        works correctly.
        """
        point = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        indices = np.array([[1, 0]], dtype=np.int16)
        point.shuffle_contextuals(indices)
        self.assertEqual(point.contextual(0), 0.2)
        self.assertEqual(point.contextual(1), 1.4)

    def test_shuffle_contextual_values_wrong_index(self):
        """
        Verifies that an exception is throw when an index greater or equal to
        the number of contextual values or an negative number is given.
        """
        p = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0], dtype=np.int16)
        )
        self.assertRaises(ValueError, p.shuffle_contextuals,
                          np.array([[0, 3]], dtype=np.int16))
        self.assertRaises(ValueError, p.shuffle_contextuals,
                          np.array([[0, 10]], dtype=np.int16))
        self.assertRaises(ValueError, p.shuffle_contextuals,
                          np.array([[-1, 0]], dtype=np.int16))
        self.assertRaises(ValueError, p.shuffle_contextuals,
                          np.array([[-10, 0]], dtype=np.int16))

    def test_shuffle_contextual_values_wo_indices(self):
        """
        Verifies that the shuffle of contextual values always results to a
        different order of its values.
        """
        for n_ctx in range(100, 1000):
            ctxs = np.random.rand(n_ctx)
            p = Point(ctxs, np.array([0], dtype=np.int16))
            p.shuffle_contextuals()
            is_same = all(
                p.contextual(i_ctx) == ctxs[i_ctx]
                for i_ctx in range(0, n_ctx)
            )
            self.assertFalse(is_same)

    def test_shuffle_behavior(self):
        """
        Verifies that the shuffle of behavioral values with a given indices
        works correctly.
        """
        p = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0, 1], dtype=np.int16)
        )
        indices = np.array([[1, 0]], dtype=np.int16)
        p.shuffle_behaviors(indices)
        self.assertEqual(p.behavior(0), 1)
        self.assertEqual(p.behavior(1), 0)

    def test_shuffle_behavior_values_wrong_index(self):
        """
        Verifies that an exception is throw when an index greater or equal to
        the number of behavioral values or an negative number is given.
        """
        p = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0, 1], dtype=np.int16)
        )
        self.assertRaises(ValueError, p.shuffle_behaviors,
                          np.array([[0, 2]], dtype=np.int16))
        self.assertRaises(ValueError, p.shuffle_behaviors,
                          np.array([[0, 10]], dtype=np.int16))
        self.assertRaises(ValueError, p.shuffle_behaviors,
                          np.array([[-1, 0]], dtype=np.int16))
        self.assertRaises(ValueError, p.shuffle_behaviors,
                          np.array([[-10, 0]], dtype=np.int16))

    def test_shuffle_behavior_wo_indices(self):
        """
        Verifies that the shuffle of behavioral values always results to a
        different order of its values.
        """
        for n_bhv in range(100, 1000):
            bhvs = np.random.randint(-5, 5, n_bhv, dtype=np.int16)
            p = Point(
                np.array([1.4, 0.2], dtype=np.float64),
                bhvs
            )
            p.shuffle_behaviors()
            is_same = all(
                p.behavior(i_bhv) == bhvs[i_bhv]
                for i_bhv in range(0, n_bhv)
            )
            self.assertFalse(is_same)

    def test_distance_same_point(self):
        """
        Verifies that the distance result is equal to zero when the points
        have the same values.
        """
        p0 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0, 1], dtype=np.int16)
        )
        p1 = Point(
            np.array([1.4, 0.2], dtype=np.float64),
            np.array([0, 1], dtype=np.int16)
        )
        result = p0.distance(p1)
        self.assertEqual(result, 0.0)
