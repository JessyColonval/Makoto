"""
Written by Jessy Colonval.
"""
from unittest import TestCase

from random import shuffle
from copy import copy, deepcopy
import numpy as np
from pandas import DataFrame

from src.modelisation.point import Point
from src.modelisation.dataset import Dataset


class TestDataset(TestCase):
    """
    Unit tests for functions in the Dataset's object.
    """

    @classmethod
    def setUpClass(cls):
        cls.__points = [
            Point(
                np.array([1.4, 0.2], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.3, 0.2], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.5, 0.2], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.7, 0.4], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.4, 0.3], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.7], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([6.3, 1.8], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([5.8, 1.8], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([6.1, 2.5], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([5.1, 2.0], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([4.1, 1.0], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.5], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([3.9, 1.1], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([4.8, 1.8], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([4.0, 1.3], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            )
        ]
        cls.__lbs_ctx = ["a1", "a2"]
        cls.__lbs_bhv = ["c1", "c2"]
        cls.__dataset = Dataset(
            cls.__lbs_ctx,
            cls.__lbs_bhv,
            cls.__points
        )

    def test_cant_add_int(self):
        """
        Verifies that an exception is throw when an integer is added in the
        dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        with self.assertRaises(TypeError):
            dataset += 1

    def test_cant_add_str(self):
        """
        Verifies that an exception is throw when a string is added in the
        dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        with self.assertRaises(TypeError):
            dataset += "dQw4w9WgXcQ"

    def test_cant_add_list(self):
        """
        Verifies that an exception is throw when an empty list is added in the
        dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        with self.assertRaises(TypeError):
            dataset += []

    def test_cant_add_dataset(self):
        """
        Verifies that an exception is throw when a Dataset object is added in
        the dataset.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        d1 = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        with self.assertRaises(TypeError):
            d0 += d1

    def test_cant_add_point_w_too_much_contextual_values(self):
        """
        Verifies that an exception is throw when a point with a number of
        contextual values greater than those in the dataset is added.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        p = Point(
            np.array([1.0, 1.0, 1.0]),
            np.array([0, -1], dtype=np.int16)
        )
        with self.assertRaises(ValueError):
            dataset += p

    def test_add_point_w_too_much_behavioral_values(self):
        """
        Verifies that an exception is throw when a point with a number of
        contextual values greater than those in the dataset is added.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, [])
        p = Point(np.array([1.0, 1.0]), np.array([0, -1, -1], dtype=np.int16))
        with self.assertRaises(ValueError):
            dataset += p

    def test_labels_cant_be_tuple(self):
        """
        Verifies that an exception is throw when the labels of contextual
        and behavioral attributes are stored in a tuple.
        """
        self.assertRaises(TypeError, Dataset, ("attr0", "attr1"),
                          self.__lbs_bhv, [])
        self.assertRaises(TypeError, Dataset, self.__lbs_ctx, ("class",), [])
        self.assertRaises(TypeError, Dataset, ("attr0", "attr1"),
                          ("class",), [])

    def test_init_points_wrong_contextuals_number(self):
        """
        Verifies that an exception is throw when the number of contextual
        values in points aren't equal to the number of contextual labels.
        """
        ctxs = ["attr0", "attr1", "attr2"]
        bhvs = ["class"]
        self.assertRaises(ValueError, Dataset, ctxs, bhvs,
                          [Point(
                                np.array([1.0, 1.0], dtype=np.float64),
                                np.array([-1], dtype=np.int16))]
                          )
        self.assertRaises(ValueError, Dataset, ctxs, bhvs,
                          [Point(
                                np.array([1.0, 1.0, 1.0, 1.0],
                                         dtype=np.float64),
                                np.array([-1], dtype=np.int16))]
                          )

    def test_init_points_wrong_behaviors_number(self):
        """
        Verifies that an exception is throw when the number of behavioral
        values in points aren't equal to the number of behavioral labels.
        """
        ctxs = ["attr0", "attr1"]
        bhvs = ["c0", "c1"]
        self.assertRaises(ValueError, Dataset, ctxs, bhvs,
                          [Point(
                                np.array([1.0, 1.0], dtype=np.float64),
                                np.array([0, -1, 20], dtype=np.int16))]
                          )
        self.assertRaises(ValueError, Dataset, ctxs, bhvs,
                          [Point(
                                np.array([1.0, 1.0], dtype=np.float64),
                                np.array([0], dtype=np.int16))]
                          )

    def test_index_out_of_range(self):
        """
        Verifies that an exception is throw when the index of the desired
        point is negative or greater or equal to the total number of points.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(TypeError, dataset.index, len(self.__points))
        self.assertRaises(TypeError, dataset.index, 177013)
        self.assertRaises(TypeError, dataset.index, -177013)

    def test_index_must_be_an_integer(self):
        """
        Verifies that an exception is throw when the index of the desired
        point isn't an interger.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(TypeError, dataset.index, "dQw4w9WgXcQ")
        self.assertRaises(TypeError, dataset.index, 1.0)
        self.assertRaises(ValueError, dataset.index,
                          Point(np.array([1.0, 1.0], dtype=np.float64),
                                np.array([0, -1], dtype=np.int16)))

    def test_contextual_index_wrong_type(self):
        """
        Verifies that an exception is throw when the index of the desired
        contextual value isn't an integer.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(TypeError, dataset.contextuals_index, "0")
        self.assertRaises(TypeError, dataset.contextuals_index, 0.0)
        self.assertRaises(TypeError, dataset.contextuals_index, [])
        self.assertRaises(TypeError, dataset.contextuals_index, (0,))

    def test_behvior_index_wrong_type(self):
        """
        Verifies that an exception is throw when the index of the desired
        behavioral value isn't an integer.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(TypeError, dataset.behaviors_index, "0")
        self.assertRaises(TypeError, dataset.behaviors_index, 0.0)
        self.assertRaises(TypeError, dataset.behaviors_index, [])
        self.assertRaises(TypeError, dataset.behaviors_index, (0,))

    def test_contextual_index_out_of_range(self):
        """
        Verifies that an exception is throw when the index of the desired
        contextual value is negative or greater or equal to the number of
        points.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(ValueError, dataset.contextuals_index,
                          len(self.__points))
        self.assertRaises(ValueError, dataset.contextuals_index, 177013)
        self.assertRaises(ValueError, dataset.contextuals_index, -1)
        self.assertRaises(ValueError, dataset.contextuals_index, -228922)

    def test_behavior_index_out_of_range(self):
        """
        Verifies that an exception is throw when the index of the desired
        behavioral value is negative or greater or equal to the number of
        points.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(ValueError, dataset.behaviors_index,
                          len(self.__points))
        self.assertRaises(ValueError, dataset.behaviors_index, 177013)
        self.assertRaises(ValueError, dataset.behaviors_index, -1)
        self.assertRaises(ValueError, dataset.behaviors_index, -228922)

    def test_contextual_key_not_exist(self):
        """
        Verifies that an exception is throw when the contextual label used
        doesn't exists in the dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(KeyError, dataset.contextuals_key, "attr3")

    def test_behavior_key_not_exist(self):
        """
        Verifies that an exception is throw when the contextual label used
        doesn't exists in the dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertRaises(KeyError, dataset.behaviors_key, "attr3")

    def test_from_dict_different_behavioral_values(self):
        """
        Verifies that an exception is throw when the number of behavioral
        values isn't equal to the number of behavioral attributes provided
        during the dataset creation.
        """
        data0 = {
            "row0": {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            "row1": {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            "row2": {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            "row3": {"a1": 1.7, "a2": 0.4, "c1": 0}
        }
        self.assertRaises(ValueError, Dataset.from_dict, data0, ["a1", "a2"],
                          ["c1", "c2"])
        data1 = {
            "row0": {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            "row1": {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            "row2": {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            "row3": {"a1": 1.7, "a2": 0.4, "c1": 0, "c2": -1, "c3": 10}
        }
        self.assertRaises(ValueError, Dataset.from_dict, data1, ["a1", "a2"],
                          ["c1", "c2"])

    def test_from_dict_different_contextual_values(self):
        """
        Verifies that an exception is throw when the number of contextual
        values isn't equal to the number of contextual attributes provided
        during the dataset creation.
        """
        data0 = {
            "row0": {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            "row1": {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            "row2": {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            "row3": {"a1": 1.7, "c1": 0, "c2": -1}
        }
        self.assertRaises(ValueError, Dataset.from_dict, data0, ["a1", "a2"],
                          ["c1", "c2"])
        data1 = {
            "row0": {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            "row1": {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            "row2": {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            "row3": {"a1": 1.7, "a2": 0.4, "a3": 5.4, "c1": 0, "c2": -1}
        }
        self.assertRaises(ValueError, Dataset.from_dict, data1, ["a1", "a2"],
                          ["c1", "c2"])

    def test_from_dict_contextual_not_exist(self):
        """
        Verifies that an exception is throw when the contextual label in the
        dictionnary isn't present in the labels given in parameter.
        """
        data = {
            "row0": {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            "row1": {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            "row2": {"a1": 1.5, "a3": 0.2, "c1": 0, "c2": -1},
            "row3": {"a1": 1.7, "a2": 0.4, "c1": 0, "c2": -1}
        }
        self.assertRaises(ValueError, Dataset.from_dict, data, ["a1", "a2"],
                          ["c1", "c2"])

    def test_from_dict_behavioral_not_exist(self):
        """
        Verifies that an exception is throw when the behavioral label in the
        dictionnary isn't present in the labels given in parameter.
        """
        data = {
            "row0": {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            "row1": {"a1": 1.3, "a2": 0.2, "c0": 0, "c2": -1},
            "row2": {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            "row3": {"a1": 1.7, "a2": 0.4, "c1": 0, "c2": -1}
        }
        self.assertRaises(ValueError, Dataset.from_dict, data, ["a1", "a2"],
                          ["c1", "c2"])

    def test_from_dataframe_too_much_contextual_attributes(self):
        """
        Verifies that an exception is throw when the number of contextual
        attributes given isn't equal to the number of contextual attributes
        in the DataFrame.
        """
        df = DataFrame()
        df["a1"] = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5,
                    3.9, 4.8, 4.0]
        df["c1"] = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        df["c2"] = [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2]
        self.assertRaises(ValueError, Dataset.from_dataframe, df, ["a1", "a2"],
                          ["c1", "c2"])

    def test_from_dataframe_too_much_behavioral_attributes(self):
        """
        Verifies that an exception is throw when the number of behavioral
        attributes given isn't equal to the number of behavioral attributes
        in the DataFrame.
        """
        df = DataFrame()
        df["a1"] = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5,
                    3.9, 4.8, 4.0]
        df["a2"] = [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0, 1.5,
                    1.1, 1.8, 1.3]
        df["c1"] = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        self.assertRaises(ValueError, Dataset.from_dataframe, df, ["a1", "a2"],
                          ["c1", "c2"])

    def test_from_dataframe_contextual_labels_not_exist(self):
        """
        Verifies that an exception is throw when one of the contextual labels
        given doesn't exist in the DataFrame.
        """
        df = DataFrame()
        df["a1"] = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5,
                    3.9, 4.8, 4.0]
        df["a2"] = [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0, 1.5,
                    1.1, 1.8, 1.3]
        df["c1"] = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        df["c2"] = [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2]
        self.assertRaises(ValueError, Dataset.from_dataframe, df, ["a1", "a3"],
                          ["c1", "c2"])

    def test_from_dataframe_behavior_not_exist(self):
        """
        Verifies that an exception is throw when one of the behavioral labels
        given doesn't exist in the DataFrame.
        """
        df = DataFrame()
        df["a1"] = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5,
                    3.9, 4.8, 4.0]
        df["a2"] = [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0, 1.5,
                    1.1, 1.8, 1.3]
        df["c1"] = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        df["c2"] = [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2]
        self.assertRaises(ValueError, Dataset.from_dataframe, df, ["a1", "a3"],
                          ["c1", "c3"])

    def test_length(self):
        """
        Verifies that the length of the dataset is equal to the number of
        points.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = len(dataset)
        self.assertEqual(actual, 15)

    def test_getitem(self):
        """
        Verifies that all points can be retrieves in the correct order without
        any modification.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        is_present = all(
            (p == dataset[i] and id(p) == id(dataset[i]))
            for i, p in enumerate(self.__points)
        )
        self.assertTrue(is_present)

    def test_add(self):
        """
        Verifies we can add a point at the end of a dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv,
                          [
                              Point(
                                  np.array([1.4, 0.2], dtype=np.float64),
                                  np.array([0, -1], dtype=np.int16))
                          ]
                          )
        point = Point(np.array([1.3, 0.2]), np.array([0, -1], dtype=np.int16))
        dataset += point
        last_index = len(dataset) - 1
        last_point = dataset[last_index]
        self.assertEqual(last_point, point)
        self.assertEqual(id(last_point), id(point))

    def test_equal_not_a_dataset(self):
        """
        Verifies that a dataset isn't equal to the collection of points given
        in parameter.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertNotEqual(dataset, self.__points)

    def test_equal_same_objects_in_constructor(self):
        """
        Verifies that two datasets builds with the same objects are equals.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        d1 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertEqual(d0, d1)

    def test_equal_same_object_dataset(self):
        """
        Verifies that a dataset is equal to itself.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertEqual(dataset, dataset)

    def test_equal_different_object_points(self):
        """
        Verifies that two datasets builds from two different collections of
        points are equal when the points are equals.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        points = [
            Point(np.array([1.4, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.3, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.5, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.7, 0.4]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.4, 0.3]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([4.5, 1.7]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([6.3, 1.8]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([5.8, 1.8]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([6.1, 2.5]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([5.1, 2.0]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([4.1, 1.0]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.5, 1.5]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([3.9, 1.1]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.8, 1.8]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.0, 1.3]), np.array([1, -2], dtype=np.int16))
            ]
        d1 = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        self.assertEqual(d0, d1)

    def test_equal_different_order(self):
        """
        Verifies that two datasets builds from two different collections of
        points are equal when the points are equal but are in a different
        order in the collection.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        points = [
            Point(np.array([1.4, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.5, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.3, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.7, 0.4]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.4, 0.3]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([4.1, 1.0]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.5, 1.5]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([3.9, 1.1]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.8, 1.8]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.0, 1.3]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.5, 1.7]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([6.3, 1.8]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([5.8, 1.8]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([6.1, 2.5]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([5.1, 2.0]), np.array([2, -3], dtype=np.int16))
            ]
        d1 = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        self.assertEqual(d0, d1)

    def test_equal_different_object_contextual_labels(self):
        """
        Verifies that two datasets are equals even when two different
        collections for the contextual labels are used (but still equal).
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        d1 = Dataset(["a1", "a2"], self.__lbs_bhv, self.__points)
        self.assertEqual(d0, d1)

    def test_equal_different_object_behavioral_labels(self):
        """
        Verifies that two datasets are equals even when two different
        collections for the behavioral labels are used (but still equal).
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        d1 = Dataset(self.__lbs_ctx, ["c1", "c2"], self.__points)
        self.assertEqual(d0, d1)

    def test_not_equal_when_only_sub_points_are_presents(self):
        """
        Verifies that two datasets aren't equal when the second one contains
        only a part of points of the first dataset.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        sub_points = self.__points[1:8]
        d1 = Dataset(self.__lbs_ctx, self.__lbs_bhv, sub_points)
        self.assertNotEqual(d0, d1)

    def test_not_equal_when_different_labels_contextual(self):
        """
        Verifies that two datasets aren't equal when the contextual labels
        aren't equal.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        d1 = Dataset(["attrA", "attrB"], self.__lbs_bhv, self.__points)
        self.assertNotEqual(d0, d1)

    def test_not_equal_when_different_labels_behavior(self):
        """
        Verifies that two datasets aren't equal when the behavioral labels
        aren't equal.
        """
        d0 = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        d1 = Dataset(self.__lbs_ctx, ["classA", "classB"], self.__points)
        self.assertNotEqual(d0, d1)

    def test_copy_different_id(self):
        """
        Verifies that the simple copy of a dataset give an another different
        object dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = copy(dataset)
        self.assertNotEqual(id(dataset), id(dataset_copy))

    def test_copy_equal_dataset(self):
        """
        Verifies that the simple copy of a dataset give an another dataset
        equal to the original one.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = copy(dataset)
        self.assertEqual(dataset, dataset_copy)

    def test_copy_same_contextual_labels_object(self):
        """
        Verifies that the simple copy didn't copy the collection of contextual
        labels.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = copy(dataset)
        actual = id(dataset.contextual_labels())
        expected = id(dataset_copy.contextual_labels())
        self.assertEqual(actual, expected)

    def test_copy_same_behavioral_labels_object(self):
        """
        Verifies that the simple copy didn't copy the collection of behavioral
        labels.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = copy(dataset)
        actual = id(dataset.behavioral_labels())
        expected = id(dataset_copy.behavioral_labels())
        self.assertEqual(actual, expected)

    def test_copy_same_object_points(self):
        """
        Verifies that the simple copy didn't copy the collection of points
        and its content.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = copy(dataset)
        self.assertEqual(len(dataset), len(dataset_copy))
        result = all(
            id(p) == id(dataset_copy[i])
            for i, p in enumerate(dataset)
        )
        self.assertTrue(result)

    def test_deepcopy_different_id(self):
        """
        Verifies that the deep copy of a dataset give an another different
        object dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = deepcopy(dataset)
        self.assertNotEqual(id(dataset), id(dataset_copy))

    def test_deepcopy_equal_dataset(self):
        """
        Verifies that the deep copy of a dataset give an another dataset equal
        to the original one.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        dataset_copy = deepcopy(dataset)
        self.assertEqual(dataset, dataset_copy)

    def test_same_order_and_object(self):
        """
        Verifies that the dataset keep the same point order than the original
        collection ginven in parameter.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        self.assertEqual(len(dataset), len(self.__points))
        is_same_order = all(
            id(p) == id(self.__dataset[i])
            for i, p in enumerate(dataset)
            )
        self.assertTrue(is_same_order)

    def test_different_object_contextual_labels(self):
        """
        Verifies that the contextual labels object isn't the same object given
        in parameter.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = id(dataset.contextual_labels())
        expected = id(self.__lbs_ctx)
        self.assertNotEqual(actual, expected)

    def test_different_object_behavioral_labels(self):
        """
        Verifies that the behavioral labels object isn't the same object given
        in parameter.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        result = id(dataset.behavioral_labels())
        expected = id(self.__lbs_bhv)
        self.assertNotEqual(result, expected)

    def test_contextual_labels(self):
        """
        Verifies that the function 'contextual_labels' give the same labels
        used to build the dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = dataset.contextual_labels()
        expected = self.__lbs_ctx
        self.assertListEqual(actual, expected)

    def test_behavioral_labels(self):
        """
        Verifies that the function 'behavioral_labels' give the same labels
        used to build the dataset.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = dataset.behavioral_labels()
        expected = self.__lbs_bhv
        self.assertListEqual(actual, expected)

    def test_index_same_order(self):
        """
        Verifies that the function 'index' give the correct index for every
        points.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        is_same_order = all(
            i == dataset.index(p)
            for i, p in enumerate(dataset)
            )
        self.assertTrue(is_same_order)

    def test_index_random(self):
        """
        Verifies that the function 'index' give the correct index for every
        random indices.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        indices = list(range(0, len(dataset)))
        for i in range(0, 1000):
            shuffle(indices)
            result = all(
                i == dataset.index(dataset[i])
                for i in indices
                )
            self.assertTrue(result)

    def test_contextual_length(self):
        """
        Verifies that the function 'contextual_length' give a number equal to
        the length of contextual labels.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = dataset.contextual_length
        expected = len(self.__lbs_ctx)
        self.assertEqual(actual, expected)

    def test_behavioral_length(self):
        """
        Verifies that the function 'behavioral_length' give a number equal to
        the length of behavioral labels.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        expected = len(self.__lbs_bhv)
        actual = dataset.behavioral_length
        self.assertEqual(actual, expected)

    def test_contextuals_index(self):
        """
        Verifies that the function 'contextuals_index' give the contextual
        values of every points at a given index.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual_0 = dataset.contextuals_index(0)
        actual_1 = dataset.contextuals_index(1)
        expected_0 = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1,
                      4.5, 3.9, 4.8, 4.0,]
        expected_1 = [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0,
                      1.5, 1.1, 1.8, 1.3,]
        self.assertListEqual(actual_0, expected_0)
        self.assertListEqual(actual_1, expected_1)

    def test_behaviors_index(self):
        """
        Verifies that the function 'behaviors_index' give the behavioral
        values of every points at a given index.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual_0 = dataset.behaviors_index(0)
        actual_1 = dataset.behaviors_index(1)
        expected_0 = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        expected_1 = [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2,
                      -2]
        self.assertListEqual(actual_0, expected_0)
        self.assertListEqual(actual_1, expected_1)

    def test_contextuals_key(self):
        """
        Verifies that the function 'contextuals_key' give the contextual
        values of every points at a given contextual label.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual_a1 = dataset.contextuals_key("a1")
        actual_a2 = dataset.contextuals_key("a2")
        expected_a1 = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1,
                       4.5, 3.9, 4.8, 4.0,]
        expected_a2 = [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0,
                       1.5, 1.1, 1.8, 1.3,]
        self.assertListEqual(actual_a1, expected_a1)
        self.assertListEqual(actual_a2, expected_a2)

    def test_behaviors_key(self):
        """
        Verifies that the function 'behaviors_key' give the contextual
        values of every points at a given contextual label.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual_c1 = dataset.behaviors_key("c1")
        actual_c2 = dataset.behaviors_key("c2")
        expected_c1 = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        expected_c2 = [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2,
                       -2]
        self.assertListEqual(actual_c1, expected_c1)
        self.assertListEqual(actual_c2, expected_c2)

    def test_count_behaviors(self):
        """
        Verifies that the function 'count_behaviors' give a dictionnary that
        contains for every behavioral attribute, the number of point for all
        behavioral values.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = dataset.count_behaviors()
        expected = {"c1": {0: 5, 1: 5, 2: 5}, "c2": {-1: 5, -2: 5, -3: 5}}
        self.assertDictEqual(actual, expected)

    def test_group_by(self):
        """
        Verifies that the function 'group_by' given a dictionnary that link
        every behavioral values combinations to a collection of points with
        these values.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        expected = {
            (0, -1): self.__points[0:5],
            (1, -2): self.__points[10:15],
            (2, -3): self.__points[5:10]
            }
        actual = dataset.group_by()
        self.assertDictEqual(actual, expected)

    def test_boudaries(self):
        """
        Verifies that the function 'boundaries' give for every contextual
        attributes the minimal and maximal values.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        expected = [(1.3, 6.3), (0.2, 2.5)]
        actual = dataset.boundaries()
        self.assertListEqual(actual, expected)

    def test_shuffle_rows_all_points_are_present(self):
        """
        Verifies that all points are stil present in the dataset after it
        rows are shulled.
        """
        points = deepcopy(self.__points)  # Avoid modification in main list.
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        for _ in range(0, 1000):
            dataset.shuffle_row()
            are_points_present = all(
                point in dataset
                for point in self.__points
            )
            self.assertTrue(are_points_present)

    def test_shuffle_rows(self):
        """
        Verifies that the function 'shuffle_row' have, at least, made one swap
        between two rows.
        """
        points = deepcopy(self.__points)  # Avoid modification in main list.
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        for i in range(0, 1000):
            dataset.shuffle_row()
            at_least_one_swap = any(
                i != dataset.index(p)
                for i, p in enumerate(self.__points)
            )
            self.assertTrue(at_least_one_swap)

    def test_shuffle_col_contextuals_same_order(self):
        """
        Verifies that the function 'shuffle_contextuals' have swap the two
        contextual values contains in points.
        """
        points = deepcopy(self.__points)  # Avoid modification in main list.
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        indices = np.array([[1, 0]])
        dataset.shuffle_contextuals(indices)
        expected = [
            Point(
                np.array([0.2, 1.4], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([0.2, 1.3], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([0.2, 1.5], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([0.4, 1.7], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([0.3, 1.4], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.7, 4.5], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([1.8, 6.3], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([1.8, 5.8], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([2.5, 6.1], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([2.0, 5.1], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([1.0, 4.1], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([1.5, 4.5], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([1.1, 3.9], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([1.8, 4.8], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([1.3, 4.0], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            )
        ]
        self.assertEqual(len(dataset), len(expected))
        is_shuffled = all(
            p == expected[i]
            for i, p in enumerate(dataset)
        )
        self.assertTrue(is_shuffled)

    def test_shuffle_contextual_labels_swaped(self):
        """
        Verifies that the function 'shuffle_contextuals' swap correctly the
        contextual labels, but not the behavioral ones.
        """
        points = deepcopy(self.__points)  # Avoid modification in main list.
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        dataset.shuffle_contextuals(np.array([[1, 0]], dtype=np.int16))

        # Contextual labels.
        expected = ["a2", "a1"]
        actual = dataset.contextual_labels()
        self.assertListEqual(actual, expected)

        # Behavioral labels
        expected = ["c1", "c2"]
        actual = dataset.behavioral_labels()
        self.assertListEqual(actual, expected)

    def test_shuffle_behavioral_labels_swaped(self):
        """
        Verifies that the function 'shuffle_behaviors' swap correctly the
        behavioral labels, but not the contextual ones.
        """
        points = deepcopy(self.__points)  # Avoid modification in main list.
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, points)
        dataset.shuffle_behaviors(np.array([[1, 0]], dtype=np.int16))

        # Contextual labels.
        expected = ["a1", "a2"]
        actual = dataset.contextual_labels()
        self.assertListEqual(actual, expected)

        # Behavioral labels
        expected = ["c2", "c1"]
        actual = dataset.behavioral_labels()
        self.assertListEqual(actual, expected)

    def test_to_dict(self):
        """
        Verifies that the convertion of a dataset into a dictionnay work
        correctly.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = dataset.to_dict()
        expected = {
            0: {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            1: {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            2: {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            3: {"a1": 1.7, "a2": 0.4, "c1": 0, "c2": -1},
            4: {"a1": 1.4, "a2": 0.3, "c1": 0, "c2": -1},
            5: {"a1": 4.5, "a2": 1.7, "c1": 2, "c2": -3},
            6: {"a1": 6.3, "a2": 1.8, "c1": 2, "c2": -3},
            7: {"a1": 5.8, "a2": 1.8, "c1": 2, "c2": -3},
            8: {"a1": 6.1, "a2": 2.5, "c1": 2, "c2": -3},
            9: {"a1": 5.1, "a2": 2.0, "c1": 2, "c2": -3},
            10: {"a1": 4.1, "a2": 1.0, "c1": 1, "c2": -2},
            11: {"a1": 4.5, "a2": 1.5, "c1": 1, "c2": -2},
            12: {"a1": 3.9, "a2": 1.1, "c1": 1, "c2": -2},
            13: {"a1": 4.8, "a2": 1.8, "c1": 1, "c2": -2},
            14: {"a1": 4.0, "a2": 1.3, "c1": 1, "c2": -2}
        }
        self.assertDictEqual(actual, expected)

    def test_to_dataframe(self):
        """
        Verifies that the convertion of a dataset into a DataFrame work
        correctly.
        """
        dataset = Dataset(self.__lbs_ctx, self.__lbs_bhv, self.__points)
        actual = dataset.to_dataframe()
        datas = {
            "a1": [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5,
                   3.9, 4.8, 4.0],
            "a2": [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0, 1.5,
                   1.1, 1.8, 1.3],
            "c1": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
            "c2": [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2]
        }
        expected = DataFrame(datas)
        expected["c1"] = expected["c1"].astype(np.int16)
        expected["c2"] = expected["c2"].astype(np.int16)
        self.assertTrue(expected.equals(actual))

    def test_from_dict(self):
        """
        Verfies that the build of a dataset from a dictionnary work correctly.
        """
        data_dict = {
            0: {"a1": 1.4, "a2": 0.2, "c1": 0, "c2": -1},
            1: {"a1": 1.3, "a2": 0.2, "c1": 0, "c2": -1},
            2: {"a1": 1.5, "a2": 0.2, "c1": 0, "c2": -1},
            3: {"a1": 1.7, "a2": 0.4, "c1": 0, "c2": -1},
            4: {"a1": 1.4, "a2": 0.3, "c1": 0, "c2": -1},
            5: {"a1": 4.5, "a2": 1.7, "c1": 2, "c2": -3},
            6: {"a1": 6.3, "a2": 1.8, "c1": 2, "c2": -3},
            7: {"a1": 5.8, "a2": 1.8, "c1": 2, "c2": -3},
            8: {"a1": 6.1, "a2": 2.5, "c1": 2, "c2": -3},
            9: {"a1": 5.1, "a2": 2.0, "c1": 2, "c2": -3},
            10: {"a1": 4.1, "a2": 1.0, "c1": 1, "c2": -2},
            11: {"a1": 4.5, "a2": 1.5, "c1": 1, "c2": -2},
            12: {"a1": 3.9, "a2": 1.1, "c1": 1, "c2": -2},
            13: {"a1": 4.8, "a2": 1.8, "c1": 1, "c2": -2},
            14: {"a1": 4.0, "a2": 1.3, "c1": 1, "c2": -2}
        }
        dataset = Dataset.from_dict(data_dict, ["a1", "a2"], ["c1", "c2"])
        expected = [
            Point(np.array([1.4, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.3, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.5, 0.2]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.7, 0.4]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([1.4, 0.3]), np.array([0, -1], dtype=np.int16)),
            Point(np.array([4.5, 1.7]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([6.3, 1.8]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([5.8, 1.8]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([6.1, 2.5]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([5.1, 2.0]), np.array([2, -3], dtype=np.int16)),
            Point(np.array([4.1, 1.0]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.5, 1.5]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([3.9, 1.1]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.8, 1.8]), np.array([1, -2], dtype=np.int16)),
            Point(np.array([4.0, 1.3]), np.array([1, -2], dtype=np.int16))
        ]
        self.assertEqual(len(dataset), len(expected))
        is_all_equal = all(
            p == expected[i]
            for i, p in enumerate(dataset)
        )
        self.assertTrue(is_all_equal)

    def test_from_dataframe(self):
        """
        Verfies that the build of a dataset from a DataFrame work correctly.
        """
        df = DataFrame()
        df["a1"] = [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5,
                    3.9, 4.8, 4.0]
        df["a2"] = [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0, 1.5,
                    1.1, 1.8, 1.3]
        df["c1"] = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        df["c2"] = [-1, -1, -1, -1, -1, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2]
        dataset = Dataset.from_dataframe(df, ["a1", "a2"], ["c1", "c2"])
        expected = [
            Point(
                np.array([1.4, 0.2], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.3, 0.2], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.5, 0.2], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.7, 0.4], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([1.4, 0.3], dtype=np.float64),
                np.array([0, -1], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.7], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([6.3, 1.8], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([5.8, 1.8], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([6.1, 2.5], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([5.1, 2.0], dtype=np.float64),
                np.array([2, -3], dtype=np.int16)
            ),
            Point(
                np.array([4.1, 1.0], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.5], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([3.9, 1.1], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([4.8, 1.8], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            ),
            Point(
                np.array([4.0, 1.3], dtype=np.float64),
                np.array([1, -2], dtype=np.int16)
            )
        ]
        self.assertEqual(len(dataset), len(expected))
        is_all_equal = all(
            p == expected[i]
            for i, p in enumerate(dataset)
        )
        self.assertTrue(is_all_equal)
