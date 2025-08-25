"""
Written by Jessy Colonval.
"""
from unittest import TestCase
from random import random

from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

from src.detection.correlation import Spearman
from src.modelisation.point import Point
from src.modelisation.dataset import Dataset


class TestSpearman(TestCase):
    """
    Unit tests that evaluate the correctness of the spearman approach
    implemented.
    """

    def test_matrix_numpy_no_python_list(self):
        """
        Verifies that an exception is throw when a python list is used to store
        the dataset.
        """
        data = [
            [1.4, 0.2, 0],
            [1.3, 0.2, 0],
            [1.5, 0.2, 0],
            [1.7, 0.4, 0],
            [1.4, 0.3, 0],
            [4.5, 1.7, 2],
            [6.3, 1.8, 2],
            [5.8, 1.8, 2],
            [6.1, 2.5, 2],
            [5.1, 2.0, 2],
            [4.1, 1.0, 1],
            [4.5, 1.5, 1],
            [3.9, 1.1, 1],
            [4.8, 1.8, 1],
            [4.0, 1.3, 1],
        ]
        self.assertRaises(TypeError, Spearman.matrix_from_numpy, data)

    def test_matrix_numpy_no_dataframe(self):
        """
        Verifies that an exception is throw when a DataFrame is used to store
        the dataset.
        """
        data = np.array(
            [
                [1.4, 0.2, 0],
                [1.3, 0.2, 0],
                [1.5, 0.2, 0],
                [1.7, 0.4, 0],
                [1.4, 0.3, 0],
                [4.5, 1.7, 2],
                [6.3, 1.8, 2],
                [5.8, 1.8, 2],
                [6.1, 2.5, 2],
                [5.1, 2.0, 2],
                [4.1, 1.0, 1],
                [4.5, 1.5, 1],
                [3.9, 1.1, 1],
                [4.8, 1.8, 1],
                [4.0, 1.3, 1],
            ],
            dtype=np.float64
        )
        df = pd.DataFrame(data)
        self.assertRaises(TypeError, Spearman.matrix_from_numpy, df)

    def test_matrix_numpy_no_dataset(self):
        """
        Verifies that an exception is throw when a Dataset is used to store
        the dataset.
        """
        points = [
            Point(
                np.array([1.4, 0.2], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.3, 0.2], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.5, 0.2], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.7, 0.4], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.4, 0.3], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.7], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([6.3, 1.8], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([5.8, 1.8], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([6.1, 2.5], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([5.1, 2.0], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([4.1, 1.0], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.5], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([3.9, 1.1], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([4.8, 1.8], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([4.0, 1.3], dtype=np.float64),
                np.array([1], dtype=np.int16)
            )
        ]
        dataset = Dataset(
            ["attr0", "attr1"],
            ["class"],
            points
        )
        self.assertRaises(TypeError, Spearman.matrix_from_numpy, dataset)

    def test_matrix_from_numpy_iris(self):
        """
        Verifies that the calculation to the correlation matrix of the Iris
        dataset is correct with the spearman approach.
        """
        data = np.array(
            [
                [1.4, 0.2, 0],
                [1.3, 0.2, 0],
                [1.5, 0.2, 0],
                [1.7, 0.4, 0],
                [1.4, 0.3, 0],
                [4.5, 1.7, 2],
                [6.3, 1.8, 2],
                [5.8, 1.8, 2],
                [6.1, 2.5, 2],
                [5.1, 2.0, 2],
                [4.1, 1.0, 1],
                [4.5, 1.5, 1],
                [3.9, 1.1, 1],
                [4.8, 1.8, 1],
                [4.0, 1.3, 1],
            ],
            dtype=np.float64
        )
        result = Spearman.matrix_from_numpy(data)
        expected = np.array(
            [
                [1.0, 0.9459597657658984, 0.918204965366088],
                [0.9459597657658984, 1.0, 0.9136643851522572],
                [0.918204965366088, 0.9136643851522572, 1.0]
            ],
            dtype=np.float64
        )
        assert_almost_equal(result, expected)

    @settings(max_examples=10000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(10, 10),
            elements=st.integers(min_value=-100, max_value=99)
        )
    )
    def test_matrix_from_numpy_good_shape(self, data):
        """
        Verifies that all correlation scores in the matrix are in the interval
        [-1.0;1.0] and equal to 1.0 when the both attributes are identical.
        Plus, verifies is the matrix is symetric.
        """
        tol = 1e-9  # Epsilon

        # Correlation matrix.
        matrix = Spearman.matrix_from_numpy(data)

        # Is between -1.0 and 1.0.
        low = -1.0 - tol
        high = 1.0 + tol
        is_good_interval = all(
            low <= e <= high
            for line in matrix
            for e in line
        )
        self.assertTrue(is_good_interval, "Wrong interval")

        # Is symetric.
        n_c = len(matrix[0])
        is_symetric = all(
            matrix[i0][i1] == matrix[i1][i0]
            for i0 in range(0, n_c)
            for i1 in range(0, n_c)
        )
        self.assertTrue(is_symetric)

        # Is equal to 1 with identical attributes.
        n_l = len(matrix)
        low = 1.0 - tol
        high = 1.0 + tol
        is_equal_to_1 = all(
            low <= matrix[i][i] <= high
            for i in range(0, n_l)
        )
        self.assertTrue(
            is_equal_to_1,
            "Correlation scores between same attributes aren't equal to 1!"
        )

    def test_matrix_dataframe_iris(self):
        """
        Verifies that the correlation matrix calculated for the Iris dataset
        is correct when it's store in a DataFrame.
        """
        data = np.array(
            [
                [1.4, 0.2, 0],
                [1.3, 0.2, 0],
                [1.5, 0.2, 0],
                [1.7, 0.4, 0],
                [1.4, 0.3, 0],
                [4.5, 1.7, 2],
                [6.3, 1.8, 2],
                [5.8, 1.8, 2],
                [6.1, 2.5, 2],
                [5.1, 2.0, 2],
                [4.1, 1.0, 1],
                [4.5, 1.5, 1],
                [3.9, 1.1, 1],
                [4.8, 1.8, 1],
                [4.0, 1.3, 1],
            ],
            dtype=np.float64
        )
        df = pd.DataFrame(data)
        result = Spearman.matrix(df)
        expected = np.array(
            [
                [1.0, 0.9459597657658984, 0.918204965366088],
                [0.9459597657658984, 1.0, 0.9136643851522572],
                [0.918204965366088, 0.9136643851522572, 1.0]
            ],
            dtype=np.float64
        )
        assert_almost_equal(result, expected)

    @settings(max_examples=10000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(10, 10),
            elements=st.integers(min_value=-100, max_value=99)
        )
    )
    def test_matrix_from_dataframe_good_shape(self, data):
        """
        Verifies that all correlation scores in the matrix are in the interval
        [-1.0;1.0] and equal to 1.0 when the both attributes are identical.
        Plus, verifies is the matrix is symetric.
        """
        tol = 1e-9  # Epsilon

        # Correlation matrix.
        df = pd.DataFrame(data)
        matrix = Spearman.matrix(df)

        # Is between -1.0 and 1.0.
        low = -1.0 - tol
        high = 1.0 + tol
        is_good_interval = all(
            low <= e <= high
            for line in matrix
            for e in line
        )
        self.assertTrue(is_good_interval, "Wrong interval")

        # Is symetric.
        n_c = len(matrix[0])
        is_symetric = all(
            matrix[i0][i1] == matrix[i1][i0]
            for i0 in range(0, n_c)
            for i1 in range(0, n_c)
        )
        self.assertTrue(is_symetric)

        # Is equal to 1 with identical attributes.
        n_l = len(matrix)
        low = 1.0 - tol
        high = 1.0 + tol
        is_equal_to_1 = all(
            low <= matrix[i][i] <= high
            for i in range(0, n_l)
        )
        self.assertTrue(
            is_equal_to_1,
            "Correlation scores between same attributes aren't equal to 1!"
        )

    def test_matrix_dataset_iris(self):
        """
        Verifies that the correlation matrix calculated for the Iris dataset
        is correct when it's store in a Dataset.
        """
        points = [
            Point(
                np.array([1.4, 0.2], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.3, 0.2], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.5, 0.2], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.7, 0.4], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([1.4, 0.3], dtype=np.float64),
                np.array([0], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.7], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([6.3, 1.8], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([5.8, 1.8], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([6.1, 2.5], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([5.1, 2.0], dtype=np.float64),
                np.array([2], dtype=np.int16)
            ),
            Point(
                np.array([4.1, 1.0], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([4.5, 1.5], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([3.9, 1.1], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([4.8, 1.8], dtype=np.float64),
                np.array([1], dtype=np.int16)
            ),
            Point(
                np.array([4.0, 1.3], dtype=np.float64),
                np.array([1], dtype=np.int16)
            )
        ]
        dataset = Dataset(
            ["attr0", "attr1"],
            ["class"],
            points
        )
        result = Spearman.matrix(dataset)
        expected = np.array(
            [
                [1.0, 0.9459597657658984, 0.918204965366088],
                [0.9459597657658984, 1.0, 0.9136643851522572],
                [0.918204965366088, 0.9136643851522572, 1.0]
            ],
            dtype=np.float64
        )
        assert_almost_equal(result, expected)

    def test_matrix_dataset_between_0_and_1(self):
        """
        Verifies that all correlation scores in the matrix are between -1.0 and
        1.0 inclusive when datasets are store in a Dataset.
        """
        n_l = 10  # Number of lines in the dataset.
        n_c = 10  # Number of columns in the dataset.
        _min = -100.0  # Minimum value in the dataset.
        _max = 100.0  # Maximum value in the dataset.

        # Labels common to all datasets.
        col_ctx = ["attr%d" for i in range(0, n_c)]
        col_bhv = ["class"]

        # Generates 10000 random datasets.
        for _ in range(0, 10000):
            points = [
                Point(
                    np.array(
                        [
                            (random() * (_max - _min)) + _min
                            for _ in range(0, n_c)
                        ],
                        dtype=np.float64
                    ),
                    np.array([0], dtype=np.int16)
                )
                for _ in range(0, n_l)
            ]
            dataset = Dataset(
                col_ctx,
                col_bhv,
                points
            )
            matrix = Spearman.matrix(dataset)
            is_good_interval = all(
                -1.0 <= e <= 1.0
                for line in matrix
                for e in line
            )
            self.assertTrue(is_good_interval)

    def test_matrix_dataset_symetric(self):
        """
        Verifies that the correlation matrix is symmetrical, i.e. that two
        pairs of the same attributes are equal when the datasets are store in
        a Dataset.
        """
        n_l = 10  # Number of lines in the dataset.
        n_c = 10  # Number of columns in the dataset.
        _min = -100.0  # Minimum value in the dataset.
        _max = 100.0  # Maximum value in the dataset.

        # Labels common to all datasets.
        col_ctx = ["attr%d" for i in range(0, n_c)]
        col_bhv = ["class"]

        # Generates 10000 random datasets.
        for _ in range(0, 10000):
            points = [
                Point(
                    np.array(
                        [
                            (random() * (_max - _min)) + _min
                            for _ in range(0, n_c)
                        ],
                        dtype=np.float64
                    ),
                    np.array([0], dtype=np.int16)
                )
                for _ in range(0, n_l)
            ]
            dataset = Dataset(
                col_ctx,
                col_bhv,
                points
            )
            matrix = Spearman.matrix(dataset)
            result = all(
                matrix[i0][i1] == matrix[i1][i0]
                for i0 in range(0, n_c)
                for i1 in range(0, n_c)
            )
            self.assertTrue(result)

    def test_matrix_dataset_equals_to_1_for_same_attributes(self):
        """
        Verifies that the correlation scores between two identical attributes
        are equal to 1.0 when datasets are store in a Dataset.
        """
        tol = 1e-9  # Epsilon
        low = 1.0 - tol  # Lower bound.
        high = 1.0 + tol  # Upper bound.

        n_l = 10  # Number of lines in the dataset.
        n_c = 10  # Number of columns in the dataset.
        _min = -100.0  # Minimum value in the dataset.
        _max = 100.0  # Maximum value in the dataset.

        # Labels common to all datasets.
        col_ctx = ["attr%d" for i in range(0, n_c)]
        col_bhv = ["class"]

        # Generates 10000 random datasets.
        for _ in range(0, 10000):
            col = [
                (random() * (_max - _min)) + _min
                for _ in range(0, n_l)
            ]
            points = [
                Point(
                    np.array(
                        [col[i] for _ in range(0, n_c)],
                        dtype=np.float64
                    ),
                    np.array([0], dtype=np.int16)
                )
                for i in range(0, n_l)
            ]

            # Converts the numpy array into a Dataset object.
            dataset = Dataset(
                col_ctx,
                col_bhv,
                points
            )

            # Correlation matrix.
            matrix = Spearman.matrix(dataset)

            is_equal_to_1 = all(
                low <= matrix[i][i] <= high
                for i in range(0, n_l)
            )
            self.assertTrue(
                is_equal_to_1,
                "Correlation scores between same attributes aren't equal to 1!"
            )
