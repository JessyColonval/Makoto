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

from src.detection.correlation import MIC
from src.modelisation.point import Point
from src.modelisation.dataset import Dataset


class TestMIC(TestCase):
    """
    Unit tests that evaluate the correctness of the MIC approach
    implemented.
    """

    def test_matrix_from_numpy_no_python_list(self):
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
        self.assertRaises(TypeError, MIC.matrix_from_numpy, data, 0.85, 15.0)

    def test_matrix_from_numpy_no_dataframe(self):
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
        self.assertRaises(TypeError, MIC.matrix_from_numpy, df, 0.85, 15.0)

    def test_matrix_from_numpy_no_dataset(self):
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
        self.assertRaises(TypeError, MIC.matrix_from_numpy, dataset, 0.85, 15)

    @given(st.integers(min_value=-2**63, max_value=0))
    def test_estimator_alpha_less_than_0(self, length: int):
        """
        Verifies that an exception is throw when the length given to computes
        the alpha is negative.
        """
        self.assertRaises(ValueError, MIC.estimator_alpha, length)

    @given(st.integers(min_value=1, max_value=24))
    def test_estimator_alpha_between_1_and_24(self, length: int):
        """
        Verifies that all alpha calculates with a length between 1 and 24 are
        equal to 0.85.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.85)

    @given(st.integers(min_value=25, max_value=49))
    def test_estimator_alpha_between_25_and_49(self, length: int):
        """
        Verifies that all alpha calculates with a length between 25 and 49 are
        equal to 0.80.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.80)

    @given(st.integers(min_value=50, max_value=249))
    def test_estimator_alpha_between_50_and_249(self, length: int):
        """
        Verifies that all alpha calculates with a length between 50 and 249 are
        equal to 0.75.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.75)

    @given(st.integers(min_value=250, max_value=499))
    def test_estimator_alpha_between_250_and_499(self, length: int):
        """
        Verifies that all alpha calculates with a length between 250 and 499
        are equal to 0.70.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.70)

    @given(st.integers(min_value=500, max_value=999))
    def test_estimator_alpha_between_500_and_999(self, length: int):
        """
        Verifies that all alpha calculates with a length between 500 and 999
        are equal to 0.65.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.65)

    @given(st.integers(min_value=1000, max_value=2499))
    def test_estimator_alpha_between_1000_and_2499(self, length: int):
        """
        Verifies that all alpha calculates with a length between 1000 and 2499
        are equal to 0.60.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.60)

    @given(st.integers(min_value=2500, max_value=4999))
    def test_estimator_alpha_between_2500_and_4999(self, length: int):
        """
        Verifies that all alpha calculates with a length between 2500 and 4999
        are equal to 0.55.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.55)

    @given(st.integers(min_value=5000, max_value=9999))
    def test_estimator_alpha_between_5000_and_9999(self, length: int):
        """
        Verifies that all alpha calculates with a length between 5000 and 9999
        are equal to 0.50.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.50)

    @given(st.integers(min_value=10000, max_value=39999))
    def test_estimator_alpha_between_10000_and_39999(self, length: int):
        """
        Verifies that all alpha calculates with a length between 10000 and
        39999 are equal to 0.45.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.45)

    @given(st.integers(min_value=40000, max_value=2**63-1))
    def test_estimator_alpha_between_greater_than_40000(self, length: int):
        """
        Verifies that all alpha calculates with a length greater than 40000.
        """
        alpha = MIC.estimator_alpha(length)
        self.assertEqual(alpha, 0.40)

    def test_matrix_from_numpy_iris(self):
        """
        Verifies that the calculation to the correlation matrix of the Iris
        dataset is correct.
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
        result = MIC.matrix_from_numpy(data, 0.85, 15.0)
        expected = np.array(
            [
                [1.0, 1.0, 0.91829583],
                [1.0, 1.0, 0.91829583],
                [0.91829583, 0.91829583, 1.0]
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
        matrix = MIC.matrix_from_numpy(data, 0.75, 15.0)

        # Is between -1 and 1.
        low = -1.0 - tol
        high = 1.0 + tol
        is_good_interval = all(
            low <= e <= high
            for line in matrix
            for e in line
        )
        self.assertTrue(is_good_interval, "Wrong interval!")

        # Is symetric.
        n_c = len(matrix[0])
        is_symetric = all(
            matrix[i0][i1] == matrix[i1][i0]
            for i0 in range(0, n_c)
            for i1 in range(0, n_c)
        )
        self.assertTrue(is_symetric, "Not symetric!")

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
        result = MIC.matrix(df)
        expected = np.array(
            [
                [1.0, 1.0, 0.91829583],
                [1.0, 1.0, 0.91829583],
                [0.91829583, 0.91829583, 1.0]
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
        matrix = MIC.matrix(df, 0.75, 15.0)

        # Is between -1 and 1.
        low = -1.0 - tol
        high = 1.0 + tol
        is_good_interval = all(
            low <= e <= high
            for line in matrix
            for e in line
        )
        self.assertTrue(is_good_interval, "Wrong interval!")

        # Is symetric.
        n_c = len(matrix[0])
        is_symetric = all(
            matrix[i0][i1] == matrix[i1][i0]
            for i0 in range(0, n_c)
            for i1 in range(0, n_c)
        )
        self.assertTrue(is_symetric, "Not symetric!")

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
        result = MIC.matrix(dataset)
        expected = np.array(
            [
                [1.0, 1.0, 0.91829583],
                [1.0, 1.0, 0.91829583],
                [0.91829583, 0.91829583, 1.0]
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

        tol = 1e-9  # Epsilon
        low = 0.0 - tol  # Lower bound.
        high = 1.0 + tol  # Upper bound.

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
            matrix = MIC.matrix(dataset)
            is_good_interval = all(
                low <= e <= high
                for line in matrix
                for e in line
            )
            self.assertTrue(is_good_interval, "Wrong interval!")

    def test_matrix_dataset_symetric(self):
        """
        Verifies that the correlation scores between two identical attributes
        are equal to 1.0 when datasets are store in a Dataset.
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
            matrix = MIC.matrix(dataset)
            result = all(
                matrix[i0][i1] == matrix[i1][i0]
                for i0 in range(0, n_c)
                for i1 in range(0, n_c)
            )
            self.assertTrue(result, "Not symetric!")

    def test_matrix_dataset_equals_to_1_for_same_attributes(self):
        """
        Verifies that the correlation scores between two identical attributes
        are equal to 1.0 when datasets are store in a Dataset.
        """
        tol = 1e-9  # Epsilon.
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
            dataset = Dataset(col_ctx, col_bhv, points)
            matrix = MIC.matrix(dataset)
            is_equal_to_1 = all(
                low <= matrix[i][i] <= high
                for i in range(0, n_l)
            )
            self.assertTrue(
                is_equal_to_1,
                "Correlation scores between same attributes aren't equal to 1!"
            )
