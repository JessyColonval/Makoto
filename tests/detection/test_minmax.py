"""
Written by Jessy Colonval.
"""
from unittest import TestCase

from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp

import numba as nb
from numba.typed import Dict

import numpy as np
from numpy.testing import assert_equal

from src.detection.normalisation import MinMax


class TestMinMax(TestCase):
    """
    Unit tests that evaluate the MinMax normalization approach.
    """

    @staticmethod
    def to_numba_dict(d):
        """
        Convert a dictionary into a dictionary compatible with numba.
        """
        nb_dict = Dict.empty(
            key_type=nb.int64,
            value_type=nb.float64
        )
        for k, v in d.items():
            nb_dict[k] = v
        return nb_dict

    @settings(max_examples=1000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(100,),
            elements=st.integers(min_value=-2**63, max_value=2**63-1)
        )
    )
    def test_normalize_sequence_good_shape(self, data):
        """
        Verify that a sequence, once normalized using the MinMax approach,
        contains only numbers between 0 and 1.
        """
        # Normalize the random sequence.
        MinMax.normalize_sequence(data)

        # Values must be between 0.0 and 1.0.
        is_between_0_and_1 = all(0 <= e <= 1.0 for e in data)
        self.assertTrue(is_between_0_and_1, "Not in the interval [0.0;1.0].")

    @settings(max_examples=1000, deadline=None)
    @given(
        st.dictionaries(
            keys=st.integers(min_value=-2**63, max_value=2**63-1),  # int64
            values=st.floats(width=64, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=100
        )
    )
    def test_normalize_mapping_good_shape(self, data):
        """
        Verify that a map, once normalized using the MinMax approach, contains
        only values between 0 and 1.
        """
        # Convert the dictionnay into a Numba's dict.
        nb_dict = TestMinMax.to_numba_dict(data)

        MinMax.normalize_mapping(nb_dict)
        is_between_0_and_1 = all(0 <= e <= 1.0 for e in nb_dict.values())
        self.assertTrue(is_between_0_and_1, "Not in the interval [0.0;1.0].")

    @settings(max_examples=1000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(100,),
            elements=st.integers(min_value=-2**63, max_value=2**63)
        )
    )
    def test_renormalize_sequence_have_no_effect(self, data):
        """
        Verify that normalizing an already normalized sequence has no effect.
        """
        MinMax.normalize_sequence(data)
        expected = data.copy()
        MinMax.normalize_sequence(data)
        assert_equal(data, expected)

    @settings(max_examples=1000, deadline=None)
    @given(
        st.dictionaries(
            keys=st.integers(min_value=-2**63, max_value=2**63-1),  # int64
            values=st.floats(width=64, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=100
        )
    )
    def test_renormalize_mapping_have_no_effect(self, data):
        """
        Verify that normalizing an already normalized map has no effect.
        """
        # Convert the dictionnay into a Numba's dict.
        nb_dict = TestMinMax.to_numba_dict(data)

        MinMax.normalize_mapping(nb_dict)
        expected = data.copy()
        MinMax.normalize_mapping(nb_dict)
        assert_equal(data, expected)

    @settings(max_examples=1000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(100,),
            elements=st.integers(min_value=0, max_value=2**63-1)
        )
    )
    def test_normalize_positive_sequence_remains_ascending_order(self, data):
        """
        Verify that a positive sequence sorted in ascending order remains
        sorted after normalization.
        """
        data.sort()
        MinMax.normalize_sequence(data)
        is_ascending = all(
            data[i0] <= data[i1]
            for i0 in range(0, len(data))
            for i1 in range(i0+1, len(data))
        )
        self.assertTrue(is_ascending)

    @settings(max_examples=1000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(100,),
            elements=st.integers(min_value=-2**63, max_value=0)
        )
    )
    def test_normalize_negative_sequence_becomes_descending_order(self, data):
        """
        Verify that a negative sequence sorted in ascending order becomes
        sorted in descending order after normalization.
        """
        data.sort()
        MinMax.normalize_sequence(data)
        is_descending = all(
            data[i0] >= data[i1]
            for i0 in range(0, len(data))
            for i1 in range(i0+1, len(data))
        )
        self.assertTrue(is_descending)

    @settings(max_examples=1000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(100,),
            elements=st.integers(min_value=0, max_value=2**63-1)
        )
    )
    def test_normalize_positive_sequence_remains_descending_order(self, data):
        """
        Verify that a positive sequence sorted in descending order remains
        sorted after normalization.
        """
        data[::-1].sort()
        MinMax.normalize_sequence(data)
        is_descending = all(
            data[i0] >= data[i1]
            for i0 in range(0, len(data))
            for i1 in range(i0+1, len(data))
        )
        self.assertTrue(is_descending)

    @settings(max_examples=1000)
    @given(
        hnp.arrays(
            dtype=np.float64, shape=(100,),
            elements=st.integers(min_value=-2**63, max_value=0)
        )
    )
    def test_normalize_negative_sequence_stay_ascending_order(self, data):
        """
        Verify that a negative sequence sorted in descending order becomes
        sorted in ascending order after normalization.
        """
        data[::-1].sort()
        MinMax.normalize_sequence(data)
        is_ascending = all(
            data[i0] <= data[i1]
            for i0 in range(0, len(data))
            for i1 in range(i0+1, len(data))
        )
        self.assertTrue(is_ascending)

    @settings(max_examples=1000, deadline=None)
    @given(
        st.dictionaries(
            keys=st.integers(min_value=-2**63, max_value=2**63-1),  # int64
            values=st.floats(width=64, allow_nan=False, allow_infinity=False,
                             min_value=0),
            min_size=2, max_size=100
        )
    )
    def test_normalize_positive_mapping_remains_ascending_order(self, data):
        """
        Verify that a positive mapping sorted in ascending order remains
        sorted after normalization.
        """
        # Sort the dict.
        data = dict(sorted(data.items(), key=lambda item: item[1]))

        # Convert the dictionnay into a Numba's dict.
        nb_dict = TestMinMax.to_numba_dict(data)

        # Gets indices before the normalization.
        original_indices = list(nb_dict.keys())

        # Normalize the mapping.
        MinMax.normalize_mapping(nb_dict)

        # Are the values sorted in ascending order?
        values = list(nb_dict.values())
        is_ascending = all(
            values[i0] <= values[i1]
            for i0 in range(0, len(values))
            for i1 in range(i0+1, len(values))
        )
        self.assertTrue(is_ascending)

        # Are indices still in the same order?
        indices = list(nb_dict.keys())
        is_same = len(indices) == len(original_indices) and all(
            indices[i] == original_indices[i]
            for i in range(0, len(original_indices))
        )
        self.assertTrue(is_same)

    @given(
        st.dictionaries(
            keys=st.integers(min_value=-2**63, max_value=2**63-1),  # int64
            values=st.floats(width=64, allow_nan=False, allow_infinity=False,
                             min_value=-2**63, max_value=0),
            min_size=2, max_size=100
        )
    )
    def test_normalize_negative_mapping_becomes_descending_order(self, data):
        """
        Verify that a negative mapping sorted in ascending order becomes
        sorted in descending order after normalization.
        """
        # Sort the dict.
        data = dict(sorted(data.items(), key=lambda item: item[1]))

        # Convert the dictionnay into a Numba's dict.
        nb_dict = TestMinMax.to_numba_dict(data)

        # Gets indices before the normalization.
        original_indices = list(nb_dict.keys())

        # Normalize the mapping.
        MinMax.normalize_mapping(nb_dict)

        # Are the values sorted in descending order?
        values = list(nb_dict.values())
        is_ascending = all(
            values[i0] >= values[i1]
            for i0 in range(0, len(values))
            for i1 in range(i0+1, len(values))
        )
        self.assertTrue(is_ascending)

        # Are indices still in the same order?
        indices = list(nb_dict.keys())
        is_same = len(indices) == len(original_indices) and all(
            indices[i] == original_indices[i]
            for i in range(0, len(original_indices))
        )
        self.assertTrue(is_same)

    @settings(max_examples=1000, deadline=None)
    @given(
        st.dictionaries(
            keys=st.integers(min_value=-2**63, max_value=2**63-1),  # int64
            values=st.floats(width=64, allow_nan=False, allow_infinity=False,
                             min_value=0),
            min_size=2, max_size=100
        )
    )
    def test_normalize_positive_mapping_remains_descending_order(self, data):
        """
        Verify that a positive mapping sorted in descending order remains
        sorted after normalization.
        """
        # Sort the dict.
        data = dict(sorted(data.items(), key=lambda item: item[1],
                           reverse=True))

        # Convert the dictionnay into a Numba's dict.
        nb_dict = TestMinMax.to_numba_dict(data)

        # Gets indices before the normalization.
        original_indices = list(nb_dict.keys())

        # Normalize the mapping.
        MinMax.normalize_mapping(nb_dict)

        # Are the values sorted in ascending order?
        values = list(nb_dict.values())
        is_ascending = all(
            values[i0] >= values[i1]
            for i0 in range(0, len(values))
            for i1 in range(i0+1, len(values))
        )
        self.assertTrue(is_ascending)

        # Are indices still in the same order?
        indices = list(nb_dict.keys())
        is_same = len(indices) == len(original_indices) and all(
            indices[i] == original_indices[i]
            for i in range(0, len(original_indices))
        )
        self.assertTrue(is_same)

    @given(
        st.dictionaries(
            keys=st.integers(min_value=-2**63, max_value=2**63-1),  # int64
            values=st.floats(width=64, allow_nan=False, allow_infinity=False,
                             min_value=-2**63, max_value=0),
            min_size=2, max_size=100
        )
    )
    def test_normalize_negative_mapping_becomes_ascending_order(self, data):
        """
        Verify that a negative mapping sorted in descending order becomes
        sorted in ascending order after normalization.
        """
        # Sort the dict.
        data = dict(sorted(data.items(), key=lambda item: item[1],
                           reverse=True))

        # Convert the dictionnay into a Numba's dict.
        nb_dict = TestMinMax.to_numba_dict(data)

        # Gets indices before the normalization.
        original_indices = list(nb_dict.keys())

        # Normalize the mapping.
        MinMax.normalize_mapping(nb_dict)

        # Are the values sorted in descending order?
        values = list(nb_dict.values())
        is_ascending = all(
            values[i0] <= values[i1]
            for i0 in range(0, len(values))
            for i1 in range(i0+1, len(values))
        )
        self.assertTrue(is_ascending)

        # Are indices still in the same order?
        indices = list(nb_dict.keys())
        is_same = len(indices) == len(original_indices) and all(
            indices[i] == original_indices[i]
            for i in range(0, len(original_indices))
        )
        self.assertTrue(is_same)
