"""
Written by Jessy Colonval.
"""
from abc import abstractmethod
from statistics import mean, stdev
from typing import MutableSequence, MutableMapping, Mapping

import numba as nb
import numpy as np
import numpy.typing as npt


class Normalisation():
    """
    Parent class of the different normalization methods implemented, it defines
    abstract methods for sequence and mapping normalization and static methods
    that retrieve the minimum and maximum values from a list and a mapping.
    """

    @staticmethod
    @abstractmethod
    def normalize_sequence(distances: MutableSequence) -> None:
        """
        Abstract method to normalize of list of distances according a specific
        approach.

        Parameters
        ----------
        distances : MutableSequence
            A list of distances.
        """

    @staticmethod
    @abstractmethod
    def normalize_mapping(distances: MutableMapping) -> None:
        """
        Abstract method to normalize of list of distances according a specific
        approach.

        Parameters
        ----------
        distances : MutableMapping
            A list of distances.
        """

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:]), fastmath=True)
    def _min_sequence(data: npt.NDArray[np.float64]) -> np.float64:
        """
        Gets the smallest value in a list of real numbers.

        Parameters
        ----------
        data : NDArray[float64]
            A list of real numbers.

        Return
        ------
        float64
            The minimum.
        """
        _min = abs(data[0])
        for i in range(1, len(data)):
            _min = min(_min, abs(data[i]))
        return _min

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:]), fastmath=True)
    def _max_sequence(data: npt.NDArray[np.float64]) -> np.float64:
        """
        Gets the highest value in a list of real numbers.

        Parameters
        ----------
        data : NDArray[float64]
            A list of real numbers.

        Return
        ------
        float64
            The maximum.
        """
        _max = abs(data[0])
        for i in range(1, len(data)):
            _max = max(_max, abs(data[i]))
        return _max

    @staticmethod
    @nb.njit(nb.types.UniTuple(nb.float64, 2)(nb.float64[:]), fastmath=True)
    def _min_max_sequence(
        data: npt.NDArray[np.float64]
    ) -> (np.float64, np.float64):
        """
        Gets the smallest and highest value in a list of real numbers.

        Parameters
        ----------
        data : NDArray[float64]
            A list of real numbers.

        Return
        ------
        (float64, float64)
            The minimum and maximum.
        """
        _min = abs(data[0])
        _max = abs(data[0])
        for i in range(1, len(data)):
            abs_val = abs(data[i])
            _min = min(_min, abs_val)
            _max = max(_max, abs_val)
        return _min, _max

    @staticmethod
    @nb.njit(
        nb.float64((nb.types.DictType(nb.int64, nb.float64))),
        fastmath=True
    )
    def _min_mapping(data: Mapping[np.int64, np.float64]) -> np.float64:
        """
        Gets the smallest value in the values of a mapping.

        Parameters
        ----------
        data : Mapping[int64, float64]
            A map between integers and real numbers.

        Return
        ------
        float64
            The minimum.
        """
        _min = np.finfo(np.float64).max
        for val in data.values():
            _min = min(_min, abs(val))
        return _min

    @staticmethod
    @nb.njit(
        nb.float64((nb.types.DictType(nb.int64, nb.float64))),
        fastmath=True
    )
    def _max_mapping(data: Mapping[np.int64, np.float64]) -> float:
        """
        Gets the highest value in the values of a mapping.

        Parameters
        ----------
        data : Mapping[int64, float64]
            A map between integers and real numbers.

        Return
        ------
        float64
            The maximum.
        """
        _max = 0.0
        for val in data.values():
            _max = max(_max, abs(val))
        return _max

    @staticmethod
    @nb.njit(
        nb.types.UniTuple(nb.float64, 2)(nb.types.DictType(nb.int64,
                                                           nb.float64)),
        fastmath=True
    )
    def _min_max_mapping(
        data: Mapping[np.int64, np.float64]
    ) -> (np.float64, np.float64):
        """
        Gets the smallest and highest value in the values of a mapping.

        Parameters
        ----------
        data : Mapping[int64, float64]
            A map between integers and real numbers.

        Return
        ------
        (float64, float64]
            The minimum and maximum.
        """
        _min = np.finfo(np.float64).max
        _max = 0.0
        for val in data.values():
            abs_val = abs(val)
            _min = min(_min, abs_val)
            _max = max(_max, abs_val)
        return _min, _max


class MinMax(Normalisation):
    """
    Class that implements MinMax normalization for lists and mappings.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath=True)
    def __compute(
        val: np.float64, d_min: np.float64, d_max: np.float64
    ) -> np.float64:
        """
        Calculates the normalization of a single value according the MinMax
        approach.

        Parameters
        ----------
        val: float64
            The value to be normalized.
        d_min: float64
            The minimum value present in the collection it shares with 'val'.
        d_max: float64
            The maximum value present in the collection it shares with 'val'.

        Return
        ------
        float64
            The normalization of 'val'.
        """
        if d_max == d_min:
            return 0
        return (val - d_min) / (d_max - d_min)

    @staticmethod
    def normalize_sequence(distances: npt.NDArray[np.float64]) -> None:
        """
        Modifies a list to make it normalized according to the MinMax approach.

        Parameters
        ----------
        distances: NDArray[float64]
            The list of real numbers to be normalized.

        Raises
        ------
        TypeError
            when the list isn't a numpy array.
        ValueError
            when the values aren't integers or real numbers.
        """
        if not isinstance(distances, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        if any(not isinstance(e, (int, float)) for e in distances):
            raise ValueError("Elements must be integers or real.")

        d_min, d_max = MinMax._min_max_sequence(distances)
        for i, d in enumerate(distances):
            distances[i] = MinMax.__compute(abs(d), d_min, d_max)

    @staticmethod
    def normalize_mapping(distances: Mapping[np.int64, np.float64]) -> None:
        """
        Modifies a mapping to make it normalized according to the MinMax
        approach.

        Parameters
        ----------
        distances: Mapping[int64, float64]
            A map between integers and real numbers.

        Raises
        ------
        TypeError
            when the input isn't a mapping.
        ValueError
            when the values aren't integers or real numbers.
        """
        if not isinstance(distances, MutableMapping):
            raise TypeError("Data must be a mutable mapping.")
        if any(not isinstance(e, (int, float)) for e in distances.values()):
            raise ValueError("Elements must be integers or real.")

        d_min, d_max = MinMax._min_max_mapping(distances)
        for key, d in distances.items():
            distances[key] = MinMax.__compute(abs(d), d_min, d_max)


class ZScore(Normalisation):
    """
    Class that implements ZScore normalization for lists and mappings.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath=True)
    def __compute(val: float, val_mean: float, val_std: float) -> float:
        """
        Calculates the normalization of a single value according the ZScore
        approach.

        Parameters
        ----------
        val: float64
            The value to be normalized.
        d_min: float64
            The minimum value present in the collection it shares with 'val'.
        d_max: float64
            The maximum value present in the collection it shares with 'val'.

        Return
        ------
        float64
            The normalization of 'val'.
        """
        if val_std == 0:
            return 0
        return (val - val_mean) / val_std

    @staticmethod
    def normalize_sequence(distances: npt.NDArray[np.float64]) -> None:
        """
        Modifies a list to make it normalized according to the ZScore approach.

        Parameters
        ----------
        distances: NDArray[float64]
            The list of real numbers to be normalized.

        Raises
        ------
        TypeError
            when the list isn't a numpy array.
        ValueError
            when the values aren't integers or real numbers.
        """
        if not isinstance(distances, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        if any(not isinstance(e, (int, float)) for e in distances):
            raise ValueError("Elements must be integers or real.")

        val_mean = mean(distances)
        val_std = stdev(distances)
        for i, d in enumerate(distances):
            distances[i] = ZScore.__compute(d, val_mean, val_std)

    @staticmethod
    def normalize_mapping(distances: Mapping[np.int64, np.float64]) -> None:
        """
        Modifies a mapping to make it normalized according to the ZScore
        approach.

        Parameters
        ----------
        distances: Mapping[int64, float64]
            A map between integers and real numbers.

        Raises
        ------
        TypeError
            when the input isn't a mapping.
        ValueError
            when the values aren't integers or real numbers.
        """
        if not isinstance(distances, MutableMapping):
            raise TypeError("Data must be a mutable mapping.")
        if any(not isinstance(e, (int, float)) for e in distances.values()):
            raise ValueError("Elements must be integers or real.")

        val_mean = mean(distances.values())
        val_std = stdev(distances.values())
        for key, d in distances.items():
            distances[key] = ZScore.__compute(d, val_mean, val_std)


class KPlus1(Normalisation):
    """
    Class that implements KPlus1 normalization for lists and mappings.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True)
    def __compute(val: float, abs_max: float) -> float:
        """
        Calculates the normalization of a single value according the KPlus1
        approach.

        Parameters
        ----------
        val: float64
            The value to be normalized.
        d_min: float64
            The minimum value present in the collection it shares with 'val'.
        d_max: float64
            The maximum value present in the collection it shares with 'val'.

        Return
        ------
        float64
            The normalization of 'val'.
        """
        if abs_max == 0:
            return 0
        return abs(val) / abs_max

    @staticmethod
    def normalize_sequence(distances: npt.NDArray[np.float64]) -> None:
        """
        Modifies a list to make it normalized according to the KPlus1 approach.

        Parameters
        ----------
        distances: NDArray[float64]
            The list of real numbers to be normalized.

        Raises
        ------
        TypeError
            when the list isn't a numpy array.
        ValueError
            when the values aren't integers or real numbers.
        """
        if not isinstance(distances, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        if any(not isinstance(e, (int, float)) for e in distances):
            raise ValueError("Elements must be integers or real.")

        abs_max = KPlus1._max_sequence(distances)
        for i, d in enumerate(distances):
            distances[i] = KPlus1.__compute(d, abs_max)

    @staticmethod
    def normalize_mapping(distances: Mapping[np.int64, np.float64]) -> None:
        """
        Modifies a mapping to make it normalized according to the KPlus1
        approach.

        Parameters
        ----------
        distances: Mapping[int64, float64]
            A map between integers and real numbers.

        Raises
        ------
        TypeError
            when the input isn't a mapping.
        ValueError
            when the values aren't integers or real numbers.
        """
        if not isinstance(distances, MutableMapping):
            raise TypeError("Data must be a mutable mapping.")
        if any(not isinstance(e, (int, float)) for e in distances.values()):
            raise ValueError("Elements must be integers or real.")

        abs_max = KPlus1._max_mapping(distances)
        for key, d in distances.items():
            distances[key] = KPlus1.__compute(d, abs_max)
