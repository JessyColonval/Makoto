"""
Written by Jessy Colonval.
"""
from abc import abstractmethod
from math import sqrt
from numba import njit, float64, int64
import numpy as np
import numpy.typing as npt


class MethodDistance():
    """
    Parent object of distance methods' objects.

    Methods
    -------
    distance(a: NDArray[float64], b: NDArray[float64])
        Abstract method of calculating the distance between two points ('a' and
        'b') with identical dimensions.
    subdistance(a: NDArray[float64], b: NDArray[float64],
                indices: NDArray[int64])
        Abstract method of calculating the distance between two points ('a' and
        'b') with identical dimensions using only part of their values.
    """

    @staticmethod
    @abstractmethod
    def distance(a: npt.NDArray[np.float64],
                 b: npt.NDArray[np.float64]) -> float:  # pragma: no cover
        """
        Calculates the distance between two points with identical dimensions.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.
        """

    @staticmethod
    @abstractmethod
    def subdistance(a: npt.NDArray[np.float64],
                    b: npt.NDArray[np.float64],
                    indices: npt.NDArray[np.int64]
                    ) -> float:  # pragma: no cover
        """
        Calculates the distance between two points with identical dimensions
        using only part of their values.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.
        indices : NDArray[int64]
            The list of indices of the 'a' and 'b' which will be used for the
            calculation of the distance.
        """


class Euclidean(MethodDistance):
    """
    Object that groups functions useful for calculating Euclidean distances.

    Methods
    -------
        distance(a: NDArray[float64], b: NDArray[float64]):
            Calculates the Euclidean distance between two points with identical
            dimensions.
        subdistance(a: NDArray[float64], b: NDArray[float64],
                    indices: NDArray[int64]):
            Calculates the Euclidean distance between two points with identical
            dimensions by using only part of their values.
    """

    @staticmethod
    @njit(float64(float64[:], float64[:]), fastmath=True)
    def distance(a: npt.NDArray[np.float64],
                 b: npt.NDArray[np.float64]) -> float:  # pragma: no cover
        """
        Calculates the Euclidean distance between two points with identical
        dimensions.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.

        Returns
        -------
        float
            :math:`\\sqrt{ \\overset{n}{\\underset{i=0}{\\sum}} (b_{i}
            - a_{i})^{2} }`

        Raises
        ------
        ValueError
            when the points doesn't have the same dimension.
        """
        if len(a) != len(b):
            raise ValueError("Points must have same dimension.")

        _sum = 0.0
        for i in range(0, len(a)):
            _sum += (a[i] - b[i])**2
        return sqrt(_sum)

    @staticmethod
    @njit(float64(float64[:], float64[:], int64[:]), fastmath=True)
    def subdistance(a: npt.NDArray[np.float64],
                    b: npt.NDArray[np.float64],
                    indices: npt.NDArray[np.int64]
                    ) -> float:  # pragma: no cover
        """
        Calculates the euclidean distance between two points of identical
        dimensions using only part of the values.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.
        indices : Sequence, optional
            The list of indices of the 'a' and 'b' which will be used for the
            calculation of the distance.

        Returns
        -------
        float
            :math:`\\sqrt{ \\overset{n}{\\underset{i=0}{\\sum}} (b_{i} -
            a_{i})^{2} }`

        Raises
        ------
        ValueError
            when the points doesn't have the same dimension.
        """
        if len(a) != len(b):
            raise ValueError("Points must have same dimension.")

        _sum = 0.0
        for i in indices:
            if i >= len(a):
                raise ValueError("At least one of indices is greater than the",
                                 " dimension of the points.")
            _sum += (a[i] - b[i])**2
        return sqrt(_sum)


class Manhattan(MethodDistance):
    """
    Object that groups functions useful for calculating Manhattan distances.

    Methods
    -------
    distance(a: NDArray[float64], b: NDArray[float64]):
        Calculates the Manhattan distance between two points with identical
        dimensions.
    subdistance(a: NDArray[float64], b: NDArray[float64],
                indices: NDArray[int64]):
        Calculates the Manhattan distance between two points with identical
        dimensions by using only part of their values.
    """

    @staticmethod
    @njit(float64(float64[:], float64[:]), fastmath=True)
    def distance(a: npt.NDArray[np.float64],
                 b: npt.NDArray[np.float64]) -> float:  # pragma: no cover
        """
        Calculates the Manhattan distance between two points with identical
        dimensions.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.

        Returns
        -------
        float
            :math:`\\overset{n}{\\underset{i=0}{\\sum}} |b_{i} - a_{i}|`

        Raises
        ------
        ValueError
            when the points doesn't have the same dimension.
        """
        if len(a) != len(b):
            raise ValueError("Points must have same dimension.")

        _sum = 0.0
        for i in range(0, len(a)):
            _sum += abs(a[i] - b[i])
        return _sum

    @staticmethod
    @njit(float64(float64[:], float64[:], int64[:]), fastmath=True)
    def subdistance(a: npt.NDArray[np.float64],
                    b: npt.NDArray[np.float64],
                    indices: npt.NDArray[np.int64]
                    ) -> float:  # pragma: no cover
        """
        Calculates the Manhattan distance between two points with identical
        dimensions by using only part of their values.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.
        indices : NDArray[int64], optional
            The list of indices of the 'a' and 'b' which will be used for the
            calculation of the distance.

        Returns
        -------
        float
            :math:`\\overset{n}{\\underset{i=0}{\\sum}} |b_{i} - a_{i}|`

        Raises
        ------
        ValueError
            when the points doesn't have the same dimension.
        """
        if len(a) != len(b):
            raise ValueError("Points must have same dimension.")

        _sum = 0.0
        for i in indices:
            if i >= len(a):
                raise ValueError("At least one of indices is greater than the",
                                 " dimension of the points.")
            _sum += abs(a[i] - b[i])
        return _sum


class Absolute(MethodDistance):
    """
    Object that groups functions useful for calculating an absolute distance
    between two points.

    Methods
    -------
    distance(a: NDArray[str], b: NDArray[str]):
        Calculates the absolute distance between two points with identical
        dimensions.
    subdistance(a: NDArray[str], b: NDArray[str],
                indices: NDArray[int64]):
        Calculates the absolute distance between two points with identical
        dimensions by using only part of their values.
    """

    @staticmethod
    def distance(a: npt.NDArray[np.str_], b: npt.NDArray[np.str_]) -> int:
        """
        Calculates the absolute distance between two points with identical
        dimensions.

        Parameters
        ----------
        a : NDArray[str]
            Coordinates of the first point of n dimensions.
        b : NDArray[str]
            Coordinates of the second point of n dimensions.

        Raises
        ------
        ValueError
            when the points doesn't have the same dimension.
        """
        if len(a) != len(b):
            raise ValueError("Points must have same dimension.")

        return sum(
            1
            for i, ai in enumerate(a)
            if ai != b[i]
        )

    @staticmethod
    def subdistance(a: npt.NDArray[np.str_], b: npt.NDArray[np.str_],
                    indices: npt.NDArray[np.int64]) -> int:
        """
        Calculates the absolute distance between two points with identical
        dimensions by using only part of their values.

        Parameters
        ----------
        a : NDArray[float64]
            Coordinates of the first point of n dimensions.
        b : NDArray[float64]
            Coordinates of the second point of n dimensions.
        indices : NDArray[int64]
            The list of indices of the 'a' and 'b' which will be used for the
            calculation of the distance.

        Raises
        ------
        ValueError
            when the points doesn't have the same dimension or when an index
            isn't in the coordinates' interval.
        """
        if len(a) != len(b):
            raise ValueError("Points must have same dimension.")
        if len(a) == 0:
            return 0
        if max(indices) >= len(a) or min(indices) < 0:
            raise ValueError("At least one of indices is greater than the",
                             " dimension of the points.")

        return sum(
            1
            for i in indices
            if a[i] != b[i]
        )
