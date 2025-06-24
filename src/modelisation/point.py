"""
Written by Jessy Colonval.
"""
from __future__ import annotations

from math import sqrt
from random import shuffle
import numpy as np
import numpy.typing as npt


class PointIterator():
    """
    Iterator useful to iterate contextual values of a point.
    """

    def __init__(self, point):
        self.__point = point
        self.__index = 0

    def __next__(self):
        if self.__index < len(self.__point):
            result = self.__point[self.__index]
            self.__index += 1
            return result
        raise StopIteration


class Point():
    """
    A point object defining two types of data:
        - Behavioral attributes is an attribute of interest measured for each
        object. An object can have more than one. These attributes are often
        non-spatial because it measures a certain quantity. For example, the
        type of glass, the presence of a heart abnormality or the description
        of an image.

        - Contextual attributes is an attribute expressing the characteristics
        of an object. An object often has several of them. These attributes are
        often spatial and express coordinates. For example, the composition of
        a glass pane, the number of heartbeats per minute or the color of a
        pixel.

    Attributes
    ----------
    __ctxs : tuple
        The list of contextual data.
    __bhvs : tuple
        The list of behavioral data.

    Methods
    -------
    contextuals()
        Returns the contextual data of the point.
    behaviors()
        Returns the behavioral data of the point.
    is_equal(other)
        Verifies if the two points are identical.
    distance(other, method = Euclidean)
        Calculates the distance to another point on contextual data.
    """

    def __init__(self, contextuals: npt.NDArray[np.float64],
                 behaviors: npt.NDArray[np.int16]):
        """
        Parameters
        ----------
        contextuals : NDArray[float64]
            List of the contextual data of the point.
        behaviors : NDArray[int16]
            List of the contextual data of the point.

        Raises
        ------
        ValueError
            when the array of contextual values is empty.
        ValueError
            when the array of behavioral values is empty.
        """
        if len(contextuals) == 0:
            raise ValueError("The point must contains contextual values.")
        if len(behaviors) == 0:
            raise ValueError("The point must contains behavioral values.")
        self.__ctxs = contextuals.copy()
        self.__bhvs = behaviors.copy()

    def __len__(self) -> int:
        """
        Return
        ------
        int
            The amount of contextual data.
        """
        return len(self.__ctxs) + len(self.__bhvs)

    def __getitem__(self, index: int) -> None:
        if index >= (len(self.__ctxs) + len(self.__bhvs)):
            raise IndexError()
        if index < len(self.__ctxs):
            return self.__ctxs[index]
        return self.__bhvs[index - len(self.__ctxs)]

    def __str__(self) -> str:
        """
        Return
        ------
        str
            The string containing the information of the point.
        """
        return "".join([
            "[",
            ", ".join([f"{v_ctx:f}" for v_ctx in self.__ctxs]),
            "] - [",
            ", ".join([f"{v_bhv:d}" for v_bhv in self.__bhvs]),
            "]"
        ])

    def __eq__(self, other: Point) -> bool:
        """
        Parameters
        ----------
        other : Point
            Another point we want to compare to our point.

        Return
        ------
        bool
            True if the contextual and behavioral data of the points are
            identical, false otherwise.
        """
        if (other is None or len(self) != len(other)):
            return False
        for i, val in enumerate(self):
            if val != other[i]:
                return False
        return True

    def copy(self) -> Point:
        return Point(
            np.copy(self.__ctxs),
            np.copy(self.__bhvs)
        )

    def contextual(self, index: int) -> float:
        if index < 0 or index >= len(self.__ctxs):
            raise IndexError("Out of range.")
        return self.__ctxs[index]

    def contextuals(self):
        return self.__ctxs.copy()

    def number_contextuals(self) -> int:
        return len(self.__ctxs)

    def set_contextual(self, i_ctx, const) -> None:
        self.__ctxs[i_ctx] = const

    def behavior(self, index: int) -> int:
        if index < 0 or index >= len(self.__bhvs):
            raise IndexError("Out of range.")
        return self.__bhvs[index]

    def behaviors(self):
        return self.__bhvs.copy()

    def number_behaviors(self) -> int:
        return len(self.__bhvs)

    def distance(self, other: Point) -> float:
        if self.number_contextuals() != other.number_contextuals():
            raise ValueError("Points must have same dimension.")
        _sum = 0.0
        for i in range(0, self.number_contextuals()):
            _sum += (self.contextual(i) - other.contextual(i))**2
        return sqrt(_sum)

    def shuffle_contextuals(self, indices: np.ndarray = None) -> None:
        if indices is None:
            shuffle(self.__ctxs)
        else:
            if any(i < 0 or j < 0
                   or i >= len(self.__ctxs)
                   or j >= len(self.__ctxs)
                   for i, j in indices):
                raise ValueError("Indices must be between 0 and the number ",
                                 "of contextual attributes.")

            for i, j in indices:
                self.__ctxs[i], self.__ctxs[j] = self.__ctxs[j], self.__ctxs[i]

    def shuffle_behaviors(self, indices: np.ndarray = None) -> None:
        if indices is None:
            shuffle(self.__bhvs)
        else:
            if any(i < 0 or j < 0
                   or i >= len(self.__bhvs)
                   or j >= len(self.__bhvs)
                   for i, j in indices):
                raise ValueError("Indices must be between 0 and the number ",
                                 "of behavioral attributes.")

            for i, j in indices:
                self.__bhvs[i], self.__bhvs[j] = self.__bhvs[j], self.__bhvs[i]

    def change_behaviors(self, i_bhv, new_bhv) -> None:
        self.__bhvs[i_bhv] = new_bhv

    def to_numpy(self):
        return np.concatenate((self.__ctxs, self.__bhvs))
