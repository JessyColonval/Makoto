"""
Written by Jessy Colonval.
"""
from __future__ import annotations

from random import shuffle
import numpy as np
import numpy.typing as npt
from src.mathematics.method_distance import Euclidean, Manhattan, Absolute


class PointIterator():
    """
    Iterator useful to iterate contextual and behavioral values of a point.
    """

    def __init__(self, point: Point):
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
        """
        Copy the point into a new Point object.

        Return
        ------
        Point
            A copy of this point.
        """
        return Point(
            self.__ctxs,  # Already copied in the constructor.
            self.__bhvs
        )

    def contextual(self, index: int) -> float:
        """
        Gets the contextual value at a given index.

        Parameters
        ----------
        index : int
            The index of the wanted contextual value.

        Return
        ------
        float
            The contextual value at the given index.

        Raises
        ------
        IndexError
            when the index is negative or greater or equal to the number of
            contextual values.
        """
        if index < 0 or index >= len(self.__ctxs):
            raise IndexError("Out of range.")
        return self.__ctxs[index]

    def contextuals(self) -> npt.NDArray[np.float64]:
        """
        Gets a copy of the array that contains all the contextual values.

        Return
        ------
        NDArray[float64]
            a copy of the numpy that contains all contextual values.
        """
        return self.__ctxs.copy()

    def number_contextuals(self) -> int:
        """
        Gets the number of contextual values.

        Return
        ------
        int
            The number of contextual values.
        """
        return len(self.__ctxs)

    def set_contextual(self, i_ctx: int, value: np.float64) -> None:
        """
        Sets the contextual value at the given index with the given value.

        Parameters
        ----------
        i_ctx : int
            The index of the contextual value that will be modify.
        value: float64
            Its new contextual value.
        """
        self.__ctxs[i_ctx] = value

    def behavior(self, index: int) -> int:
        """
        Gets the behavioral value at a given index.

        Parameters
        ----------
        index : int
            The index of the wanted behavioral value.

        Return
        ------
        float
            The behavioral value at the given index.

        Raises
        ------
        IndexError
            when the index is negative or greater or equal to the number of
            contextual values.
        """
        if index < 0 or index >= len(self.__bhvs):
            raise IndexError("Out of range.")
        return self.__bhvs[index]

    def behaviors(self) -> npt.NDArray[np.float64]:
        """
        Gets a copy of the array that contains all the behavioral values.

        Return
        ------
        NDArray[float64]
            a copy of the numpy that contains all contextual values.
        """
        return self.__bhvs.copy()

    def number_behaviors(self) -> int:
        """
        Gets the number of behavioral values.

        Return
        ------
        int
            The number of behavioral values.
        """
        return len(self.__bhvs)

    def set_behavior(self, i_bhv: int, value: np.int16) -> None:
        """
        Sets the behavioral value at the given index with the given value.

        Parameters
        ----------
        i_bhv : int
            The index of the behavioral value that will be modify.
        value: int16
            Its new behavioral value.
        """
        self.__bhvs[i_bhv] = value

    def distance(self, other: Point, method: str = "euclidean") -> float:
        """
        Computes the distance between two points.

        Parameters
        ----------
        other : Point
            An another point where its distance with this point will be
            calculated.
        method : str, optional
            The method of distance used between 'euclidean', 'manhattan' and
            'absolute'.
            Default is 'euclidean'.

        Return
        ------
        float
            The distance between these two points according the method used.

        Raises
        ------
        ValueError
            when an unknow method of distance is given.
        """
        if method == "euclidean":
            return Euclidean.distance(self.__ctxs, other.contextuals())
        if method == "manhattan":
            return Manhattan.distance(self.__ctxs, other.contextuals())
        if method == "absolute":
            return Absolute.distance(self.__ctxs, other.contextuals())
        raise ValueError("Unknow method of distance computation.")

    def shuffle_contextuals(self, indices: npt.NDArray[np.int16] = None
                            ) -> None:
        """
        Shuffles the contextual values.

        Parameters
        ----------
        indices : NDArray[int16], optional
            The pairs of indices to which each value corresponds will be
            interchanged.

        Raises
        ------
        ValueError
            when one of the indices are negative or greater or equal to the
            number of contextual values.
        """
        # Without indices, all contextual values are randomly shuffle.
        if indices is None:
            shuffle(self.__ctxs)

        else:
            # Checks if all indices are between 0 and the contextuals' size.
            if any(i < 0 or j < 0
                   or i >= len(self.__ctxs)
                   or j >= len(self.__ctxs)
                   for i, j in indices):
                raise ValueError("Indices must be between 0 and the number ",
                                 "of contextual attributes.")

            # Swaps values between each indices.
            for i, j in indices:
                self.__ctxs[i], self.__ctxs[j] = self.__ctxs[j], self.__ctxs[i]

    def shuffle_behaviors(self, indices: npt.NDArray[np.int16] = None
                          ) -> None:
        """
        Shuffles the behavioral values.

        Parameters
        ----------
        indices : NDArray[int16], optional
            The pairs of indices to which each value corresponds will be
            interchanged.

        Raises
        ------
        ValueError
            when one of the indices are negative or greater or equal to the
            number of behavioral values.
        """
        # Without indices, all behavioral values are randomly shuffle.
        if indices is None:
            shuffle(self.__bhvs)

        # Checks if all indices are between 0 and the contextuals' size.
        else:
            if any(i < 0 or j < 0
                   or i >= len(self.__bhvs)
                   or j >= len(self.__bhvs)
                   for i, j in indices):
                raise ValueError("Indices must be between 0 and the number ",
                                 "of behavioral attributes.")

        # Swaps values between each indices.
            for i, j in indices:
                self.__bhvs[i], self.__bhvs[j] = self.__bhvs[j], self.__bhvs[i]

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """
        Converts the point into a single numpy array.

        Return
        ------
        NDArray[float64]
            The contextual and behavioral values of its point.
        """
        return np.concatenate((self.__ctxs, self.__bhvs))
