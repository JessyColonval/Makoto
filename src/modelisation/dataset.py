"""
Written by Jessy Colonval.
"""
from __future__ import annotations
from typing import Tuple, List, Any, Dict

from copy import copy
from random import shuffle
import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from src.modelisation.point import Point


class DatasetIterator():

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__index = 0

    def __next__(self):
        if self.__index < len(self.__dataset):
            result = self.__dataset[self.__index]
            self.__index += 1
            return result
        raise StopIteration


class Dataset():
    """
    This object is used to store contextual and behavioral information of a
    dataset. It is composed of points that represent each piece of data in the
    dataset. Labels cannot be changed after initialization to avoid confusion.
    The order of the labels is important, it allows to describe the information
    of the points.

    Attributes
    ----------
    __lbs_ctx : List[str]
        Contextual data labels.
    __lbs_bhv : List[str]
        Behavioral data labels.
    __pts : list
        Points contained in the dataset.
    """

    def __init__(self, lbs_ctx: List[str], lbs_bhv: List[str],
                 points: List[Point]):
        if not isinstance(lbs_ctx, List):
            raise TypeError("Contextual labels must be stored in a sequence ",
                            "of str.")
        if not isinstance(lbs_bhv, List):
            raise TypeError("Behavioral labels must be stored in a sequence ",
                            "of str.")

        # Copy the sequences to avoid outside modification.
        self.__lbs_ctx = copy(lbs_ctx)
        self.__lbs_bhv = copy(lbs_bhv)

        # Adds points in another list for the same reason.
        self.__pts = []
        for point in points:
            self += point

    def __len__(self):
        return len(self.__pts)

    def __getitem__(self, index: int):
        return self.__pts[index]

    def __add__(self, point: Point):
        if not isinstance(point, Point):
            raise TypeError("Only objects of type Point can be added.")
        if point.number_contextuals() != self.contextual_length:
            raise ValueError("The added point must have the same number of ",
                             "contextual values as those present in the ",
                             "dataset.")
        if point.number_behaviors() != self.behavioral_length:
            raise ValueError("The added point must have the same number of ",
                             "behavioral values as those present in the ",
                             "dataset.")
        self.__pts.append(point)
        return self

    def __iter__(self):
        return DatasetIterator(self)

    def __contains__(self, point: Point) -> bool:
        if not isinstance(point, Point):
            raise TypeError("The searched point must be an object of type ",
                            "Point.")
        return point in self.__pts

    def __str__(self) -> str:
        return "\n".join(
            [f"{i+1:d} - {self[i]:}" for i in range(len(self))]
        )

    def __eq__(self, other: Dataset) -> bool:
        """
        Verifies if this Dataset object is equal to an another one.

        Parameters
        ----------
        other : Dataset
            Another dataset for which we want to verify the equality.

        Return
        ------
        bool
            True if all points contains in this dataset are present in the
            given dataset, false otherwise.
        """
        # Can't be equal if the another dataset isn't an object Dataset.
        if other is None or not isinstance(other, Dataset):
            return False

        # Can't be equal if they don't have the same amount of points or the
        # same contextual and behavioral labels.
        if (len(self.__pts) != len(other)
           or self.__lbs_ctx != other.contextual_labels()
           or self.__lbs_bhv != other.behavioral_labels()):
            return False

        return (
            id(self) == id(other)
            or all(any(p0 == p1 for p1 in other) for p0 in self)
        )

    def __copy__(self) -> Dataset:
        """
        Copy this Dataset.

        Return
        ------
        Dataset
            The copied object.
        """
        return Dataset(
            self.__lbs_ctx,  # Already copied in the constructor.
            self.__lbs_bhv,  # Already copied in the constructor.
            list(self.__pts)  # Different list but same Point objects.
        )

    def __deepcopy__(self, memo) -> Dataset:
        """
        In-depth copying of all components to this Dataset.

        Return
        ------
        Dataset
            The copied object.
        """
        return Dataset(
            self.__lbs_ctx,  # Already copied in the constructor.
            self.__lbs_bhv,  # Already copied in the constructor.
            [p.copy() for p in self]  # List and Point are different.
        )

    def index(self, point: Point) -> int:
        """
        Search the index of a given point in the dataset.

        Parameters
        ----------
        point : Point
            The desired point in the dataset.

        Return
        ------
        int
            The index corresponding to the point provided in parameter.

        Raises
        ------
        TypeError
            when the provided point is not an object Point.
        """
        if not isinstance(point, Point):
            raise TypeError("The searched point must be an object Point.")
        return self.__pts.index(point)

    def contextual_labels(self) -> Tuple[Any]:
        """
        Return
        ------
        Tuple[Any]
            The copy of the contextual label list.
        """
        return copy(self.__lbs_ctx)

    def behavioral_labels(self) -> Tuple[Any]:
        """
        Return
        ------
        Tuple[Any]
            The copy of the behavioral label list.
        """
        return copy(self.__lbs_bhv)

    @property
    def contextual_length(self) -> int:
        """
        Gets the number of contextual values in each point contained.

        Return
        ------
        int
            Number of contextual values.
        """
        return len(self.__lbs_ctx)

    @property
    def behavioral_length(self) -> int:
        """
        Gets the number of behavioral values in each point contained.

        Return
        ------
        int
            Number of behavioral values.
        """
        return len(self.__lbs_bhv)

    def contextuals_index(self, index: int) -> List[float]:
        """
        Gets all contextual values at a given index for every contained points.

        Parameters
        ----------
        index : int
            Index of the wanted contextual value.

        Return
        ------
        List[float]
            Contextual values at the given index.

        Raises
        ------
        TypeError
            when the index isn't an integer.
        ValueError
            when the index is negative or greater or equal to the number of
            contextual values.
        """
        if not isinstance(index, int):
            raise TypeError("The index must be an integer.")
        if index < 0 or index >= self.contextual_length:
            raise ValueError("Out of bound.")
        return [point.contextual(index) for point in self]

    def behaviors_index(self, index: int) -> List[int]:
        """
        Gets all behavioral values at a given index for every contained points.

        Parameters
        ----------
        index : int
            Index of the wanted behavioral values.

        Return
        ------
        List[int]
            Behavioral values at the given index.

        Raises
        ------
        TypeError
            when the index isn't an integer.
        ValueError
            when the index is negative or greater or equal to the number of
            behavioral values.
        """
        if not isinstance(index, int):
            raise TypeError("The index must be an integer.")
        if index < 0 or index >= self.behavioral_length:
            raise ValueError("Out of bound.")
        return [point.behavior(index) for point in self]

    def contextuals_key(self, label: Any) -> List[float]:
        """
        Gets all contextual values at a given label for every contained points.

        Parameters
        ----------
        label : Any
            The label of the desired contextual values.

        Return
        ------
        List[float]
            Contextual values at the given label of each point.
            The values order corresponds to the order of the points in the
            dataset.

        Raises
        ------
        KeyError
            when the label isn't present in this dataset.
        """
        if label not in self.__lbs_ctx:
            raise KeyError("The contextual label isn't present in the ",
                           "dataset.")
        index = self.__lbs_ctx.index(label)
        return [point.contextual(index) for point in self.__pts]

    def behaviors_key(self, label: Any) -> List[int]:
        """
        Gets all behavioral values at a given label for every contained points.

        Parameters
        ----------
        label : Any
            The label of the desired behavioral values.

        Return
        ------
        List[float]
            Behavioral values at the given label of each point.
            The values order corresponds to the order of the points in the
            dataset.

        Raises
        ------
        KeyError
            Returns an exception if the label is not present.
        """
        if label not in self.__lbs_bhv:
            raise KeyError("The contextual label isn't present in the ",
                           "dataset.")
        index = self.__lbs_bhv.index(label)
        return [point.behavior(index) for point in self.__pts]

    def count_behaviors(self) -> Dict[Any, Dict[Any, int]]:
        '''
        Count the number of occurrences of each behavioral value.

        Return
        ------
        Dict[Any, Dict[Any, int]]
            A dictionary containing for each behavioral attribute a another
            dictionary indicating the number of occurrences of each behavioral
            value.
        '''
        result = {}
        for i_bhv, key in enumerate(self.__lbs_bhv):
            result[key] = {}
            for point in self.__pts:
                bhv = point.behavior(i_bhv)
                if bhv in result[key]:
                    result[key][bhv] += 1
                else:
                    result[key][bhv] = 1
        return result

    def group_by(self) -> Dict[Tuple[int], List[Point]]:
        """
        Groups points according to their behavioral values.

        Return
        ------
        Dict[Tuple[int], List[Point]]
            A list of point for each combination of behavioral values.
        """
        result = {}
        for point in self.__pts:
            bhv = tuple(point.behaviors())
            if bhv in result:
                result[bhv].append(point)
            else:
                result[bhv] = [point]
        return result

    def boundaries(self) -> List[Tuple[float, float]]:
        """
        Computes the dataset's boundaries, i.e. the minimal and maximal values
        for each contextual attributes.

        Return
        ------
        List[Tuple[float, float]]
            A list of tuple contains the minimal and maximal value of each
            attributes.
            The attributes order is respected.
        """
        result = []
        for i_ctx in range(0, self.contextual_length):
            p_max = self.__pts[0].contextual(i_ctx)
            p_min = self.__pts[0].contextual(i_ctx)
            for point in self.__pts:
                v_ctx = point.contextual(i_ctx)
                p_max = max(p_max, v_ctx)
                p_min = min(p_min, v_ctx)
            result.append((p_min, p_max))
        return result

    def shuffle_row(self) -> None:
        """
        Shuffle the points order in the dataset.
        """
        shuffle(self.__pts)

    def shuffle_contextuals(self, correspond: List[List[int]]) -> None:
        """
        Shuffles the contextual values order of each point contains in the
        dataset.

        Parameters
        ----------
        correspond : List[List[int]]
            Pairs of indices of contextual values that will be swapped.
        """
        for i, j in correspond:
            self.__lbs_ctx[i], self.__lbs_ctx[j] = (self.__lbs_ctx[j],
                                                    self.__lbs_ctx[i])
        for point in self:
            point.shuffle_contextuals(correspond)

    def shuffle_behaviors(self, correspond: List[List[int]]) -> None:
        """
        Shuffles the behavioral values order of each point contains in the
        dataset.

        Parameters
        ----------
        correspond : List[List[int]]
            Pairs of indices of behavioral values that will be swapped.
        """
        for i, j in correspond:
            self.__lbs_bhv[i], self.__lbs_bhv[j] = (self.__lbs_bhv[j],
                                                    self.__lbs_bhv[i])
        for point in self:
            point.shuffle_behaviors(correspond)

    def change_behaviors(self, i_bhv: int, old_bhv: int, new_hbv: int) -> None:
        """
        Changes for every point with a given behavioral value to an another
        one.

        Parameters
        ----------
        i_bhv : int
            Index of the behavioral attributes aimed.
        old_bhv: int
            Value of the behavioral value that will be changed.
        new_bhv: int
            The new behavioral value the will replace the old one.
        """
        for point in self:
            if point.behavior(i_bhv) == old_bhv:
                point.set_behavior(i_bhv, new_hbv)

    def swap_behaviors(self, i_bhv: int, first: int, second: int) -> None:
        """
        Swaps two already present behavioral values.

        Parameters
        ----------
        i_bhv : int
            Index of the behavioral attributes aimed.
        first : int
            The first behavioral value that will be swapped with the second
            one.
        second : int
            The second behavioral value that will be swapped with the first
            one.
        """
        for point in self:
            if point.behavior(i_bhv) == first:
                point.set_behavior(i_bhv, second)
            elif point.behavior(i_bhv) == second:
                point.set_behavior(i_bhv, first)

    def add_constant(self, i_ctx: int, const: float) -> None:
        """
        Adds a constant to every points at a given contextual index.

        Parameters
        ----------
        i_ctx : int
            Index of the contextual attributes aimed.
        const : float
            The constant added.
        """
        for point in self:
            point.set_contextual(i_ctx, const)

    def to_dict(self) -> Dict[int, Dict[Any, float]]:
        """
        Converts this dataset into a python's dictionnary.

        Return
        ------
        Dict[int, Dict[Any, float]]
            A dictionnary that link for every point index an another
            dictionnary that link every contextual and behavioral labels to
            its corresponding value.
        """
        return {
            i: dict(
                {
                    self.__lbs_ctx[j]: self.__pts[i].contextual(j)
                    for j in range(0, len(self.__lbs_ctx))
                },
                **{
                    self.__lbs_bhv[j]: self.__pts[i].behavior(j)
                    for j in range(0, len(self.__lbs_bhv))
                }
                )
            for i in range(0, len(self.__pts))
        }

    def to_dataframe(self) -> DataFrame:
        """
        Converts this dataset into a DataFrame object.

        Return
        ------
        DataFrame
            A DataFrame that contains every contextual and behavioral values
            and their labels.
        """
        data = dict({
            self.__lbs_ctx[i_ctx]: np.array(
                [
                    self.__pts[i_pts].contextual(i_ctx)
                    for i_pts in range(0, len(self))
                ],
                dtype=np.float64)
            for i_ctx in range(0, self.contextual_length)
            },
            **{
            self.__lbs_bhv[i_bhv]: np.array(
                [
                    self.__pts[i_pts].behavior(i_bhv)
                    for i_pts in range(0, len(self))
                ],
                dtype=np.int16)
            for i_bhv in range(0, self.behavioral_length)
            }
        )
        columns = self.__lbs_ctx + self.__lbs_bhv
        result = DataFrame(data, columns=columns)
        return result

    def to_numpy(self, method: str = "rows") -> npt.NDArray[np.float64]:
        """
        Converts this dataset into a numpy array.

        Parameters
        ----------
        method : str
            Determines how the datas will be represented in the array.
            - "cols": one array for each contextual and behavioral attributes;
            - "rows": each array contains every contextual and behavioral
            values of each point.
            By default the method used is "rows".

        Return
        ------
        NDArray[float64]
            A numpy array that contains all values of this dataset according to
            the method selected.

        Raises
        ------
        ValueError
            when the method isn't equal to 'cols' or 'rows'.
        """
        if method == "rows":
            n_attr = len(self.__lbs_ctx) + len(self.__lbs_bhv)
            result = np.empty([len(self.__pts), n_attr], dtype=np.float64)
            for i, p in enumerate(self.__pts):
                result[i] = p.to_numpy()
            return result

        if method == "cols":
            result = []
            for i_ctx in range(0, len(self.__lbs_ctx)):
                result.append(
                    np.array(
                        [point.contextual(i_ctx) for point in self.__pts],
                        dtype=np.float64)
                )
            for i_bhv in range(0, len(self.__lbs_bhv)):
                result.append(
                    np.array(
                        [point.behavior(i_bhv) for point in self.__pts],
                        dtype=np.int16
                    )
                )
            return np.array(result)

        raise ValueError("Unexpected method.")

    @staticmethod
    def from_dict(data: Dict[Any, Dict[Any, Any]], lbs_ctx: Tuple[Any],
                  lbs_bhv: Tuple[Any]) -> Dataset:
        """
        Parameters
        ----------
        data : Dict[Any, Dict[Any, Any]]
            A dictionary, with n line and m column, in the form
            :math:`\\{ row_{i} : \\{ col_{j} : any | j \\in [1, m] \\} |
            i \\in [1, n] \\}`
            which will be converted into Dataset.
        lbs_ctx : Tuple[Any]
            Labels, contained in the dictionary, corresponding to the
            contextual data.
        lbs_bhv : Tuple[Any]
            Labels, contained in the dictionary, corresponding to the
            behavioral data.

        Return
        ------
        dataset : Dataset
            A Dataset containing n points that represnets the contextual and
            behavioral data present in the original dictionary.
        """
        # Type error.
        if not isinstance(data, dict):
            raise TypeError("Data to convert must be a \"dict\".")

        # Initializes the dataset witout any points.
        result = Dataset(lbs_ctx, lbs_bhv, [])

        # Iterates rows in the dictionnary.
        for row in data.values():
            # Verifies that all labels are present in the dictionary.
            if (any(lb not in row for lb in lbs_ctx)
               or any(lb not in row for lb in lbs_bhv)
               or any(
                   lb not in lbs_ctx and lb not in lbs_bhv
                   for lb in row.keys()
                   )):
                raise ValueError("Inconsistency in the contextual labels.")

            # Builds the contextual and behavioral values.
            ctxs = np.array([row[lb] for lb in lbs_ctx], dtype=np.float64)
            bhvs = np.array([row[lb] for lb in lbs_bhv], dtype=np.int16)

            # Adds the point.
            result += Point(ctxs, bhvs)

        return result

    @staticmethod
    def from_dataframe(df: DataFrame, lbs_ctx: Tuple[Any],
                       lbs_bhv: Tuple[Any]) -> Dataset:
        """
        Parameters
        ----------
        df : DataFrame
            A DataFrame object with labels for its columns that will be
            converted into a Dataset object.
        lbs_ctx : Tuple[Any]
            The list of labels, contained in the DataFrame, corresponding to
            the contextual data.
        lbs_bhv : Tuple[Any]
            The list of labels, contained in the DataFrame, corresponding to
            the behavioral data.

        Return
        ------
        Dataset
            A Dataset object has as many points as the DataFrame has lines.
            These points contain the contextual and behavioral data present
            in the original DataFrame..
        """
        if not isinstance(df, DataFrame):
            raise TypeError("The data to convert must be an object of type ",
                            "DataFrame.")

        # Converts the DataFrame into a dictionnary and converts this latter
        # into a Dataset.
        return Dataset.from_dict(df.to_dict("index"), lbs_ctx, lbs_bhv)
