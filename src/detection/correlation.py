"""
Written by Jessy Colonval.
"""
from abc import abstractmethod
from typing import List

from math import sqrt

from numba import njit, float64, int64
import numpy as np
import numpy.typing as npt

from minepy import MINE
from pandas import DataFrame

from src.modelisation.dataset import Dataset


class Correlation:
    """
    Mother class of correlation calculation approaches, grouping static
    functions common to all approaches.
    """

    @staticmethod
    @abstractmethod
    def matrix_from_numpy(
        data: npt.NDArray[npt.NDArray[np.float64]]
    ) -> npt.NDArray[npt.NDArray[np.float64]]:  # pragma: no cover
        """
        Abstract method for calculating the correlation matrix of a data set
        stored in a numpy array.
        """

    @staticmethod
    @abstractmethod
    def matrix(
        data: object
    ) -> npt.NDArray[npt.NDArray[np.float64]]:  # pragma: no cover
        """
        Abstract method for calculating the correlation matrix of a data set
        stored in a 'DataFrame' or a 'Dataset'.
        """

    @staticmethod
    def best_correlation(feat: str, matrix: dict) -> str:
        """
        Gives the name of the attribute with the highest correlation with the
        attribute given as a parameter.

        Parameters
        ----------
        feat : str
            The attribute label.
        matrix : dict
            The correlation matrix.

        Return
        ------
        str
            Label of the best correlated attribute.

        Raises
        ------
        ValueError
            when the label given in the parament isn't present in the matrix.
        """
        if feat not in matrix:
            raise ValueError("The attribute is not present in the matrix.")
        return max(
            filter(lambda e: e != feat, matrix[feat]),
            key=matrix[feat].get
        )

    @staticmethod
    def best_pairwise(matrix: npt.NDArray[npt.NDArray[np.float64]]) -> list:
        """
        Gives the pair of attributes that are most strongly correlated with
        each other.

        Parameters
        ----------
        matrix : NDArray[NDArray[float64]]
            The correlation matrix.

        Return
        ------
        List[str]
            The pair of best-correlated attribute labels.
        """
        # Gets attributes' label in the matrix.
        features = list(matrix.keys())

        # The starting pair.
        pair = [features[0], features[1]]

        # Go through all the combinations looking for the best pair.
        for i in range(0, len(features)-1):
            f1 = features[i]
            for j in range(i+1, len(features)):
                f2 = features[j]
                if abs(matrix[f1][f2]) > abs(matrix[pair[0]][pair[1]]):
                    pair[0] = f1
                    pair[1] = f2

        return pair

    @staticmethod
    def best_correlated_pairs(matrix: npt.NDArray[npt.NDArray[np.float64]],
                              n_rank: int = 10) -> list:
        """
        Parameters
        ----------
        matrix : NDArray[NDArray[float64]]
            The correlation matrix.
        n_rank : int
            The ranking size.
        """
        if n_rank < 0:
            raise ValueError("The size of the requested ranking must be stric",
                             "tly positive.")
        if n_rank > len(matrix):
            raise ValueError("The size of the requested ranking must not exce",
                             "ed the number of contextual attributes.")

        # The list of contextual attributes.
        features = list(matrix.keys())

        # Initializes the ranking with the first n different attribute
        # combinations.
        ranking = []
        i = 0
        j = 0
        while len(ranking) < n_rank:
            f1 = features[i]
            f2 = features[j]

            if f1 != f2:
                ranking.append([f1, f2])

            if j == len(features):
                i += 1
                j = 0
            else:
                j += 1

        for i1, f1 in enumerate(features):
            for i2 in range(i1+1, len(features)):
                f2 = features[i2]
                if f1 != f2:
                    min_pair = ranking[0]
                    for i in range(1, len(ranking)):
                        current_pair = ranking[i]
                        if (
                            abs(matrix[min_pair[0]][min_pair[1]])
                            > abs(matrix[current_pair[0]][current_pair[1]])
                        ):
                            min_pair = current_pair
                    if (
                        abs(matrix[f1][f2])
                        > abs(matrix[min_pair[0]][min_pair[1]])
                    ):
                        min_pair[0] = f1
                        min_pair[1] = f2

        return ranking

    @staticmethod
    def transitive(
        a: str, b: str, c: str,
        matrix: npt.NDArray[npt.NDArray[np.float64]]
    ) -> float:
        """
        Calculates a transitivity score of three contextual attributes.

        Parameters
        ----------
        a : str
            The first contextual attribute.
        b : str
            The second contextual attribute.
        c : str
            The third contextual attribute.
        matrix : NDArray[NDArray[float64]]
            The correlation matrix.

        Return
        ------
        float
            The transitivity score.
        """
        return matrix[a][b]**2 + matrix[b][c]**2

    @staticmethod
    def is_transitive(
        a: str, b: str, c: str,
        matrix: npt.NDArray[npt.NDArray[np.float64]]
    ) -> bool:
        """
        Verifies if three contextual attributes are transitive between each
        other.
        They are if the transitivity score of A, B, C and B, A, C are greater
        or equal than 1.0.

        Parameters
        ----------
        a : str
            The first contextual attribute.
        b : str
            The second contextual attribute.
        c : str
            The third contextual attribute.
        matrix : NDArray[NDArray[float64]]
            The correlation matrix

        Return
        ------
        bool
            True if these three contextual attributes are transitive.
        """
        return (
            Correlation.transitive(a, b, c, matrix) >= 1.0
            or Correlation.transitive(b, a, c, matrix) >= 1.0
        )

    @staticmethod
    def at_least_one_transitive(
        a: str, b: str,
        matrix: npt.NDArray[npt.NDArray[np.float64]]
    ) -> bool:
        """
        Verifies if at least one other contextual attribute is transitive with
        two given attributes.
        They are if the transitivity score of A, B, C and B, A, C are greater
        or equal than 1.0.

        Parameters
        ----------
        a : str
            The first contextual attribute.
        b : str
            The second contextual attribute.
        matrix : NDArray[NDArray[float64]]
            The correlation matrix.

        Return
        ------
        bool
            True if at least one contextual attribute (different to A and B)
            is transitive with A and B.
        """
        return any(
            c not in (a, b) and Correlation.is_transitive(a, b, c, matrix)
            for c in matrix.keys()
        )

    @staticmethod
    def is_all_transitive(
        c: str, features: List[str],
        matrix: npt.NDArray[npt.NDArray[np.float64]]
    ) -> bool:
        """
        Verifies if all attribute combinations are transitive with a given
        attribute C.

        Parameters
        ----------
        c : str
            The third contextual attribute.
        features : List[str]
            The list of contextual attributes that will be verified to see if
            they are transitive or not with C.
        matrix : NDArray[NDArray[float64]]
            The correlation matrix.

        Return
        ------
        bool
            True all attribute combinations are transitive with C.
        """
        for i1 in range(0, len(features)-1):
            a = features[i1]
            for i2 in range(i1+1, len(features)):
                b = features[i2]
                if (
                    not Correlation.is_transitive(a, b, c, matrix)
                    or not Correlation.is_transitive(b, a, c, matrix)
                ):
                    return False
        return True

    @staticmethod
    def sum_transitive(
        c: str, attributes: List[str],
        matrix: npt.NDArray[npt.NDArray[np.float64]]
    ) -> float:
        """
        Calculates the sum of the transitivity scores of all combinations of
        the given list with attribute C.

        Parameters
        ----------
        c : str
            The third contextual attribute.
        attributes : List[str]
            The list of contextual attributes that will be used to calculate
            the sum of the transitivity score with C.
        matrix : NDArray[NDArray[float64]]
            The correlations matrix.

        Return
        ------
        float
            The transitivity sum of all contextual attribute combinaitions.
        """
        result = 0.0
        for i1 in range(0, len(attributes)-1):
            a = attributes[i1]
            for i2 in range(i1+1, len(attributes)):
                b = attributes[i2]
                result += Correlation.transitive(a, b, c, matrix)
                result += Correlation.transitive(b, a, c, matrix)
        return result


class Pearson(Correlation):
    """
    Object that contains static functions for calculating a Pearson correlation
    matrix of a dataset according to how it is stored.
    """

    @staticmethod
    @njit(float64[:, :](float64[:, :]), fastmath=True)
    def matrix_from_numpy(
        data: npt.NDArray[npt.NDArray[np.float64]]
    ) -> npt.NDArray[npt.NDArray[np.float64]]:
        """
        Calculates the Pearson correlation matrix of a dataset stored in a
        numpy array.

        Parameters
        ----------
        data : object
            The dataset.

        Return
        ------
        NDArray[NDArray[float64]]
            The correlation matrix

        Raises
        ------
        TypeError
            when the dataset isn't stored in a numpy array.
        """
        n_c = len(data[0])

        # Initialization of the correlation matrix.
        result = np.zeros((n_c, n_c), dtype=np.float64)
        for i in range(0, n_c):
            result[i][i] = 1.0

        # Initializes the lists of means, previous means and M2 for each
        # contextual attribute.
        means = np.zeros(n_c, dtype=np.float64)
        m_2 = np.zeros(n_c, dtype=np.float64)

        # Computes the means, M2 and C2S according to the equations in
        # [SMKN17].
        for n in range(0, len(data)):
            point = data[n]

            # Computes the current average and M2 for each contextual
            # attribute.
            for i in range(0, n_c):
                x_n = point[i]
                diff = x_n - means[i]
                means[i] += diff / (n + 1)
                m_2[i] += (diff * (x_n - means[i]))

        # Compute C2S for all pairs of contextual attributes.
        for n in range(0, len(data)):
            point = data[n]
            for i0 in range(0, n_c - 1):
                dx = point[i0] - means[i0]
                for i1 in range(i0 + 1, n_c):
                    dy = point[i1] - means[i1]
                    result[i0][i1] += (dx * dy)
                    result[i1][i0] = result[i0][i1]

        # Computes the correlation coefficients for all pairs of contextual
        # attributes from the previously computed M2 and C2S.
        for i0 in range(0, n_c - 1):
            m_2_x = m_2[i0]
            for i1 in range(i0 + 1, n_c):
                m_2_y = m_2[i1]
                deno = (sqrt(m_2_x) * sqrt(m_2_y))
                if deno != 0.0:
                    result[i0][i1] /= deno
                result[i1][i0] = result[i0][i1]

        return result

    @staticmethod
    def matrix(data: object) -> npt.NDArray[npt.NDArray[np.float64]]:
        """
        Calculates the Pearson correlation matrix of a dataset stored in a
        numpy array, DataFrame, or Dataset.

        Parameters
        ----------
        data : object
            The dataset.

        Return
        ------
        NDArray[NDArray[float64]]
            The correlation matrix

        Raises
        ------
        TypeError
            when the dataset isn't stored in a numpy array, a DataFrame or a
            Dataset.
        """
        if isinstance(data, (DataFrame, Dataset)):
            return Pearson.matrix_from_numpy(data.to_numpy())

        if isinstance(data, np.ndarray):
            return Pearson.matrix_from_numpy(data)

        raise TypeError("Unexpected object, please provide a 'DataFrame', 'Da",
                        "taset', or a numpy array.")


class Spearman(Correlation):
    """
    Object that contains static functions for calculating a Spearman
    correlation matrix of a dataset according to how it is stored.
    """

    @staticmethod
    @njit(float64[:, :](float64[:, :]), fastmath=True)
    def __rankdata(
        data: npt.NDArray[npt.NDArray[np.float64]]
    ) -> npt.NDArray[npt.NDArray[np.float64]]:
        # Numbers of lines and columns.
        n_l = len(data)
        n_c = len(data[0])

        # Initializes the result matrix.
        result = np.empty((n_l, n_c), dtype=np.float64)

        for i_c in range(0, n_c):
            arr = np.ravel(data[:, i_c])

            sorter = np.argsort(arr, kind="quicksort")

            inv = np.empty(sorter.size, dtype=np.intp)
            inv[sorter] = np.arange(sorter.size, dtype=np.intp)

            arr = arr[sorter]
            obs = np.ones(len(arr), dtype=np.bool8)
            for i in range(0, len(arr)-1):
                obs[i+1] = arr[i] != arr[i+1]
            dense = obs.cumsum()[inv]

            tmp = np.nonzero(obs)[0]

            count = np.zeros(len(tmp)+1, dtype=np.int64)
            for i in range(0, len(tmp)):
                count[i] = tmp[i]
            count[len(count)-1] = len(obs)

            result[:, i_c] = .5 * (count[dense] + count[dense - 1] + 1)

        return result

    @staticmethod
    def matrix_from_numpy(
        data: npt.NDArray[npt.NDArray[np.float64]]
    ) -> npt.NDArray[npt.NDArray[np.float64]]:
        """
        Calculates the Spearman correlation matrix of a dataset stored in a
        numpy array.

        Parameters
        ----------
        data : object
            The dataset.

        Return
        ------
        NDArray[NDArray[float64]]
            The correlation matrix

        Raises
        ------
        TypeError
            when the dataset isn't stored in a numpy array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Unexpected object, please provide a numpy array.")
        ranks = Spearman.__rankdata(data)
        return Pearson.matrix_from_numpy(ranks)

    @staticmethod
    def matrix(data: object) -> npt.NDArray[npt.NDArray[np.float64]]:
        """
        Calculates the Spearman correlation matrix of a dataset stored in a
        numpy array, DataFrame, or Dataset.

        Parameters
        ----------
        data : object
            The dataset.

        Return
        ------
        NDArray[NDArray[float64]]
            The correlation matrix

        Raises
        ------
        TypeError
            when the dataset isn't stored in a numpy array, a DataFrame or a
            Dataset.
        """
        if isinstance(data, (DataFrame, Dataset)):
            return Spearman.matrix_from_numpy(data.to_numpy())

        if isinstance(data, np.ndarray):
            return Spearman.matrix_from_numpy(data)

        raise TypeError("Unexpected object, please provide a 'DataFrame', 'Da",
                        "taset', or a numpy array.")


class MIC(Correlation):
    """
    Object that contains static functions for calculating a MIC correlation
    matrix of a dataset according to how it is stored.
    """

    @staticmethod
    @njit(float64(int64), fastmath=True)
    def estimator_alpha(n: int) -> float:
        """
        Gives the optimal alpha value for calculating an MIC correlation based
        on the number of points in the dataset that will be used.

        Parameters
        ----------
        n : int
            The number of points in the dataset.

        Return
        ------
        float
            The optimal alpha value.

        Raises
        ------
        ValueError
            when the number of points is negative.
        """
        if n <= 0:
            raise ValueError("The number provided must be strictly positive.")
        if n < 25:
            return 0.85
        if 25 <= n < 50:
            return 0.80
        if 50 <= n < 250:
            return 0.75
        if 250 <= n < 500:
            return 0.70
        if 500 <= n < 1000:
            return 0.65
        if 1000 <= n < 2500:
            return 0.60
        if 2500 <= n < 5000:
            return 0.55
        if 5000 <= n < 10000:
            return 0.50
        if 10000 <= n < 40000:
            return 0.45
        return 0.40

    @staticmethod
    def matrix_from_numpy(
        data: npt.NDArray[npt.NDArray[np.float64]],
        alpha: float,
        c: float
    ) -> npt.NDArray[npt.NDArray[np.float64]]:
        """
        Calculates the MIC correlation matrix of a dataset stored in a
        numpy array.

        Parameters
        ----------
        data : object
            The dataset.

        Return
        ------
        NDArray[NDArray[float64]]
            The correlation matrix

        Raises
        ------
        TypeError
            when the dataset isn't stored in a numpy array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Unexpected object, please provide a numpy array.")

        # Number of columns.
        n_c = len(data[0])

        # Initialization of the correlation matrix.
        result = np.zeros((n_c, n_c), dtype=np.float64)
        for i in range(0, n_c):
            result[i][i] = 1.0

        # Calculates the MIC correlation scores for all pairs.
        mine = MINE(alpha=alpha, c=c)
        for i1 in range(0, n_c-1):
            x = data[:, i1]
            for i2 in range(i1+1, n_c):
                if i1 != i2:
                    y = data[:, i2]
                    mine.compute_score(x, y)
                    result[i1][i2] = result[i2][i1] = mine.mic()

        return result

    @staticmethod
    def matrix(
        data: object,
        alpha: float = None,
        c: float = 15.0
    ) -> npt.NDArray[npt.NDArray[np.float64]]:
        """
        Calculates the MIC correlation matrix of a dataset stored in a
        numpy array, DataFrame, or Dataset.

        Parameters
        ----------
        data : object
            The dataset.

        Return
        ------
        NDArray[NDArray[float64]]
            The correlation matrix

        Raises
        ------
        TypeError
            when the dataset isn't stored in a numpy array, a DataFrame or a
            Dataset.
        """
        # If the alpha number is not given then the best one is determined
        # according to the number of elements.
        if alpha is None:
            alpha = MIC.estimator_alpha(len(data))

        if isinstance(data, (DataFrame, Dataset)):
            return MIC.matrix_from_numpy(data.to_numpy(), alpha, c)

        if isinstance(data, np.ndarray):
            return MIC.matrix_from_numpy(data, alpha, c)

        raise TypeError("Unexpected object, please provide a 'DataFrame', 'Da",
                        "taset', or a numpy array.")
