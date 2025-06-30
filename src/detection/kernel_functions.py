from sys import maxsize
from abc import abstractmethod
from math import sqrt, exp, cos, pi, sin

import numba as nb


class KernelFunction():
    """
    Mother class of kernel functions' object.
    """

    @staticmethod
    @abstractmethod
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Abstract method to compute the weight of a distance.

        Parameters
        ----------
        distance : float
            A distance between two objects.
        """


class Rectangular(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the rectangular method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the rectangular weight.

        Parameters
        ----------
        distance : float
            A distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{1}{2}  & \\text{if }
            |d| <= 1 \\ 0 & \\text{otherwise} \\end{array} \\right .`.

        """
        if distance < -1 or distance > 1:
            return 0
        return 1/2


class Triangular(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the triangular method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the triangular weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} 1 - |d|  & \\text{if } |d| <=
            1 \\ 0 & \\text{otherwise} \\end{array} \\right .`.

        """
        if distance < -1 or distance > 1:
            return 0
        return 1 - abs(distance)


class Epanechnikov(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the epanechnikov method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the epanechnikov weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{3}{4} (1 - d^{2})  &
            \\text{if } |d| <= 1 \\ 0 & \\text{otherwise} \\end{array}
            \\right .`.

        """
        if distance < -1 or distance > 1:
            return 0
        return (3/4) * (1 - distance**2)


class BiWeight(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the bi-weight method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the bi-weight weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{15}{16} (1 -
            d^{2})^{2}  & \\text{if } |d| <= 1 \\ 0 & \\text{otherwise}
            \\end{array} \\right .`.

        """
        if distance < -1 or distance > 1:
            return 0
        return (15/16) * (1 - distance**2)**2


class TriWeight(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the tri-weight method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the tri-weight weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{35}{32} (1 -
            d^{2})^{3}  & \\text{if } |d| <= 1 \\ 0 & \\text{otherwise}
            \\end{array} \\right .`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (35/32) * (1 - distance**2)**3


class TriCube(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the tri-cube method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Compute the tri-cube weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{70}{81} (1 -
            |d^{3}|)^{3}  & \\text{if } |d| <= 1 \\ 0 & \\text{otherwise}
            \\end{array} \\right .`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (70/81) * (1 - abs(distance)**3)**3


class Cosinus(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the cosinus method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the cosinus weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{\\pi}{4} \\cos(\\frac{
            \\pi}{2}d)  & \\text{if } |d| <= 1 \\ 0 & \\text{otherwise}
            \\end{array} \\right .`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (pi / 4) * cos((pi / 2) * distance)


class Gaussian(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the gaussian method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the gaussian weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{d^{2}}{2}}`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (1 / sqrt(2 * pi)) * exp((-1 / 2) * distance**2)


class Inverse(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the inverse method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the inverse weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Returns
        -------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{1}{|d|}  & \\text{if }
            d \\ne 0 \\ \\infty & \\text{otherwise} \\end{array} \\right .`.

        """
        if distance < -1 or distance > 1:
            return 0
        if distance == 0:
            return maxsize
        return 1 / abs(distance)


class Sinus(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the sinus method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the sinus weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{sin(d * \\pi + (\\pi
            / 2))}{2}  & \\text{if } d \\ne 0 \\ \\infty & \\text{otherwise}
            \\end{array} \\right .`.

        """
        if distance < -1 or distance > 1:
            return 0
        return (sin(distance * pi + (pi / 2)) + 1) / 2


class SinusInverse(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the inverse sinus method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the sinus (by according more importance to the end)  weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{sin(d * \\pi - (\\pi /
            2) + 1)}{2}  & \\text{if } d \\ne 0 \\ \\infty & \\text{otherwise}
            \\end{array} \\right .`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (sin(distance * pi - (pi / 2)) + 1) / 2


class SinusCenterAndEnd(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the center and end sinus method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the sinus (by according more importance to the center and the
        end) weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{sin(2d * \\pi + (\\pi /
            2) + 1)}{2}  & \\text{if } d \\ne 0 \\ \\infty & \\text{otherwise}
            \\end{array} \\right .`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (sin(distance * 2 * pi + (pi / 2)) + 1) / 2


class SinusMiddle(KernelFunction):
    """
    Class inheriting from KernelFunction implementing weight calculation using
    the middle sinus method.
    """

    @staticmethod
    @nb.njit(nb.float64(nb.float64), fastmath=True)
    def compute(distance: float) -> float:  # pragma: no cover
        """
        Computes the sinus (by according more importance to the middle) weight.

        Parameters
        ----------
        distance : float
            The distance between two objects.

        Return
        ------
        float
            :math:`\\left \\{ \\begin{array}{ll} \\frac{sin(2d * \\pi - (\\pi
            2) + 1)}{2}  & \\text{if } d \\ne 0 \\ \\infty & \\text{otherwise}
            \\end{array} \\right .`.
        """
        if distance < -1 or distance > 1:
            return 0
        return (sin(distance * 2 * pi - (pi / 2)) + 1) / 2
