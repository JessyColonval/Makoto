"""
Written by Jessy Colonval.
"""
from numpy import ndarray
from numba import jit, njit, float64
import numpy as np


class BasicMath():

    @staticmethod
    @njit(float64(float64[:], float64[:]), fastmath=True)
    def dot(x: ndarray[float64], y: ndarray[float64]) -> int:
        """
        Don't replace with sum(...), numba doesn't support it.
        """
        if len(x) != len(y):
            raise ValueError("Arrays must have the same size.")
        s = 0
        for i in range(len(x)):
            s += x[i]*y[i]
        return s

    @staticmethod
    @njit(fastmath=True)
    def equation_nD(x: ndarray[float64], y: ndarray[float64]
                    ) -> (float, float):
        """
        Computes the equation 'ax + b' of a segment between two points.

        Parameters
        ----------
        x: ndarray[float64]
            Coordinates of the segment's first point.
        y: ndarray[float64]
            Coordinates of the segment's second point.

        Return
        ------
        (float, float)
            the slope 'a' and the 'b' of the equation.
        """
        dx, dy, px, py = x[0], y[0], x[0], y[0]
        for i in range(1, len(x)):
            dx -= x[i]
            dy -= y[i]
            px *= x[i]
            py *= y[i]
        slope = dy / dx
        b = (px - py) / dx
        return slope, b

    @staticmethod
    @jit(float64(float64[:], float64[:], float64[:]), forceobj=True,
         fastmath=True)
    def distance_line_nD(start: ndarray[float64], end: ndarray[float64],
                         point: ndarray[float64]) -> float:
        """
        Computes the distance between a point and a segment.

        Parameters
        ----------
        start: ndarray[float64]
            Coordinates of the segment's first point.
        end: ndarray[float64]
            Coordinates of the segment's second point.
        point: ndarray[float64]
            Coordinates of the point.

        Return
        ------
        float
            the distance between the given point and segment.

        Raises
        ------
        ValueError
            When the dimension between these three points aren't the same.
        """
        if len(start) != len(end) and len(start) != len(point):
            raise ValueError("Points must have the same dimension.")
        x = start - end
        dp0p2 = point - end
        t = BasicMath.dot(dp0p2, x) / BasicMath.dot(x, x)
        return np.linalg.norm(t*(start-end)+end-point)
