"""
Written by Jessy Colonval.
"""
from unittest import TestCase
from sys import maxsize
import numpy as np

from src.detection.kernel_functions import (Rectangular, Triangular, TriCube,
                                            Epanechnikov, BiWeight, TriWeight,
                                            Cosinus, Gaussian, Inverse, Sinus,
                                            SinusInverse, SinusMiddle,
                                            SinusCenterAndEnd)


class TestKernelFunctions(TestCase):
    """
    Unit tests for all kernel functions computation.
    """

    def test_rectangular_maximum_in_0(self):
        """
        Verifies that the maximum rectangular weight is at 0.
        """
        maximum = Rectangular.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Rectangular.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_tritangular_maximum_in_0(self):
        """
        Verifies that the maximum triangular weight is at 0.
        """
        maximum = Triangular.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Triangular.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_epanechnikov_maximum_in_0(self):
        """
        Verifies that the maximum epanechnikov weight is at 0.
        """
        maximum = Epanechnikov.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Epanechnikov.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_biweight_maximum_in_0(self):
        """
        Verifies that the maximum biweight weight is at 0.
        """
        maximum = BiWeight.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = BiWeight.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_triweight_maximum_in_0(self):
        """
        Verifies that the maximum triweight weight is at 0.
        """
        maximum = TriWeight.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = TriWeight.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_cosinus_maximum_in_0(self):
        """
        Verifies that the maximum cosinus weight is at 0.
        """
        maximum = Cosinus.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Cosinus.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_guassian_maximum_in_0(self):
        """
        Verifies that the maximum gaussian weight is at 0.
        """
        maximum = Gaussian.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Gaussian.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_inverse_maximum_in_0(self):
        """
        Verifies that the maximum inverse weight is at 0.
        """
        maximum = Inverse.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Inverse.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_tricube_maximum_in_0(self):
        """
        Verifies that the maximum tricube weight is at 0.
        """
        maximum = TriCube.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = TriCube.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_sinus_maximum_in_0(self):
        """
        Verifies that the maximum sinus weight is at 0.
        """
        maximum = Sinus.compute(0)
        for distance in np.arange(0.0001, 1.0, 0.0001):
            weight = Sinus.compute(distance)
            self.assertLessEqual(weight, maximum)

    def test_rectangular_out_of_range(self):
        """
        Verifies that the rectangular weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = Rectangular.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = Rectangular.compute(distance)
            self.assertEqual(weight, 0)

    def test_triangular_out_of_range(self):
        """
        Verifies that the triangular weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = Triangular.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = Triangular.compute(distance)
            self.assertEqual(weight, 0)

    def test_epanechnikov_out_of_range(self):
        """
        Verifies that the epanechnikov weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = Epanechnikov.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = Epanechnikov.compute(distance)
            self.assertEqual(weight, 0)

    def test_biweight_out_of_range(self):
        """
        Verifies that the bi-weight weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = BiWeight.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = BiWeight.compute(distance)
            self.assertEqual(weight, 0)

    def test_triweight_out_of_range(self):
        """
        Verifies that the tri-weight weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = TriWeight.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = TriWeight.compute(distance)
            self.assertEqual(weight, 0)

    def test_cosinus_out_of_range(self):
        """
        Verifies that the cosinus weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = TriWeight.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = TriWeight.compute(distance)
            self.assertEqual(weight, 0)

    def test_gaussian_out_of_range(self):
        """
        Verifies that the gaussian weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = Gaussian.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = Gaussian.compute(distance)
            self.assertEqual(weight, 0)

    def test_inverse_out_of_range(self):
        """
        Verifies that the inverse weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = Inverse.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = Inverse.compute(distance)
            self.assertEqual(weight, 0)

    def test_tricube_out_of_range(self):
        """
        Verifies that the tri-cube weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = TriCube.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = TriCube.compute(distance)
            self.assertEqual(weight, 0)

    def test_sinus_out_of_range(self):
        """
        Verifies that the sinus weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = Sinus.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = Sinus.compute(distance)
            self.assertEqual(weight, 0)

    def test_sinus_inverse_out_of_range(self):
        """
        Verifies that the sinus inverse weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = SinusInverse.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = SinusInverse.compute(distance)
            self.assertEqual(weight, 0)

    def test_sinus_middle_out_of_range(self):
        """
        Verifies that the sinus middle weight is always equal to 0 when the
        distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = SinusMiddle.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = SinusMiddle.compute(distance)
            self.assertEqual(weight, 0)

    def test_sinus_center_and_end_out_of_range(self):
        """
        Verifies that the sinus center and end weight is always equal to 0 when
        the distance is outside the interval [0.0; 1.0].
        """
        for distance in range(-10000, -1):
            weight = SinusCenterAndEnd.compute(distance)
            self.assertEqual(weight, 0)
        for distance in range(2, 10000):
            weight = SinusCenterAndEnd.compute(distance)
            self.assertEqual(weight, 0)

    def test_rectangular_increases_in_0_and_1(self):
        """
        Verifies that the rectangular weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Rectangular.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Rectangular.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_triangular_increases_in_0_and_1(self):
        """
        Verifies that the triangular weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Triangular.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Triangular.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_epanechnikov_increases_in_0_and_1(self):
        """
        Verifies that the epanechnikov weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Epanechnikov.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Epanechnikov.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_biweight_increases_in_0_and_1(self):
        """
        Verifies that the bi-weight weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = BiWeight.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = BiWeight.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_triweight_increases_in_0_and_1(self):
        """
        Verifies that the tri-weight weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = TriWeight.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = TriWeight.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_guassian_increases_in_0_and_1(self):
        """
        Verifies that the gaussian weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Gaussian.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Gaussian.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_cosinus_increases_in_0_and_1(self):
        """
        Verifies that the cosinus weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Cosinus.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Cosinus.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_inverse_increases_in_0_and_1(self):
        """
        Verifies that the inverse weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Inverse.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Inverse.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_tricube_increases_in_0_and_1(self):
        """
        Verifies that the tri-cube weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = TriCube.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = TriCube.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_sinus_increases_in_0_and_1(self):
        """
        Verifies that the sinus weight increases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = Sinus.compute(d0)
            for d1 in np.arange(d0+pas, 1.0, pas):
                w1 = Sinus.compute(d1)
                self.assertGreaterEqual(w0, w1)

    def test_rectangular_decreases_in_minus_1_and_0(self):
        """
        Verifies that the rectangular weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Rectangular.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Rectangular.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_triangular_decreases_in_minus_1_and_0(self):
        """
        Verifies that the triangular weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Triangular.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Triangular.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_epanechnikov_decreases_in_minus_1_and_0(self):
        """
        Verifies that the epanechnikov weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Epanechnikov.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Epanechnikov.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_biweight_decreases_in_minus_1_and_0(self):
        """
        Verifies that the bi-weight weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = BiWeight.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = BiWeight.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_triweight_decreases_in_minus_1_and_0(self):
        """
        Verifies that the tri-weight weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = TriWeight.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = TriWeight.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_guassian_decreases_in_minus_1_and_0(self):
        """
        Verifies that the gaussian weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Gaussian.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Gaussian.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_cosinus_decreases_in_minus_1_and_0(self):
        """
        Verifies that the cosinus weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Cosinus.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Cosinus.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_inverse_decreases_in_minus_1_and_0(self):
        """
        Verifies that the inverse weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Inverse.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Inverse.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_tricube_decreases_in_minus_1_and_0(self):
        """
        Verifies that the tri-cube weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = TriCube.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = TriCube.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_sinus_decreases_in_minus_1_and_0(self):
        """
        Verifies that the sinus weight decreases when the distance
        increases in the interval [-1.0; 0.0].
        """
        pas = 0.001
        for d0 in np.arange(-1, 0.0-pas, pas):
            w0 = Sinus.compute(d0)
            for d1 in np.arange(d0+pas, 0.0, pas):
                w1 = Sinus.compute(d1)
                self.assertLessEqual(w0, w1)

    def test_rectangular_always_positive(self):
        """
        Verifies that the rectanguar weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            weight = Rectangular.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_triangular_always_positive(self):
        """
        Verifies that the triangular weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = Triangular.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_epanechnikov_always_positive(self):
        """
        Verifies that the epanechnikov weight is always greater or equal to
        zero when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = Epanechnikov.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_biweight_always_positive(self):
        """
        Verifies that the bi-weight weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = BiWeight.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_triweight_always_positive(self):
        """
        Verifies that the tri-weight weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = TriWeight.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_cosinus_always_positive(self):
        """
        Verifies that the cosinus weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = Cosinus.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_guassian_always_positive(self):
        """
        Verifies that the guassian weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = Gaussian.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_inverse_always_positive(self):
        """
        Verifies that the inverse weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = Inverse.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_tricube_always_positive(self):
        """
        Verifies that the tri-cube weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = TriCube.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_sinus_always_positive(self):
        """
        Verifies that the sinus weight is always greater or equal to zero
        when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = Sinus.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_sinus_inverse_always_positive(self):
        """
        Verifies that the sinus inverse weight is always greater or equal to
        zero when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = SinusInverse.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_sinus_middle_always_positive(self):
        """
        Verifies that the sinus middle weight is always greater or equal to
        zero when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = SinusMiddle.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_sinus_center_and_end_always_positive(self):
        """
        Verifies that the sinus center and end weight is always greater or
        equal to zero when the distance is in the interval [-1.0; 1.0].
        """
        for distance in np.arange(-1, 1.0, 0.001):
            weight = SinusCenterAndEnd.compute(distance)
            self.assertGreaterEqual(weight, 0)

    def test_rectangular_symetric(self):
        """
        Verifies that the rectangular weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Rectangular.compute(distance)
            w1 = Rectangular.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_triangular_symetric(self):
        """
        Verifies that the triangular weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Triangular.compute(distance)
            w1 = Triangular.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_epanechnikov_symetric(self):
        """
        Verifies that the epanechnikov weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Epanechnikov.compute(distance)
            w1 = Epanechnikov.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_biweight_symetric(self):
        """
        Verifies that the bi-weight weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = BiWeight.compute(distance)
            w1 = BiWeight.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_triweight_symetric(self):
        """
        Verifies that the tri-weight weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = TriWeight.compute(distance)
            w1 = TriWeight.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_tricube_symetric(self):
        """
        Verifies that the tri-cube weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = TriCube.compute(distance)
            w1 = TriCube.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_gaussian_symetric(self):
        """
        Verifies that the gaussian weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Gaussian.compute(distance)
            w1 = Gaussian.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_inverse_symetric(self):
        """
        Verifies that the inverse weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Inverse.compute(distance)
            w1 = Inverse.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_cosinus_symetric(self):
        """
        Verifies that the cosinus weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Cosinus.compute(distance)
            w1 = Cosinus.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_sinus_symetric(self):
        """
        Verifies that the sinus weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = Sinus.compute(distance)
            w1 = Sinus.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_sinus_inverse_symetric(self):
        """
        Verifies that the sinus inverse weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = SinusInverse.compute(distance)
            w1 = SinusInverse.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_sinus_middle_symetric(self):
        """
        Verifies that the sinus middle weights of a positive and negative
        values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = SinusMiddle.compute(distance)
            w1 = SinusMiddle.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_sinus_center_and_end_symetric(self):
        """
        Verifies that the sinus center and end weights of a positive and
        negative values are always equals in the interval [-1.0; 1.0].
        """
        for distance in np.arange(0, 1.0, 0.001):
            w0 = SinusCenterAndEnd.compute(distance)
            w1 = SinusCenterAndEnd.compute(-distance)
            self.assertAlmostEqual(w0, w1)

    def test_rectangular_distance_is_1(self):
        """
        Verifies that the rectangular distances at 1 and -1 are equals to 1/2.
        """
        w1 = Rectangular.compute(1)
        w2 = Rectangular.compute(-1)
        self.assertEqual(w1, 1/2)
        self.assertEqual(w2, 1/2)

    def test_triangular_distance_is_1(self):
        """
        Verifies that the triangular weights at 1 and -1 are equals to 0.
        """
        w1 = Triangular.compute(1)
        w2 = Triangular.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_epanechnikov_distance_is_1(self):
        """
        Verifies that the epanechnikov weights at 1 and -1 are equals to 0.
        """
        w1 = Epanechnikov.compute(1)
        w2 = Epanechnikov.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_biweight_distance_is_1(self):
        """
        Verifies that the bi-weight weights at 1 and -1 are equals to 0.
        """
        w1 = BiWeight.compute(1)
        w2 = BiWeight.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_triweight_distance_is_1(self):
        """
        Verifies that the tri-weight weights at 1 and -1 are equals to 0.
        """
        w1 = TriWeight.compute(1)
        w2 = TriWeight.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_cosinus_distance_is_1(self):
        """
        Verifies that the cosinus weights at 1 and -1 are equals to 0.
        """
        w1 = Cosinus.compute(1)
        w2 = Cosinus.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_guassian_distance_is_1(self):
        """
        Verifies that the gaussian weights at 1 and -1 are equals to 0.241970.
        """
        w1 = Gaussian.compute(1)
        w2 = Gaussian.compute(-1)
        self.assertAlmostEqual(w1, 0.2419707245)
        self.assertAlmostEqual(w2, 0.2419707245)

    def test_inverse_distance_is_1(self):
        """
        Verifies that the inverse weights at 1 and -1 are equals to 1.0.
        """
        w1 = Inverse.compute(1)
        w2 = Inverse.compute(-1)
        self.assertAlmostEqual(w1, 1.0)
        self.assertAlmostEqual(w2, 1.0)

    def test_tricube_distance_is_1(self):
        """
        Verifies that the tri-cube weights at 1 and -1 are equals to 0.
        """
        w1 = TriCube.compute(1)
        w2 = TriCube.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_sinus_distance_is_1(self):
        """
        Verifies that the sinus weights at 1 and -1 are equals to 0.
        """
        w1 = Sinus.compute(1)
        w2 = Sinus.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_sinus_inverse_distance_is_1(self):
        """
        Verifies that the sinus inverse weights at 1 and -1 are equals to 1.0.
        """
        w1 = SinusInverse.compute(1)
        w2 = SinusInverse.compute(-1)
        self.assertAlmostEqual(w1, 1.0)
        self.assertAlmostEqual(w2, 1.0)

    def test_sinus_middle_distance_is_1(self):
        """
        Verifies that the sinus middle weights at 1 and -1 are equals to 0.
        """
        w1 = SinusMiddle.compute(1)
        w2 = SinusMiddle.compute(-1)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_sinus_center_and_end_distance_is_1(self):
        """
        Verifies that the sinus center and end weights at 1 and -1 are equals
        to 1.0.
        """
        w1 = SinusCenterAndEnd.compute(1)
        w2 = SinusCenterAndEnd.compute(-1)
        self.assertAlmostEqual(w1, 1.0)
        self.assertAlmostEqual(w2, 1.0)

    def test_rectangular_distance_is_0(self):
        """
        Verifies that the rectangular weight at 0 is equals to 1/2.
        """
        weight = Rectangular.compute(0)
        self.assertEqual(weight, 1/2)

    def test_triangular_distance_is_0(self):
        """
        Verifies that the triangular weight at 0 is equals to 1.0.
        """
        weight = Triangular.compute(0)
        self.assertAlmostEqual(weight, 1.0)

    def test_epanechnikov_distance_is_0(self):
        """
        Verifies that the epanechnikov weight at 0 is equals to 3/4.
        """
        weight = Epanechnikov.compute(0)
        self.assertAlmostEqual(weight, 3/4)

    def test_biweight_distance_is_0(self):
        """
        Verifies that the bi-weight weight at 0 is equals to 15/16.
        """
        weight = BiWeight.compute(0)
        self.assertAlmostEqual(weight, 15/16)

    def test_triweight_distance_is_0(self):
        """
        Verifies that the tri-weight weight at 0 is equals to 35/32.
        """
        weight = TriWeight.compute(0)
        self.assertAlmostEqual(weight, 35/32)

    def test_cosinus_distance_is_0(self):
        """
        Verifies that the cosinus weight at 0 is equals to 0.7853981634.
        """
        weight = Cosinus.compute(0)
        self.assertAlmostEqual(weight, 0.7853981634)

    def test_guassian_distance_is_0(self):
        """
        Verifies that the gaussian weight at 0 is equals to 0.3989422804.
        """
        weight = Gaussian.compute(0)
        self.assertAlmostEqual(weight, 0.3989422804)

    def test_inverse_distance_is_0(self):
        """
        Verifies that the inverse weight at 0 is equals to +infite.
        """
        weight = Inverse.compute(0)
        self.assertAlmostEqual(weight, maxsize)

    def test_tricube_distance_is_0(self):
        """
        Verifies that the tri-cube weight at 0 is equals to 0.8641975308641975.
        """
        weight = TriCube.compute(0)
        self.assertAlmostEqual(weight, 0.8641975308641975)

    def test_sinus_distance_is_0(self):
        """
        Verifies that the sinus weight at 0 is equals to 1.0.
        """
        weight = Sinus.compute(0)
        self.assertAlmostEqual(weight, 1.0)

    def test_sinus_inverse_distance_is_0(self):
        """
        Verifies that the sinus inverse weight at 0 is equals to 0.0.
        """
        weight = SinusInverse.compute(0)
        self.assertAlmostEqual(weight, 0.0)

    def test_sinus_middle_distance_is_0(self):
        """
        Verifies that the sinus middle weight at 0 is equals to 0.0.
        """
        weight = SinusMiddle.compute(0)
        self.assertAlmostEqual(weight, 0.0)

    def test_sinus_center_and_end_distance_is_0(self):
        """
        Verifies that the sinus center and end weight at 0 is equals to 1.0.
        """
        weight = SinusCenterAndEnd.compute(0)
        self.assertAlmostEqual(weight, 1.0)

    def test_rectangular_distance_0dot5(self):
        """
        Verifies that the rectangular distances at 1/2 and -1/2 are equals to
        1/2.
        """
        w1 = Rectangular.compute(0.5)
        w2 = Rectangular.compute(-0.5)
        self.assertEqual(w1, 1/2)
        self.assertEqual(w2, 1/2)

    def test_triangular_distance_0dot5(self):
        """
        Verifies that the triangular distances at 1/2 and -1/2 are equals to
        1/2.
        """
        w1 = Triangular.compute(0.5)
        w2 = Triangular.compute(-0.5)
        self.assertAlmostEqual(w1, 0.5)
        self.assertAlmostEqual(w2, 0.5)

    def test_epanechnikov_distance_0dot5(self):
        """
        Verifies that the epanechnikov distances at 1/2 and -1/2 are equals to
        0.5625.
        """
        w1 = Epanechnikov.compute(0.5)
        w2 = Epanechnikov.compute(-0.5)
        self.assertAlmostEqual(w1, 0.5625)
        self.assertAlmostEqual(w2, 0.5625)

    def test_biweight_distance_0dot5(self):
        """
        Verifies that the bi-weight distances at 1/2 and -1/2 are equals to
        0.52734375.
        """
        w1 = BiWeight.compute(0.5)
        w2 = BiWeight.compute(-0.5)
        self.assertAlmostEqual(w1, 0.52734375)
        self.assertAlmostEqual(w2, 0.52734375)

    def test_triweight_distance_0dot5(self):
        """
        Verifies that the tri-weight distances at 1/2 and -1/2 are equals to
        0.46142578125.
        """
        w1 = TriWeight.compute(0.5)
        w2 = TriWeight.compute(-0.5)
        self.assertAlmostEqual(w1, 0.46142578125)
        self.assertAlmostEqual(w2, 0.46142578125)

    def test_cosinus_distance_0dot5(self):
        """
        Verifies that the cosinus distances at 1/2 and -1/2 are equals to
        0.5553603673.
        """
        w1 = Cosinus.compute(0.5)
        w2 = Cosinus.compute(-0.5)
        self.assertAlmostEqual(w1, 0.5553603673)
        self.assertAlmostEqual(w2, 0.5553603673)

    def test_guassian_distance_0dot5(self):
        """
        Verifies that the gaussian distances at 1/2 and -1/2 are equals to
        0.3520653268.
        """
        w1 = Gaussian.compute(0.5)
        w2 = Gaussian.compute(-0.5)
        self.assertAlmostEqual(w1, 0.3520653268)
        self.assertAlmostEqual(w2, 0.3520653268)

    def test_inverse_distance_0dot5(self):
        """
        Verifies that the inverse distances at 1/2 and -1/2 are equals to
        2.0.
        """
        w1 = Inverse.compute(0.5)
        w2 = Inverse.compute(-0.5)
        self.assertAlmostEqual(w1, 2.0)
        self.assertAlmostEqual(w2, 2.0)

    def test_tricube_distance_0dot5(self):
        """
        Verifies that the tri-cube distances at 1/2 and -1/2 are equals to
        0.5789448302469136.
        """
        w1 = TriCube.compute(0.5)
        w2 = TriCube.compute(-0.5)
        self.assertAlmostEqual(w1, 0.5789448302469136)
        self.assertAlmostEqual(w2, 0.5789448302469136)

    def test_sinus_distance_0dot5(self):
        """
        Verifies that the sinus distances at 1/2 and -1/2 are equals to
        1/2.
        """
        w1 = Sinus.compute(0.5)
        w2 = Sinus.compute(-0.5)
        self.assertAlmostEqual(w1, 0.5)
        self.assertAlmostEqual(w2, 0.5)

    def test_sinus_inverse_distance_0dot5(self):
        """
        Verifies that the sinus inverse distances at 1/2 and -1/2 are equals to
        1/2.
        """
        w1 = SinusInverse.compute(0.5)
        w2 = SinusInverse.compute(-0.5)
        self.assertAlmostEqual(w1, 0.5)
        self.assertAlmostEqual(w2, 0.5)

    def test_sinus_middle_distance_0dot5(self):
        """
        Verifies that the sinus moddle distances at 1/2 and -1/2 are equals to
        1.0.
        """
        w1 = SinusMiddle.compute(0.5)
        w2 = SinusMiddle.compute(-0.5)
        self.assertAlmostEqual(w1, 1.0)
        self.assertAlmostEqual(w2, 1.0)

    def test_sinus_center_and_end_distance_0dot5(self):
        """
        Verifies that the sinus center and end distances at 1/2 and -1/2 are
        equals to 0.0.
        """
        w1 = SinusCenterAndEnd.compute(0.5)
        w2 = SinusCenterAndEnd.compute(-0.5)
        self.assertAlmostEqual(w1, 0.0)
        self.assertAlmostEqual(w2, 0.0)

    def test_sinus_inverse_maximum_in_extrem(self):
        """
        Verifies that the maximal sinus inverse weight is only at -1 and 1.
        """
        m1 = SinusInverse.compute(-1)
        m2 = SinusInverse.compute(1)
        for distance in np.arange(0.0001, 0.9999, 0.0001):
            weight = SinusInverse.compute(distance)
            self.assertLess(weight, m1)
            self.assertLess(weight, m2)

    def test_sinus_inverse_decreases(self):
        """
        Verifies that the sinus inverse weights decreases when the distance
        increases in the interval [0.0; 1.0].
        """
        pas = 0.001
        for d0 in np.arange(0, 1.0-pas, pas):
            w0 = SinusInverse.compute(d0)
            for d1 in np.arange(d0+pas, 1.0-pas, pas):
                w1 = SinusInverse.compute(d1)
                self.assertLessEqual(w0, w1)
