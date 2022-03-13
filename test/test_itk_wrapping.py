#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itk
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck as HC

import strategies as cst  # testing strategies

from IPT.utils import itk_constant_image_from_reference

from IPT.itk_wrapping import itk_add
from IPT.itk_wrapping import itk_subtract
from IPT.itk_wrapping import itk_multiply

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


@given(cst.random_image_strategy(), st.integers(10, 25), st.integers(30, 45))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_add_two_images(image, const1, const2):
    '''
    Given:
        - two images with different constant values
    Then:
        - add the images
    Aseert:
        - resulting image constant value is the sum of the two inputs
    '''
    gt = const1 + const2

    image1 = itk_constant_image_from_reference(image, const1)
    image2 = itk_constant_image_from_reference(image, const2)

    add_image = itk_add(image1, image2)

    res = itk.GetArrayFromImage(add_image.GetOutput())

    assert np.unique(res) == [gt]


@given(cst.random_image_strategy(), st.integers(30, 45), st.integers(10, 25))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_subtract_two_images(image, const1, const2):
    '''
    Given:
        - two images with different constant values
    Then:
        - subtract the images
    Aseert:
        - resulting image constant value is the difference of the two inputs
    '''
    gt = const1 - const2

    image1 = itk_constant_image_from_reference(image, const1)
    image2 = itk_constant_image_from_reference(image, const2)

    subtract_image = itk_subtract(image1, image2)

    res = itk.GetArrayFromImage(subtract_image.GetOutput())

    assert np.unique(res) == [gt]


@given(cst.random_image_strategy(), st.integers(30, 45), st.integers(10, 25))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_multiply_two_images(image, const1, const2):
    '''
    Given:
        - two images with different constant values
    Then:
        - multiply the images
    Aseert:
        - resulting image constant value is the product of the two inputs
    '''
    # this part because if the image has voxel type of 8-bit unsigned integer
    # if the multiiplication results exceed the maximum voxel value(255)
    # then restart from 0 and I have to consider that in  the gt calculation
    pixel_type, _ = itk.template(image)[1]
    if pixel_type is itk.UC:
        gt = np.uint8(const1 * const2)
    else:
        gt = const1 * const2

    #
    # Now start the body of the test
    #
    image1 = itk_constant_image_from_reference(image, const1)
    image2 = itk_constant_image_from_reference(image, const2)

    multiply_image = itk_multiply(image1, image2)

    res = itk.GetArrayFromImage(multiply_image.GetOutput())

    assert np.unique(res) == [gt]
