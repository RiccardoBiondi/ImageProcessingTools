#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itk
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck as HC

import strategies as cst  # testing strategies

from IPT.utils import itk_constant_image_from_reference

from IPT.itk_wrapping import *

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


__all__ = ['itk_connected_components', 'itk_relabel_components']


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


@given(cst.random_image_strategy(), st.integers(12, 25), st.integers(50, 75))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_threshold_and_mask(image, lower, upper):
    '''
    Given:
        - random image strategy
        - upper and lower thresholds

    Then:
        - apply threshold to create a mask
        - mask the input image
    Assert:
        - masked image max and min voxels are in [lower, upper]
    '''
    thr = itk_binary_threshold(image,
                               lower_thr=lower,
                               upper_thr=upper,
                               inside_value=1,
                               outside_value=0,
                               output_type=itk.Image[itk.UC, 3])

    masked = itk_mask(image, thr.GetOutput())

    im_array = itk.GetArrayFromImage(masked.GetOutput())

    assert np.max(im_array) < upper + 1
    assert np.min(im_array[im_array != 0]) > lower - 1
    assert np.min(im_array) == 0  # the minimum value is the padding one (in this case)


@given(cst.random_image_strategy(),
       st.integers(0, 10),
       st.integers(30, 40),
       st.integers(0, 5))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_threshold_initialization(image, lower_thr, upper_thr, outside_value):
    '''
    Given:
        - itk random image
        - lower_thr
        - upper_thr
        - outside value
    Then:
        - initialize threshold image filter
    Assert:
        - correct initialization
    '''

    in_array = itk.GetArrayFromImage(image)
    in_spacing = image.GetSpacing()

    thr = itk_threshold(image,
                        lower_thr=lower_thr,
                        upper_thr=upper_thr,
                        outside_value=outside_value)

    out_array = itk.GetArrayFromImage(thr.GetInput())
    out_spacing = thr.GetInput().GetSpacing()

    assert np.all(out_array == in_array)
    assert np.all(out_spacing == in_spacing)
    assert thr.GetUpper() == upper_thr
    assert thr.GetLower() == lower_thr


@given(cst.random_image_strategy(), st.integers(5, 15))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_threshold_above(image, lower_thr):
    '''
    Given:
        - itk random image
        - lower_thr
    Then:
        - init itk_threshold
        - update the filter
    Assert:
        - minimum image value, excluding the padding one, is lower_thr
    '''

    thr = itk_threshold(image, lower_thr=lower_thr)

    out_array = itk.GetArrayFromImage(thr.GetOutput())

    assert np.min(out_array[out_array != 0]) == lower_thr


@given(cst.random_image_strategy(), st.integers(5, 15))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_threshold_below(image, upper_thr):
    '''
    Given:
        - itk random image
        - upper_thr
    Then:
        - init itk_threshold
        - update the filter
    Assert:
        - maximum image value, excluding the padding one, is upper_thr
    '''

    thr = itk_threshold(image, upper_thr=upper_thr)

    out_array = itk.GetArrayFromImage(thr.GetOutput())

    assert np.max(out_array[out_array != 0]) == upper_thr


@given(cst.random_image_strategy(), st.integers(5, 15), st.integers(25, 30))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_threshold_outside(image, lower_thr, upper_thr):
    '''
    Given:
        - itk random image
        - lower thr
        - upper thr
    Then:
        - init itk_threshold
        - update the filter
    Assert:
        - maximum image value, excluding the padding one, is upper_thr
        - minimum image value, excluding the padding one, is lower_thr
    '''

    thr = itk_threshold(image, lower_thr=lower_thr, upper_thr=upper_thr)

    out_array = itk.GetArrayFromImage(thr.GetOutput())

    assert np.max(out_array[out_array != 0]) == upper_thr
    assert np.min(out_array[out_array != 0]) == lower_thr


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_binary_threshold_default(image):
    '''
    Given:
        - itk random image
    Then:
        - intilize itk BinaryThresholdImageFilter
    Assert:
        - default parameters are correctly initialized
    '''

    thr = itk_binary_threshold(image, update=False)

    assert thr.GetOutsideValue() == 0
    assert thr.GetInsideValue() == 1
    assert thr.GetLowerThreshold() == 0
    assert thr.GetUpperThreshold() == 0


@given(cst.random_image_strategy(), st.integers(12, 25), st.integers(50, 75),
       st.integers(15, 255), st.integers(0, 12))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_binary_threshold(image, lower, upper, inside, outside):
    '''
    Given:
        - random image strategy
        - itk_binary_threshold parameters
    Then:
        - intialize the filter
    Assert:
        - all parameters are correctly initialized
    '''

    thr = itk_binary_threshold(image,
                               lower_thr=lower,
                               upper_thr=upper,
                               inside_value=inside,
                               outside_value=outside,
                               update=False)

    assert thr.GetOutsideValue() == outside
    assert thr.GetInsideValue() == inside
    assert thr.GetLowerThreshold() is lower
    assert thr.GetUpperThreshold() is upper


@given(cst.random_image_strategy(), st.integers(1, 5))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_median_filter_initialization(image, radius):
    '''
    Given:
        - random image
        - radius value
    Then:
        - initialize th median image filter
    Assert:
        - correct parameter initialization
    '''

    median = itk_median(image, radius, update=False)

    assert median.GetRadius()[0] == radius


@given(cst.random_image_strategy(), st.integers(1, 5))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_median_filter_on_salt_and_pepper(image, radius):
    '''
    Given:
        - Image
        - radius
    Then:
        - create a salt and pepper image
        - apply median filter
    Assert:
        - the resulting image has less noise
    '''

    const = itk_constant_image_from_reference(image, value=0)
    s_p = itk_salt_and_pepper_noise(const, salt_value=1,
                                    pepper_value=0, prob=.005)
    median = itk_median(s_p.GetOutput(), radius)

    res = itk.GetArrayFromImage(median.GetOutput())

    assert np.unique(res) == [0]


@given(cst.random_image_strategy(),
       st.integers(1, 5),
       st.integers(12, 25),
       st.integers(50, 75))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_binary_erode_default(image,
                                  radius,
                                  foreground_value,
                                  background_value):
    '''
    Given:
        - Binary Image
        - valid radius
        - foreground value
        - background value
    Then:
        - instantiate the itk binary erosion image filter
    Assert:
        - correct initialization
    '''
    const = itk_constant_image_from_reference(image, value=background_value)
    filt = itk_binary_erode(const,
                            radius=radius,
                            foreground_value=foreground_value,
                            background_value=background_value,
                            update=False)
    kernel = filt.GetKernel()

    assert filt.GetForegroundValue() == foreground_value
    assert filt.GetBackgroundValue() == background_value
    assert kernel.GetRadius()[0] == radius


@given(cst.random_image_strategy(),
       st.integers(50, 75),
       st.integers(10, 25),
       st.integers(1, 5))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_binary_erosion_on_salt_and_pepper_image(image,
                                                 foreground_value,
                                                 background_value,
                                                 radius):
    '''
    Given:
        - random itk image
        - foreground value
        - backgroun value
        - erosion radius
    Then:
        - create a salt and pepper image
        - erode the image
    Assert:
        - the output image array is black
    '''
    const = itk_constant_image_from_reference(image, value=background_value)
    s_p = itk_salt_and_pepper_noise(const, salt_value=foreground_value,
                                    pepper_value=background_value, prob=.005)

    eroded = itk_binary_erode(s_p.GetOutput(),
                              radius=radius,
                              foreground_value=foreground_value,
                              background_value=background_value)

    out_array = itk.GetArrayFromImage(eroded.GetOutput())

    assert np.unique(out_array)[0] == background_value


@given(cst.random_image_strategy(),
       st.integers(1, 5),
       st.integers(12, 25),
       st.integers(50, 75))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_binary_dilate_default(image,
                                   radius,
                                   foreground_value,
                                   background_value):
    '''
    Given:
        - Binary Image
        - valid radius
        - foreground value
        - background value
    Then:
        - instantiate the itk binary dilation image filter
    Assert:
        - correct initialization
    '''
    const = itk_constant_image_from_reference(image, value=background_value)
    filt = itk_binary_dilate(const,
                             radius=radius,
                             foreground_value=foreground_value,
                             background_value=background_value,
                             update=False)
    kernel = filt.GetKernel()

    assert filt.GetForegroundValue() == foreground_value
    assert filt.GetBackgroundValue() == background_value
    assert kernel.GetRadius()[0] == radius


@given(cst.random_image_strategy(),
       st.integers(1, 5),
       st.integers(12, 25),
       st.integers(50, 75))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_binary_morphological_opening_default(image,
                                                  radius,
                                                  foreground_value,
                                                  background_value):
    '''
    Given:
        - Binary Image
        - valid radius
        - foreground value
        - background value
    Then:
        - instantiate the itk binary Morphological Opening image filter
    Assert:
        - correct initialization
    '''
    const = itk_constant_image_from_reference(image, value=background_value)
    filt = itk_binary_morphological_opening(const,
                                            radius=radius,
                                            foreground_value=foreground_value,
                                            background_value=background_value,
                                            update=False)
    kernel = filt.GetKernel()

    assert filt.GetForegroundValue() == foreground_value
    assert filt.GetBackgroundValue() == background_value
    assert kernel.GetRadius()[0] == radius


@given(cst.random_image_strategy(),
       st.integers(1, 5),
       st.integers(12, 25))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_binary_morphological_closing_default(image,
                                                  radius,
                                                  foreground_value):
    '''
    Given:
        - Binary Image
        - valid radius
        - foreground value
        - background value
    Then:
        - instantiate the itk binary Morphological Opening image filter
    Assert:
        - correct initialization
    '''
    const = itk_constant_image_from_reference(image, value=foreground_value)
    filt = itk_binary_morphological_closing(const,
                                            radius=radius,
                                            foreground_value=foreground_value)
    kernel = filt.GetKernel()

    assert filt.GetForegroundValue() == foreground_value
    assert kernel.GetRadius()[0] == radius


@given(cst.random_image_strategy(), st.integers(0, 25), st.booleans())
@settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
def test_connected_components_initialization(image, background_value, fully_connected):
    '''
    Given:
        - random image
        - background_value
        - foreground_value
    Then:
        - binaryze the image
        - instantiate the filter
    Assert:
        - correct parameters initialization
    '''

    thr = itk_binary_threshold(image, upper_thr=128, lower_thr=15)
    connected = itk_connected_components(thr.GetOutput(),
                                        background_value=background_value,
                                        fully_connected=fully_connected)
    assert connected.GetBackgroundValue() == background_value
    assert connected.GetFullyConnected() == fully_connected
