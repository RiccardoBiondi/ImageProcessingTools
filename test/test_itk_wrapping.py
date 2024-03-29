#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itk
import numpy as np

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck as HC

import strategies as cst  # testing strategies

from IPT.utils import itk_constant_image_from_reference

from IPT.itk_wrapping import itk_add
from IPT.itk_wrapping import itk_subtract
from IPT.itk_wrapping import itk_multiply
from IPT.itk_wrapping import itk_invert_intensity
from IPT.itk_wrapping import itk_maximum
from IPT.itk_wrapping import itk_abs
from IPT.itk_wrapping import itk_label_statistics
from IPT.itk_wrapping import itk_shift_scale
from IPT.itk_wrapping import itk_gaussian_normalization
from IPT.itk_wrapping import itk_binary_threshold
from IPT.itk_wrapping import itk_threshold
from IPT.itk_wrapping import itk_mask
from IPT.itk_wrapping import itk_median
from IPT.itk_wrapping import itk_smoothing_recursive_gaussian
from IPT.itk_wrapping import itk_salt_and_pepper_noise
from IPT.itk_wrapping import itk_binary_erode
from IPT.itk_wrapping import itk_binary_dilate
from IPT.itk_wrapping import itk_binary_morphological_opening
from IPT.itk_wrapping import itk_binary_morphological_closing
from IPT.itk_wrapping import itk_connected_components
from IPT.itk_wrapping import itk_relabel_components
from IPT.itk_wrapping import itk_extract
from IPT.itk_wrapping import itk_change_information_from_reference
from IPT.itk_wrapping import itk_voting_binary_iterative_hole_filling
from IPT.itk_wrapping import itk_cast
from IPT.itk_wrapping import itk_unsharp_mask
from IPT.itk_wrapping import itk_slice_by_slice

# TODO test me
from IPT.itk_wrapping import itk_label_overlap_measures
from IPT.itk_wrapping import itk_hausdorff_distance
from IPT.itk_wrapping import itk_hessian_recursive_gaussian
from IPT.itk_wrapping import itk_symmetric_eigen_analysis

# to test 
# itk_flip_image
# itk_sigmoid
# itk_geodesic_active_contour
# itk_signed_maurer_distance_map
# itk_image_gradient_magnitude_recursive
# itk_or

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


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_invert_intensity_default(image):
    '''
    Given:
        - random image
    Then:
        - default filter instantiation
    Assert:
        Correct paramenters init
    '''

    inverter = itk_invert_intensity(image)

    assert inverter.GetMaximum() == 1


@given(cst.random_image_strategy(), st.integers(1, 15))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_invert_intensity_init(image, maximum):
    '''
    Given:
        - random image
        - maximum value
    Then:
        - filter instantiation
    Assert:
        - Correct paramenters init
    '''

    inverter = itk_invert_intensity(image, maximum)

    assert inverter.GetMaximum() == maximum


@given(cst.random_image_strategy(), st.integers(0, 5), st.integers(5, 15))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_invert_intensity(image, value, maximum):
    '''
    Given:
        - constan image
        - maximum value
    Then:
        - filter instantiation
        - filter execution
    Assert:
        - Correct filter result

    '''

    const = itk_constant_image_from_reference(image, value)
    inverter = itk_invert_intensity(const, maximum)
    _ = inverter.Update()

    res = itk.GetArrayFromImage(inverter.GetOutput())

    gt = [maximum - value]

    assert np.unique(res) == gt


@given(cst.random_image_strategy(), st.integers(15, 30), st.integers(25, 45))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_maximum(image, const1, const2):
    '''
    Given:
        - itk.Image
        - two integers constants
    Then:
        - instantiate two constant images
        - compute the maximum
    Assert:
        - computed maximum is max(const1, const2)
    '''

    im1 = itk_constant_image_from_reference(image, const1)
    im2 = itk_constant_image_from_reference(image, const2)

    max_ = itk_maximum(im1, im2)

    res = itk.GetArrayFromImage(max_.GetOutput())

    assert np.unique(res) == [max(const1, const2)]


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_abs_default(image):
    '''
    Test correct filter initialization

    Given:
        - random image
    Then:
        - init the abs image filter
        - not update the filter

    Assert:
        - itk.AbsImageFilter is returned
        - correct filter initialization
    '''

    abs_filter = itk_abs(image, update=False)
    
    assert isinstance(abs_filter, itk.AbsImageFilter)
    assert abs_filter.GetInput() == image


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_abs_non_negative(image):
    '''
    Test correct filter updating

    Given:
        - random image
    Then:
        - init and update the itk_abs filter
    Assert:
        - no negative voxel is still present
    '''

    abs_filter = itk_abs(image)

    result_array = itk.GetArrayFromImage(abs_filter.GetOutput())
    
    assert ~np.any(result_array < 0.)

@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_label_statistics(image, lower, upper):
    '''
    Given:
        - itk random image
        - upper and lower threshold
    Then:
        - compute a mask
        - compute stats inside the mask
    Assert:
        - correct stats computation (using as reference numpy functions)
    '''

    # compute the mask
    mask = itk_binary_threshold(image, upper_thr=upper, lower_thr=lower)
# compute the stats
    stats = itk_label_statistics(image, mask.GetOutput())
    #
    # Compute the reference stats
    #

    mask_a = itk.GetArrayFromImage(mask.GetOutput())
    image_a = itk.GetArrayFromImage(image)
    image_a = image_a.astype(np.float32)

    assert np.isclose(stats.GetMean(1), np.mean(image_a[mask_a == 1]), atol=1e-3)
    assert np.isclose(stats.GetMaximum(1), np.max(image_a[mask_a == 1]), atol=1e-3)
    assert np.isclose(stats.GetMinimum(1), np.min(image_a[mask_a == 1]), atol=1e-3)
    assert np.isclose(stats.GetSigma(1), np.std(image_a[mask_a == 1]), atol=1e-2)


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_shift_scale_default(image):
    '''
    Given:
        - itk Image
    Then:
        - init filter
    Assert:
        - correct initialization with default parameters
    '''
    filter_ = itk_shift_scale(image)

    assert np.isclose(filter_.GetShift(), 0.)
    assert np.isclose(filter_.GetScale(), 1.)


@given(cst.random_image_strategy(), st.floats(-4., 4), st.floats(0., 4.))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_shift_scale_init(image, shift, scale):
    '''
    Given:
        - itk Image
        - shift value
        - scale value
    Then:
        - init filter
    Assert:
        - correct paramenters initialization
    '''

    filter_ = itk_shift_scale(image, shift=shift, scale=scale)

    assert np.isclose(filter_.GetShift(), shift)
    assert np.isclose(filter_.GetScale(), scale)


@given(cst.random_image_strategy(), st.integers(1, 7),
       st.floats(-4., 4), st.floats(.1, 4.))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_shift_scale(image, value, shift, scale):
    '''
    Given:
        - constant image value
        - shift and scale value
    Then:
        - init and update the filter
    Assert:
        - correct shift and scaling. Using numpy functions to compute the
        reference value
    '''

    gt = (float(value) + shift) * scale
    const = itk_constant_image_from_reference(image, value)
    PixelType, dimension = itk.template(const)[1]

    casted = itk.CastImageFilter[itk.Image[PixelType, dimension], itk.Image[itk.F, dimension]].New()
    _ = casted.SetInput(const)

    filter_ = itk_shift_scale(casted.GetOutput(), shift=shift, scale=scale)
    _ = filter_.Update()

    res = itk.GetArrayFromImage(filter_.GetOutput())
    pred = np.unique(res)[0]

    assert np.isclose(pred, gt)


@given(cst.random_image_strategy(), st.integers(12, 25), st.integers(50, 75))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_gaussian_normalization(image, lower, upper):
    '''
    Given:
        - random image
        - upper and lower threshold
    Then:
        - compute mask
        - gaussian normalization
    Assert:
        - zero mean and unitary std deviation inside the masked region
    '''
    PixelType, dimension = itk.template(image)[1]

    mask = itk_binary_threshold(image, upper_thr=upper, lower_thr=lower)

    casted = itk.CastImageFilter[itk.Image[PixelType, dimension], itk.Image[itk.F, dimension]].New()
    _ = casted.SetInput(image)

    normalized = itk_gaussian_normalization(casted.GetOutput(), mask.GetOutput())

    #
    # Conpute the resulting statistics
    #
    stats = itk_label_statistics(normalized.GetOutput(), mask.GetOutput())
    _ = stats.Update()

    assert np.isclose(stats.GetMean(1), 0.0, atol=1e-3)
    assert np.isclose(stats.GetSigma(1), 1.0, atol=1e-3)


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

    assert np.min(out_array[out_array != 0]) >= lower_thr


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

    assert np.max(out_array[out_array != 0]) <= upper_thr


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

    assert np.max(out_array[out_array != 0]) <= upper_thr
    assert np.min(out_array[out_array != 0]) >= lower_thr


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


@given(cst.random_image_strategy(), st.floats(.1, 2.5), st.booleans())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_smoothing_gaussian_initialization(image, sigma,
                                           normalize_across_scale):
    '''
    Given:
        - itk Image
        - sigma value
        - normalize across scale flag
    Then:
        - initilize itk_smoothing_recursive_gaussian filter
    Assert:
        - correct parameters intialization
    '''

    filter_ = itk_smoothing_recursive_gaussian(
                                                image,
                                                sigma,
                                                normalize_across_scale,
                                                update=False
                                                )

    assert filter_.GetSigma() == sigma
    assert filter_.GetNormalizeAcrossScale() == normalize_across_scale


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
def test_connected_components_initialization(image,
                                             background_value,
                                             fully_connected):
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

    _, dimension = itk.template(image)[1]

    thr = itk_binary_threshold(image,
                               upper_thr=128,
                               lower_thr=15,
                               update=False)

    connected = itk_connected_components(thr.GetOutput(),
                                         background_value=background_value,
                                         fully_connected=fully_connected,
                                         output_type=itk.Image[itk.UL, 3],
                                         update=False)

    assert connected.GetBackgroundValue() == background_value
    assert connected.GetFullyConnected() == fully_connected


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_relabel_components_default(image):
    '''
    Given:
        - random image
    Then:
        - instantiate the filter
    Assert:
        - correct parameters initialization
    '''

    relabeler = itk_relabel_components(image, update=False)

    assert relabeler.GetSortByObjectSize() is True


@given(cst.random_image_strategy(),
       st.booleans(),
       st.integers(1, 5),
       st.integers(1, 5))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_relabel_components_initialization(image,
                                           sort_by_object_size,
                                           minimum_object_size,
                                           number_of_object_to_print):
    '''
    Given:
        - random image
        - sort by size
        - minimum size
        - Number of Objects to Print
    Then:
        - instantiate the filter
    Assert:
        - correct parameters initialization
    '''

    relabeler = itk_relabel_components(image,
                                       sort_by_object_size=sort_by_object_size,
                                       minimum_object_size=np.float64(minimum_object_size),
                                       number_of_object_to_print=number_of_object_to_print,
                                       update=False)

    assert relabeler.GetSortByObjectSize() is sort_by_object_size

    # TODO: Must be fixed in the function implementation
    # assert relabeler.GetMinimumObjectSize() == minimum_object_size
    # assert relabeler.GetNumberOfObjectsToPrint() == number_of_object_to_print


@given(cst.random_image_strategy(), st.integers(1, 15), st.integers(25, 50))
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_extract(image, region_index, region_size):
    '''
    Given:
        - Random Image
        - starting indexes
        - region size
    Then:
        - create the extraction region
        - init and update the filter
    Assert:
        - resuilting image have the deisdered size
    '''

    _, dimension = itk.template(image)[1]

    index_ = dimension * [region_index]
    size_ = dimension * [region_size]

    region = itk.ImageRegion[dimension]()
    _ = region.SetIndex(index_)
    _ = region.SetSize(size_)

    extract = itk_extract(image, region)

    res = extract.GetOutput().GetLargestPossibleRegion().GetSize()

    res_size = [res[i] for i in range(dimension)]

    assert res_size == size_


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_label_overlap_measures_defaul(image):
    pass


@given(cst.random_image_strategy(), cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_change_information_from_reference_default(image, reference):
    '''
    Given:
        - random input image
        - random reference image
    Then:
        - instantiate itk_change_information_from_reference with
         default arguments and do not update
    Assert:
        - correct filter initialization
    '''

    filter_ = itk_change_information_from_reference(image, reference, update=False)

    assert filter_.GetChangeDirection() is True
    assert filter_.GetChangeOrigin() is True
    assert filter_.GetChangeSpacing() is True
    assert filter_.GetChangeRegion() is True
    assert filter_.GetUseReferenceImage() is True
    assert filter_.GetCenterImage() is False


@given(cst.random_image_strategy(), # input image
       cst.random_image_strategy(), # reference image
       st.booleans(), # change origin
       st.booleans(), # change spacing
       st.booleans(), #change direction
       st.booleans()# change region
       )
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_change_information_from_reference_init(image,
                                                    reference,
                                                    change_origin,
                                                    change_spacing,
                                                    change_direction,
                                                    change_region):
    '''
    Given:
        - random input image
        - random reference image
        - random boolean flags
    Then:
        - instantiate itk_change_information_from_reference with
          custom arguments
    Assert:
        - correct filter initialization
    '''

    filter_ = itk_change_information_from_reference(image,
                                                    reference,
                                                    change_origin=change_origin,
                                                    change_spacing=change_spacing,
                                                    change_direction=change_direction,
                                                    change_region=change_region,
                                                    update=False)

    assert filter_.GetChangeDirection() is change_direction
    assert filter_.GetChangeOrigin() is change_origin
    assert filter_.GetChangeSpacing() is change_spacing
    assert filter_.GetChangeRegion() is change_region
    assert filter_.GetUseReferenceImage() is True
    assert filter_.GetCenterImage() is False


@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_voting_binary_iterative_hole_filling_default(image):
    '''
    Given:
        - random input image
    Then:
        - instantiate itk_voting_binary_iterative_hole_filling with
         default arguments and do not update
    Assert:
        - correct filter initialization
    '''

    filter_ = itk_voting_binary_iterative_hole_filling(image, update=False)

    assert filter_.GetMajorityThreshold() == 1
    assert filter_.GetMaximumNumberOfIterations() == 10
    assert filter_.GetForegroundValue() == 1
    assert filter_.GetBackgroundValue() == 0
    assert filter_.GetRadius() == [1, 1, 1]


@given(cst.random_image_strategy(), # input image
       st.integers(1, 5), # radius
       st.integers(1, 25), # number of iteratons
       st.integers(1, 5), # majority threshold
       st.integers(10, 255), # foreground value
       st.integers(0, 9) # background
       )
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_voting_binary_iterative_hole_filling_init(
                                                       image,
                                                       radius,
                                                       max_number_of_iterations,
                                                       majority_threshold,
                                                       foreground_value,
                                                       background_value):
    '''
    Given:
        - random input image
        - random input arguments
    Then:
        - instantiate itk_voting_binary_iterative_hole_filling with
          custom arguments
    Assert:
        - correct filter initialization
    '''

    filter_ = itk_voting_binary_iterative_hole_filling(
                                                       image=image,
                                                       radius=radius,
                                                       max_number_of_iterations=max_number_of_iterations,
                                                       majority_threshold=majority_threshold,
                                                       foreground_value=foreground_value,
                                                       background_value=background_value,
                                                       update=False)

    assert filter_.GetMajorityThreshold() == majority_threshold
    assert filter_.GetMaximumNumberOfIterations() == max_number_of_iterations
    assert filter_.GetForegroundValue() == foreground_value
    assert filter_.GetBackgroundValue() == background_value
    assert filter_.GetRadius() == [radius, radius, radius]


@pytest.mark.skip(reason="Non correct testing routine: urrently under checking")
@given(cst.random_image_strategy(), st.integers(1, 5), st.integers(1, 15), st.integers(1, 5))
def test_itk_voting_binary_iterative_hole_filling_on_salt_and_pepper(const, radius, n_iter, majority_threshold):
    '''
    Given:
        - Salt and Pepper image
        - Random parameters to init the filter
    Then:
        - fill the holes of the image
    Assert:
        - result image has got more white voxels than the initial one
    '''
    
    salt_and_pepper = itk_salt_and_pepper_noise(const, salt_value=1,
                                    pepper_value=0, prob=.7)

    filler = itk_voting_binary_iterative_hole_filling(salt_and_pepper.GetOutput(),
                                                      radius=radius,
                                                      max_number_of_iterations=n_iter,
                                                      majority_threshold=majority_threshold)

    inital_image_white_voxels = np.sum(itk.GetArrayFromImage(salt_and_pepper.GetOutput()))
    final_image_white_voxels = np.sum(itk.GetArrayFromImage(filler.GetOutput()))

    assert final_image_white_voxels > inital_image_white_voxels


@given(cst.random_image_strategy(), st.sampled_from(cst.itk_types))
def test_itk_cast(image, new_type):
    '''
    Given:
        - random image
        - new image type
    Then:
        - cast the image to the new type with itk_cast
    Assert:
        - correct casting: PixelType is new_type
        - image dimension is preserved
    '''
    filter_ = itk_cast(image, new_type=new_type)

    casted_type, new_dimension = itk.template(filter_.GetOutput())[1]

    assert casted_type == new_type
    assert new_dimension == 3


#
# Test the itk unsharp mask filter
#



@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_unsharp_mask_default(image):
    '''
    Given:
        - random input image

    Then:
        - instantiate itk_unsharp_mask with
         default arguments and do not update
    Assert:
        - correct filter initialization
    '''

    filter_ = itk_unsharp_mask(image, update=False)

    assert filter_.GetSigmas() == 1.0
    assert filter_.GetAmount() == 0.5
    assert filter_.GetThreshold() == 0
    assert filter_.GetClamp() is True



@given(cst.random_image_strategy(),
       st.floats(0., 10.),
       st.floats(0.1, 2.),
       st.floats(0., 10.),
       st.booleans())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_unsharp_mask_init(image, sigma, amount, threshold, clamp):
    '''
    Given:
        - random input image
        - random sigma
        - random amount
        - random threshold
        - random clamp
    Then:
        - instantiate itk_unsharp_mask with
         the provided arguments and do not update
    Assert:
        - correct filter initialization
    '''

    filter_ = itk_unsharp_mask(
                                image,
                                sigmas=sigma,
                                amount=amount,
                                threshold=threshold,
                                clamp=clamp,
                                update=False)

    assert np.all(np.isclose(filter_.GetSigmas(), sigma))
    assert np.all(np.isclose(filter_.GetAmount(), amount))
    assert np.all(np.isclose(filter_.GetThreshold(), threshold))
    assert filter_.GetClamp() is clamp


#
# Test slice by sluice filter

@given(cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_itk_slice_by_slice_default(image):
    '''
    Given:
        - random image
        - simple pipeline
    Then:
        - instantiate itk_slice_by_slice with the default argument and 
        do no update
    Assert:
        - correct default filter initialization   
    '''
    PixelType, _ = itk.template(image)[1]
    pipeline = itk.BinaryThresholdImageFilter[itk.Image[PixelType, 2], itk.Image[PixelType, 2]].New()
    filter_ = itk_slice_by_slice(image=image, pipeline=pipeline, update=False)

    assert filter_.GetDimension() == 2