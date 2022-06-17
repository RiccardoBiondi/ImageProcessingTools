import itk
import logging
# import numpy as np

from IPT.decorators import update
from IPT.utils import infer_itk_image_type


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['itk_add', 'itk_subtract', 'itk_multiply', 'itk_invert_intensity',
           'itk_maximum', 'itk_label_statistics', 'itk_shift_scale',
           'itk_gaussian_normalization', 'itk_mask',
           'itk_salt_and_pepper_noise', 'itk_threshold', 'itk_binary_threshold',
           'itk_median', 'itk_smoothing_recursive_gaussian', 'itk_binary_erode',
           'itk_binary_dilate', 'itk_binary_morphological_opening',
           'itk_binary_morphological_closing', 'itk_connected_components',
           'itk_relabel_components', 'itk_extract',
           'itk_label_overlap_measures', 'itk_hausdorff_distance',
           'itk_hessian_recursive_gaussian', 'itk_symmetric_eigen_analysis',
           'itk_change_information_from_reference']

#
# Aritmetic Operators
#


@update
def itk_add(image1, image2, input1_type=None,
            input2_type=None, output_type=None, **kwargs):
    """
    Add two itk images. Images must heve the same size and physical space.

    Parameters
    ----------
    image1: itk.Image
        first image to add
    image2: itk.Image
        second image to add
    input1_type : itk.Image type (i.e. itk.Image[itk.UC, 2])
         input1 type. If not specified it is inferred from the input image1
    input2_type : itk.Image type (i.e. itk.Image[itk.UC, 2])
         input2 type. If not specified it is inferred from the input image2
    output_type : itk.Image type (i.e. itk.Image[itk.UC, 2])
         output type. If not specified it is iferred from the input image2
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    add_image: itk.AddImageFilter
        itk.AddImageFilter instance. As default the instance is updated. To not
        update the instance pecify update=False as kwargs.
    """
    logging.debug("Adding two images")

    Input1Type = infer_itk_image_type(image1, input1_type)
    Input2Type = infer_itk_image_type(image2, input2_type)
    OutputType = infer_itk_image_type(image1, output_type)

    add_image = itk.AddImageFilter[Input1Type, Input2Type, OutputType].New()
    _ = add_image.SetInput1(image1)
    _ = add_image.SetInput2(image2)

    return add_image


@update
def itk_subtract(image1, image2, input1_type=None,
                 input2_type=None, output_type=None, **kwargs):
    '''
    Subtract two itk images. Images must heve the same size and physical space.

    Parameters
    ----------
    image1: itk.Image
        first image to add
    image2: itk.Image
        second image to subtract
    input1_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input1 type. If not specified it is inferred from the input image1
    input2_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input2 type. If not specified it is inferred from the input image2
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output type. If not specified it is iferred from the input image2
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    subtract_image: itk.SubtractImageFilter
        itk.SubtractImageFilter instance. As default the instance is updated.
        To not update the instance pecify update=False as kwargs.
    '''
    logging.debug('Subtract two Images')

    Input1Type = infer_itk_image_type(image1, input1_type)
    Input2Type = infer_itk_image_type(image2, input2_type)
    OutputType = infer_itk_image_type(image1, output_type)

    subtract_image = itk.SubtractImageFilter[Input1Type, Input2Type, OutputType].New()
    _ = subtract_image.SetInput1(image1)
    _ = subtract_image.SetInput2(image2)

    return subtract_image


@update
def itk_multiply(image1, image2, input1_type=None,
                 input2_type=None, output_type=None, **kwargs):
    '''
    Voxel-Wise multiplication of two images.

    Parameters
    ----------
    image1: itk.Image
        first image to multiply to
    image2: itk.Image
        second image to multiply to
    input1_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input1 type. If not specified it is inferred from the input image1
    input2_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input2 type. If not specified it is inferred from the input image2
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output type. If not specified it is iferred from the input image2
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    multiply_image: itk.MultiplyImageFilter
        itk.MultiplyImageFilter instance. As default the instance is updated.
        To not update the instance pecify update=False as kwargs.
    '''
    logging.debug('Multiply two Images')

    Input1Type = infer_itk_image_type(image1, input1_type)
    Input2Type = infer_itk_image_type(image2, input2_type)
    OutputType = infer_itk_image_type(image1, output_type)

    multiply_image = itk.MultiplyImageFilter[Input1Type, Input2Type, OutputType].New()
    _ = multiply_image.SetInput1(image1)
    _ = multiply_image.SetInput2(image2)

    return multiply_image

#
# Logic Operations
#


@update
def itk_invert_intensity(image, maximum=1,
                         input_type=None, output_type=None, **kwargs):
    '''
    Invert the intensity of an image.
    InvertIntensityImageFilter inverts intensity of pixels by subtracting pixel
    value to a maximum value

    Parameters
    ----------
    image: itk.Image
        image to invert
    maximum: int
        maximum value
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Result
    ------
    inverter: itk.InvertIntensityImageFilter
        itk.InvertIntensityImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''
    logging.debug(f'Invert Intensity: -maximum: {maximum}')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    inverter = itk.InvertIntensityImageFilter[InputType, OutputType].New()
    _ = inverter.SetInput(image)
    _ = inverter.SetMaximum(maximum)

    return inverter


#
# Algebric Operations
#

#
# Statistical Operations
#
@update
def itk_maximum(image1, image2,
                image1_type=None, image2_type=None, output_type=None,
                **kwargs):
    '''
    Implements a pixel-wise operator Max(a,b) between two images.

    Parameters
    ----------
    image1: itk.Image
      first input image
    image2: itk.Image
      second input image
    image1_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
      input1 image type. If not specified it is inferred from the input image1
    image2_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
      input2 image type. If not specified it is inferred from the input image2
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
      output image type. If not specified it is iferred from the input image2
    kwargs:
        keyword arguments to control the behaviour of deorators

    Returns
    -------
    max_ : itk.MaximumImageFilter
        itk.MaximumImageFilter instance.  As default the instance is updated.
        To not update the instance pecify update=False as kwargs.
    '''

    logging.debug('Computing Maximum Between Two Images')

    Input1Type = infer_itk_image_type(image1, image1_type)
    Input2Type = infer_itk_image_type(image2, image2_type)
    OutputType = infer_itk_image_type(image1, output_type)

    max_ = itk.MaximumImageFilter[Input1Type, Input2Type, OutputType].New()

    _ = max_.SetInput1(image1)
    _ = max_.SetInput2(image2)

    return max_


@update
def itk_label_statistics(image, labelmap, input_type=None, **kwargs):
    '''
    Given an intensity image and a label map, compute min, max, variance and
    mean of the pixels associated with each label or segment.

    Parameters
    ----------
    image: itk.Image
        intensity image
    labelmap: itk.LabelMap
        label map
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    filter_ : itk.LabelStatisticsImageFilter
        itk.LabelStatisticsImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''

    logging.debug('Computing Lebel Statistics')
    InputType = infer_itk_image_type(image, input_type)

    # TODO Improve the type definition for the labelmap object
    filter_ = itk.LabelStatisticsImageFilter[InputType, type(labelmap)].New()
    _ = filter_.SetLabelInput(labelmap)
    _ = filter_.SetInput(image)

    return filter_


@update
def itk_shift_scale(image, shift=0., scale=1.,
                    input_type=None, output_type=None, **kwargs):
    '''
    Shift and scale the pixels in an image.

    Parameters
    ----------
    image: itk.Image
        image to apply filter to
    shift: float
        shift factor
    scale: float
        scale factor
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    filter_ : itk.ShiftScaleImageFilter
        itk.ShiftScaleImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''
    logging.debug(f'Shift and Scale: -shift: {shift} -scale: {scale}')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    filter_ = itk.ShiftScaleImageFilter[InputType, OutputType].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetScale(scale)
    _ = filter_.SetShift(shift)

    return filter_


@update
def itk_gaussian_normalization(image, mask, label=1,
                               input_type=None, output_type=None, **kwargs):
    '''
    Normalize the datata according to mean and standard deviation of the
    voxels inside the specified mask image

    Parameters
    ----------
    image: itk.Image
        image to normalize
    mask: itk.Image
        ROI mask
    label: int
        label value to determine the ROI
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    '''

    # TODO imporve documentation
    logging.debug(f'Running Gaussian Normalization. ROI label={label}')

    stats = itk_label_statistics(image, mask,
                                 input_type, update=kwargs.get('update', True))

    # TODO add standard values for the case in which the label filter is not
    # updated?? mbah
    shift = -stats.GetMean(label)
    scale = 1. / abs(stats.GetSigma(label))

    normalized = itk_shift_scale(image, shift=shift, scale=scale,
                                 input_type=input_type,
                                 output_type=output_type,
                                 update=kwargs.get('update', True))

    return normalized


#
# Image Functions
#


@update
def itk_mask(image, mask, masking_value=0, outside_value=0,
             input_type=None, mask_type=None, output_type=None, **kwargs):

    '''
    Mask an image with a binary mask. The image and the mask must
    be in the same physical space.

    Parameters
    ----------
    image : itk.Image
         image to mask
    mask : itk.Image
         binary mask
    masking_value : int
         label object ot mask
    outside_value : PixelType
         value to which set the voxels outside the mask
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    mask_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         mask image type. If not specified it is iferred from the mask image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    masker : itk.MaskImageFilter
         mask image filter. As default the instance is updated.
         To not update the instance pecify update=False as kwargs.
    '''
    logging.debug('Masking the Image')

    InputType = infer_itk_image_type(image, input_type)
    MaskType = infer_itk_image_type(mask, mask_type)
    OutputType = infer_itk_image_type(image, output_type)

    masker = itk.MaskImageFilter[InputType, MaskType, OutputType].New()
    _ = masker.SetInput(image)
    _ = masker.SetMaskImage(mask)
    _ = masker.SetOutsideValue(outside_value)
    _ = masker.SetMaskingValue(masking_value)

    return masker


@update
def itk_threshold(image,
                  upper_thr=None,
                  lower_thr=None,
                  outside_value=0,
                  input_type=None,
                  **kwargs):
    '''
    Apply a threshold on the image. If only lower_thr is specified, the image
    is thresholded below this value. If only upper_thr is specified, the image
    is thresholded above this value. if both values are specified,
    all the values outside this interval are thresholded

    Parameters
    ----------
    image: itk.Image
        image to threshold
    upper_thr : PixelType
        uppert threshold value
    lower_thr : PixelType
        lower threshold
    outside_value : PixelType
        value to replace for all the thresholded regions
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        input image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    threshold : itk.ThresholdImageFilter
        new instance of itk.ThresholdImageFilter. As default the instance
        is updated. To not update the instance pecify update=False as kwargs.
    '''

    # infer input image type
    InputType = infer_itk_image_type(image, input_type)

    thr = itk.ThresholdImageFilter[InputType].New()
    _ = thr.SetOutsideValue(outside_value)
    _ = thr.SetInput(image)

    if lower_thr is None:
        logging.debug('Applying threshold above: {}'.format(upper_thr))
        _ = thr.ThresholdAbove(upper_thr)

    elif upper_thr is None:
        logging.debug('Applying threshold below: {}'.format(lower_thr))

        _ = thr.ThresholdBelow(lower_thr)

    else:
        logging.debug('Applying threshold: - Upper threshold {} - Lower\
         threshold {}'.format(upper_thr, lower_thr))

        _ = thr.ThresholdOutside(lower_thr, upper_thr)

    return thr


@update
def itk_binary_threshold(image, lower_thr=0, upper_thr=0, inside_value=1,
                         outside_value=0, input_type=None, output_type=None,
                         **kwargs):
    '''
    Apply a threshold in a specified interval and return a binary image. The
    values outside the inteval are setted to outside_value, the ones inside to
    inside_value.

    Parameters
    ----------
    image: itk.Image
        itk image to process
    lower_thr: PixelType
        lower threshold value
    upper_thr: PixelType
        upper threshold value
    inside_value: PixelType
        value to which set the voxels inside the specified inteval
    outside_value: PixelType
        value to which set the voxels outside the specified inteval
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        input image type. If not specified it is iferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators
    Return
    ------
    thr: itk.BinaryThresholdImageFilter
        New instance of binary threshold filter. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''

    logging.debug('Binary Threshold: -Upper thr: {} - Lower \
    thr: {}'.format(upper_thr, lower_thr))
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    thr = itk.BinaryThresholdImageFilter[InputType, OutputType].New()
    _ = thr.SetInput(image)
    _ = thr.SetLowerThreshold(lower_thr)
    _ = thr.SetUpperThreshold(upper_thr)
    _ = thr.SetInsideValue(inside_value)
    _ = thr.SetOutsideValue(outside_value)

    return thr


#
# Smoothing
#

@update
def itk_median(image, radius=1, input_type=None, output_type=None, **kwargs):
    '''
    Apply a median filter on the input image

    Parameters
    ----------
    image: itk.Image
        ima to apply filter to
    radius: int
        kernel radius
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators
    Results
    -------
    median: itk.MedianImageFilter
        itk.MedianImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''
    logging.debug('Median Filter with Radius : {}'.format(radius))
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    median = itk.MedianImageFilter[InputType, OutputType].New()
    _ = median.SetInput(image)
    _ = median.SetRadius(radius)

    return median


@update
def itk_smoothing_recursive_gaussian(image,
                                     sigma=1.,
                                     normalize_across_scale=False,
                                     input_type=None,
                                     output_type=None,
                                     **kwargs):
    '''
    Computes the smoothing of an image by convolution with the Gaussian kernels

    Parameters
    ----------
    image : itk.Image
        image to smooth
    sigma : float default : 1.
        standard deviation of the gaussian kernel
    normalize_across_scale : bool dafault : False
        specify if normalize the Gaussian over the scale
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    filter : itk.SmoothingRecursiveGaussianImageFilter
    As default the instance is updated.
    To not update the instance pecify update=False as kwargs.
    '''
    logging.debug(f'Smoothing Recursive Gaussian Filter: -sigma: {sigma} -normalize_across_scale: {normalize_across_scale}')    # Retrive image pixel type and dimension

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    smooth = itk.SmoothingRecursiveGaussianImageFilter[InputType, OutputType].New()
    _ = smooth.SetInput(image)
    _ = smooth.SetSigma(sigma)
    _ = smooth.SetNormalizeAcrossScale(normalize_across_scale)

    return smooth
#
# Binary Images
#


@update
def itk_salt_and_pepper_noise(image, salt_value=1, pepper_value=0, prob=.05,
                              input_type=None, output_type=None, **kwargs):
    '''
    Apply Salt and Pepper Noise to an Image

    Parameters
    ----------
    image: itk.Image
        image to apply to SaltAndPepperNoise filter to.
    salt_value: pixel type
        saturated voxel value
    pepper_value: pixel type
        dead voxel value
    prob: float
        probability of noise event
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators
    Results
    -------
    filter: itk.SaltAndPepperNoiseImageFilter
        itk filter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''
    logging.debug(f'Salt and Pepper Noise: Salt Value:\
     {salt_value} Pepper Value: {pepper_value} Noise Probability:\
      {prob}')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    filt = itk.SaltAndPepperNoiseImageFilter[InputType, OutputType].New()
    _ = filt.SetInput(image)
    _ = filt.SetSaltValue(salt_value)
    _ = filt.SetPepperValue(pepper_value)
    _ = filt.SetProbability(prob)

    return filt


@update
def itk_binary_erode(image, radius=1, foreground_value=1,
                     background_value=0, input_type=None, output_type=None,
                     **kwargs):
    '''
    Erode a binary image using a ball kernel of the same dimension of the image
    volume.

    Parameters
    ----------
    image: itk.Image
        binary image to erode
    radius: int
        radius of the ball kernel
    foreground_value: voxel type
        Intensity value to erode
    background_value: voxel type
        Replacement Value
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    erode: itk.BinaryErodeImageFilter
        itk.BinaryErodeImageFilter instance. As default the instance is updated
        To not update the instance pecify update=False as kwargs.
    '''
    # TODO: add a way to chose the kind of structuring element
    _, dimension = itk.template(image)[1]
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    logging.debug('Binary Erosion with a Ball Kernel of \
                    Dimension {} and Radius: {}'.format(dimension, radius))

    StructuringElementType = itk.FlatStructuringElement[dimension]
    structuring_element = StructuringElementType.Ball(radius)

    ErodeFilterType = itk.BinaryErodeImageFilter[InputType, OutputType, StructuringElementType]
    erode = ErodeFilterType.New()
    erode.SetInput(image)
    erode.SetKernel(structuring_element)
    erode.SetForegroundValue(foreground_value)  # Intensity value to erode
    erode.SetBackgroundValue(background_value)  #

    return erode


@update
def itk_binary_dilate(image, radius=1, foreground_value=1,
                      background_value=0, input_type=None, output_type=None,
                      **kwargs):
    '''
    Dilate a binary image using a ball kernel of the same dimension of the
    image volume.

    Parameters
    ----------
    image: itk.Image
        binary image to erode
    radius: int
        radius of the ball kernel
    foreground_value: voxel type
        Intensity value to erode
    background_value: voxel type
        Replacement Value
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    dilate: itk.BinaryErodeImageFilter
        itk.BinaryErodeImageFilter instance. As default the instance is updated
        To not update the instance pecify update=False as kwargs.
    '''
    # TODO: add a way to chose the kind of structuring element
    _, dimension = itk.template(image)[1]
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    logging.debug('Binary Dilation with a Ball Kernel of \
                    Dimension {} and Radius: {}'.format(dimension, radius))

    StructuringElementType = itk.FlatStructuringElement[dimension]
    structuring_element = StructuringElementType.Ball(radius)

    DilateFilterType = itk.BinaryDilateImageFilter[InputType, OutputType, StructuringElementType]
    dilate = DilateFilterType.New()
    dilate.SetInput(image)
    dilate.SetKernel(structuring_element)
    dilate.SetForegroundValue(foreground_value)  # Intensity value to erode
    dilate.SetBackgroundValue(background_value)  #

    return dilate


@update
def itk_binary_morphological_opening(image, radius=1, foreground_value=1,
                                     background_value=0,
                                     input_type=None, output_type=None,
                                     **kwargs):
    '''
    Open a binary image using a ball kernel of the same dimension of the image
    volume.

    Parameters
    ----------
    image: itk.Image
        binary image to erode
    radius: int
        radius of the ball kernel
    foreground_value: voxel type
        Intensity value to erode
    background_value: voxel type
        Replacement Value
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    opening: itk.BinaryMorphologicalOpeningImageFilter
        itk.BinaryMorphologicalOpeningImageFilter instance.
        As default the instance is updated.
        To not update the instance pecify update=False as kwargs.
    '''
    # TODO: add a way to chose the kind of structuring element
    _, dimension = itk.template(image)[1]
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    logging.debug('Binary Morphological Opening with a Ball Kernel of \
                    Dimension {} and Radius: {}'.format(dimension, radius))

    StructuringElementType = itk.FlatStructuringElement[dimension]
    structuring_element = StructuringElementType.Ball(radius)

    OpeningFilterType = itk.BinaryMorphologicalOpeningImageFilter[InputType, OutputType, StructuringElementType]
    opening = OpeningFilterType.New()
    opening.SetInput(image)
    opening.SetKernel(structuring_element)
    opening.SetForegroundValue(foreground_value)  # Intensity value to erode
    opening.SetBackgroundValue(background_value)  #

    return opening


@update
def itk_binary_morphological_closing(image, radius=1, foreground_value=1,
                                     input_type=None, output_type=None):
    '''
    Close a binary image using a ball kernel of the same dimension of the image
    volume.

    Parameters
    ----------
    image: itk.Image
        binary image to erode
    radius: int
        radius of the ball kernel
    foreground_value: voxel type
        Intensity value to erode
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Results
    -------
    closing: itk.BinaryMorphologicalClosingImageFilter
        itk.BinaryMorphologicalOpeningImageFilter instance.
        As default the instance is updated.
        To not update the instance pecify update=False as kwargs.
    '''
    # TODO: add a way to chose the kind of structuring element
    _, dimension = itk.template(image)[1]
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    logging.debug('Binary Morphological Closing with a Ball Kernel of \
                    Dimension {} and Radius: {}'.format(dimension, radius))

    StructuringElementType = itk.FlatStructuringElement[dimension]
    structuring_element = StructuringElementType.Ball(radius)

    ClosingFilterType = itk.BinaryMorphologicalClosingImageFilter[InputType, OutputType, StructuringElementType]
    closing = ClosingFilterType.New()
    closing.SetInput(image)
    closing.SetKernel(structuring_element)
    closing.SetForegroundValue(foreground_value)  # Intensity value to erode

    return closing
#
# Labelling
#


@update
def itk_connected_components(image, fully_connected=False, background_value=0,
                             input_type=None, output_type=None, **kwargs):
    '''
    Label the object of a binary image. Assign a Unique Label to each distinct
    object.

    Parameters
    ----------
    image: itk.Image
        binary image to process
    fully_connected: bool
        Set whether the connected components are defined strictly by face
        connectivity or by face+edge+vertex connectivity
    background_value: voxel type
        Set the pixel intensity to be used for background (non-object)
        regions of the image in the output
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    connected: itk.ConnectedComponentImageFilter
        itk.ConnectedComponentImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''

    logging.debug(f'Computing Connected Components: - fully_connected: {fully_connected} - background_value: {background_value}')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    connected = itk.ConnectedComponentImageFilter[InputType, OutputType].New()
    _ = connected.SetInput(image)
    _ = connected.SetFullyConnected(fully_connected)
    _ = connected.SetBackgroundValue(background_value)

    return connected


@update
def itk_relabel_components(image,
                           sort_by_object_size=True,
                           minimum_object_size=None,
                           number_of_object_to_print=None,
                           input_type=None, output_type=None,
                           **kwargs):
    '''
    Relabel the components in an image such that consecutive labels are used.

    Parameters
    ----------
    image: itk.Image
        label image to relabel
    sort_by_object_size: bool
        specify if sort the object by their size
    minimum_object_size: int
        Set the minimum size in pixels for an object. All objects smaller than
        this size will be discarded and will not appear in the output label map
    number_of_object_to_print: int
        Set the number of objects enumerated and described when the filter is
        printed.
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    relabeler: itk::RelabelComponentImageFilter
        itk::RelabelComponentImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.

    '''

    logging.debug(f'Relabel Components. - Sort by Size: {sort_by_object_size}  \
    - minimum size: {minimum_object_size} - number of objects to print: {number_of_object_to_print}')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    relabeler = itk.RelabelComponentImageFilter[InputType, OutputType].New()
    _ = relabeler.SetInput(image)
    _ = relabeler.SetSortByObjectSize(sort_by_object_size)
    # FIXME: Problem on the input type
    #if minimum_object_size is not None:
    #    _ = relabeler.SetMinimumObjectSize(minimum_object_size)
    #if number_of_object_to_print is not None:
    #    _ = relabeler.SetNumberOfObjectsToPrint(number_of_object_to_print)

    return relabeler

#
# Region Extraction
#


@update
def itk_extract(image, region, collapse2submatrix=False,
                input_type=None, output_type=None):
    '''
    Decrease the image size by cropping the image to the selected region bounds

    Parameters
    ----------
    image: itk.Image
        image to crop
    region: itk.ImageRegion
        cropping region
    collapse2submatrix: bool

    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    filter_ : itk.ExtractImageFilter
        itk.ExtractImageFilter instance. As default the instance is updated.
        To not update the instance pecify update=False as kwargs.
    '''
    # TODO add all the other input options
    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    filter_ = itk.ExtractImageFilter[InputType, OutputType].New()

    if collapse2submatrix:
        _ = filter_.SetDirectionCollapseToSubmatrix()

    _ = filter_.SetInput(image)
    _ = filter_.SetExtractionRegion(region)

    return filter_


#
# Evaluation
#


@update
def itk_label_overlap_measures(source, target,
                               input_type=None, **kwargs):
    '''
    Compute overalpping measures between a teaget and a source images. The
    measures are:
        - Dice Coefficient
        - Intersection Over Union
        - Volume Similarity
        - Mean Overlap
        - Union Overlap
        - False Positive Error
        - False Negative Error
    Parameters
    ----------
    source: itk.Image
        source binary image
    target: itk.Image
        target binary image
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         source and target image type. If not specified it is inferred from the
         source image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    measures: itk.LabelOverlapMeasuresImageFilter
        itk.LabelOverlapMeasuresImageFilter instance. As default the instance
        is updated. To not update the instance pecify update=False as kwargs.
    '''
    InputType = infer_itk_image_type(source, input_type)

    measures = itk.LabelOverlapMeasuresImageFilter[InputType].New()
    _ = measures.SetSourceImage(source)
    _ = measures.SetTargetImage(target)

    return measures


@update
def itk_hausdorff_distance(image1, image2,
                           input1_type=None, input2_type=None, **kwargs):
    '''
    '''
    Input1Type = infer_itk_image_type(image1, input1_type)
    Input2Type = infer_itk_image_type(image2, input2_type)

    hd = itk.HausdorffDistanceImageFilter[Input1Type, Input2Type].New()
    _ = hd.SetInput1(image1)
    _ = hd.SetInput2(image2)

    return hd


#
# Variation and Eigenvalues Based Filters
#


@update
def itk_hessian_recursive_gaussian(image, sigma=1.,
                                   normalize_across_scale=False,
                                   input_type=None, **kwargs):
    '''
    Computes the Hessian matrix of an image by convolution with the Second and
    Cross derivatives of a Gaussian.

    Parameters
    ----------
    image : itk.Image
        image to process
    sigma : float Default: 1.
        standard deviation of the gaussian kernel
    normalize_across_scale : Bool Default: False
        specify if normalize the Gaussian over the scale
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image

    Result
    ------
    hessian : itk.HessianRecursiveGaussianImageFilter
        As default the instance is updated. To not update the instance pecify
        update=False as kwargs.
    '''
    logging.debug(f'Hessian Recursive Gaussian Filter: - sigma: {sigma} - normalize_across_scale: {normalize_across_scale}')
    InputType = infer_itk_image_type(image, input_type)

    hessian = itk.HessianRecursiveGaussianImageFilter[InputType].New()
    _ = hessian.SetInput(image)
    _ = hessian.SetSigma(sigma)
    _ = hessian.SetNormalizeAcrossScale(normalize_across_scale)

    return hessian


@update
def itk_symmetric_eigen_analysis(hessian, dimensions=3,
                                 order_eigenvalues_by=2, **kwargs):
    '''
    omputes the eigen-values of every input symmetric matrix pixel.

    Parameters
    ----------
    hessian :
        hessian matrix from which compute the eigen values

    dimesions: int Default: 3

    order : int Default 2
        specify the rule to use for sorting the eigenvalues:
            1 ascending order
            2 magnitude ascending order
            3 no order
    Return
    ------

    eigen : itk.SymmetricEigenAnalysisImageFilter
        itk.SymmetricEigenAnalysisImageFilter instance. As default the instance
        is updated. To not update the instance pecify update=False as kwargs.
    '''
    logging.debug(f'Symmetric Eigen Analysis: - dimensions: {dimensions} - order_eigenvalues_by: {order_eigenvalues_by}')
    # filter declaration and new obj memory allocation (using New)

    eigen = itk.SymmetricEigenAnalysisImageFilter[type(hessian)].New()
    # seting of the dedidred arguments with the specified ones

    _ = eigen.SetInput(hessian)
    _ = eigen.SetDimension(dimensions)
    _ = eigen.OrderEigenValuesBy(order_eigenvalues_by)

    return eigen


#
# Image Orientation
#

# define forcing orientations

# force orientation


#
# Image Physical Infoirmations
#

@update
def itk_change_information_from_reference(image,
                                          reference_image,
                                          change_direction=True,
                                          change_origin=True,
                                          change_spacing=True,
                                          change_region=True,
                                          input_type=None, **kwargs):
    '''
    Change the origin, spacing, direction and/or buffered region of an itkImage
    to the one of the specified reference image.

    Parameters
    ----------
    image: itk.Image
        input image
    reference_image: itk.Image
        reference image -> will be casted to the same type of input image
    change_direction: Bool
        Specify if change input image direction (default: True)
    change_origin: Bool
        Specify if change input image origin (default: True)
    change_spacing: Bool
        Specify if change input image spacing (default: True)
    change_region: Bool
        Specify if change input image region (default: True)
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    cahnger: itk.ChangeInformationImageFilter
        filter is updated by default.
        To not update the instance pecify update=False as kwargs.
    '''
    InputType = infer_itk_image_type(image, input_type)
    reference_image = itk.CastImageFilter[type(reference_image), InputType].New(reference_image)
    _ = reference_image.Update()
    changer = itk.ChangeInformationImageFilter[InputType].New()
    _ = changer.SetUseReferenceImage(True)
    _ = changer.SetInput(image)
    _ = changer.SetReferenceImage(reference_image.GetOutput())
    _ = changer.SetChangeDirection(change_direction)
    _ = changer.SetChangeOrigin(change_origin)
    _ = changer.SetChangeSpacing(change_spacing)
    _ = changer.SetChangeRegion(change_region)
    _ = changer.SetCenterImage(False)

    return changer
