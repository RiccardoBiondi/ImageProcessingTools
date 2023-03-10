import itk
import logging
# import numpy as np

from typing import Optional, List, NewType


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
           'itk_change_information_from_reference',
           'itk_voting_binary_iterative_hole_filling', 'itk_cast']

Image = NewType('Image', itk.Image)

#
# Aritmetic Operators
#


@update
def itk_add(
            image1: Image, image2: Image,
            input1_type: Optional[Image] = None,
            input2_type: Optional[Image] = None,
            output_type: Optional[Image] = None, **kwargs
            ) -> itk.AddImageFilter:
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
def itk_subtract(
                image1: Image, image2: Image,
                input1_type: Optional[Image] = None,
                input2_type: Optional[Image] = None,
                output_type: Optional[Image] = None, **kwargs
                ) -> itk.SubtractImageFilter:
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
def itk_multiply(
                 image1: Image, image2: Image,
                 input1_type: Optional[Image] = None,
                 input2_type: Optional[Image] = None,
                 output_type: Optional[Image] = None, **kwargs
                 ) -> itk.MultiplyImageFilter:
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
def itk_invert_intensity(
                         image: Image, 
                         maximum: int = 1,
                         input_type: Optional[Image] = None,
                         output_type: Optional[Image] = None, **kwargs
                         ):
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
def itk_maximum(
                image1: Image, image2: Image,
                image1_type: Optional[Image] = None,
                image2_type: Optional[Image] = None,
                output_type: Optional[Image] = None, **kwargs
                ) -> itk.MaximumImageFilter:
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
def itk_shift_scale(
                    image: Image,
                    shift: float = 0.,
                    scale: float = 1.,
                    input_type: Optional[Image] = None,
                    output_type: Optional[Image] = None,
                    **kwargs) -> itk.ShiftScaleImageFilter:
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
def itk_gaussian_normalization(
                               image: Image,
                               mask: Image,
                               label: int = 1,
                               input_type: Optional[Image] = None,
                               output_type: Optional[Image] = None,
                               **kwargs) -> itk.ShiftScaleImageFilter:
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

    normalized: itk.ShiftScaleImageFilter
        Filter with the image normalized according to the mean and standard
        deviation computed inside the specified mask.  As default the instance
        is updated. To not update the instance specify update=False as kwargs.
    '''

    logging.debug(f'Running Gaussian Normalization. ROI label={label}')

    stats = itk_label_statistics(image, mask,
                                 input_type, update=True)

    # TODO add standard values for the case in which the label filter is not
    # updated?? mbah
    shift = -stats.GetMean(label)
    scale = 1. / abs(stats.GetSigma(label))

    normalized = itk_shift_scale(image, shift=shift, scale=scale,
                                 input_type=input_type,
                                 output_type=output_type,
                                 update=kwargs.get('update', True))

    return normalized



@update
def itk_unsharp_mask(
                    image: Image,
                    sigmas: float = 1.0,
                    amount: float = .5,
                    threshold: float = 0.,
                    clamp: bool = True,
                    input_type: Optional[Image] = None,
                    output_type: Optional[Image] = None,
                    **kwargs) -> itk.UnsharpMaskImageFilter:

    '''
    Unsharp mask edge enhancement filter.
    This filter subtracts a smoothed version of the image from the image
    to achieve the edge enhancing effect.

    Parameters
    ----------
    image: itk.Image
        image to unsharp


    Return
    ------

    unsharped: itk.UnsharpMaskImageFilter
            Filter with the edge enhanced by the unsharp masking.
            As default the instance is updated. To not update the
            instance specify update=False as kwargs.
    '''

    logging.debug(f'Running Unsharp Masking.')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    unsharped = itk.UnsharpMaskImageFilter[InputType, OutputType].New()
    _ = unsharped.SetInput(image)
    _ = unsharped.SetSigmas(sigmas)
    _ = unsharped.SetAmount(amount)
    _ = unsharped.SetThreshold(threshold)
    _ = unsharped.SetClamp(clamp)


    return unsharped

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
def itk_threshold(image: Image,
                  upper_thr: Optional[float] = None, # TODO: change the type to pixel type
                  lower_thr: Optional[float] = None, # TODO: change the type to pixel type
                  outside_value: float = 0, # TODO: change the type to pixel type
                  input_type: Optional[Image] = None,
                  **kwargs) -> itk.ThresholdImageFilter:
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
def itk_binary_threshold(
                        image,
                        lower_thr=0,
                        upper_thr=0,
                        inside_value=1,
                         outside_value=0,
                         input_type=None,
                         output_type=None,
                         **kwargs) -> itk.BinaryThresholdImageFilter:
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

    logging.debug(f'Binary Threshold: -Upper thr: {upper_thr} - Lower thr: {lower_thr}')
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
    logging.debug(f'Median Filter with Radius : {radius}')
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
    logging.debug(f'Smoothing Recursive Gaussian Filter: -sigma: {sigma} -normalize_across_scale: {normalize_across_scale}') 

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

    logging.debug(f'Binary Erosion with a Ball Kernel of \
                    Dimension {dimension} and Radius: {radius}')

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

    logging.debug(f'Binary Dilation with a Ball Kernel of \
                    Dimension {dimension} and Radius: {radius}')

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

    logging.debug(f'Binary Morphological Opening with a Ball Kernel of \
                    Dimension {dimension} and Radius: {radius}')

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

    logging.debug(f'Binary Morphological Closing with a Ball Kernel of \
                    Dimension {dimension} and Radius: {radius}')

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
    logging.debug(f'Extract image filter: collapse to submatrix: {collapse2submatrix}')

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

    logging.debug(f'Label Overlap Measure')
    InputType = infer_itk_image_type(source, input_type)

    measures = itk.LabelOverlapMeasuresImageFilter[InputType].New()
    _ = measures.SetSourceImage(source)
    _ = measures.SetTargetImage(target)

    return measures


@update
def itk_hausdorff_distance(image1, image2,
                           input1_type=None, input2_type=None, **kwargs):
    '''
    Compute the hausdorff distance filteer between two images
    '''

    logging.debug("Hausdorff Distance")
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
    _ = changer.SetCenterImage(False)
    _ = changer.SetInput(image)
    _ = changer.SetReferenceImage(reference_image.GetOutput())
    _ = changer.SetChangeDirection(change_direction)
    _ = changer.SetChangeOrigin(change_origin)
    _ = changer.SetChangeSpacing(change_spacing)
    _ = changer.SetChangeRegion(change_region)

    return changer


@update
def itk_voting_binary_iterative_hole_filling(image,
                                             radius=1,
                                             max_number_of_iterations=10,
                                             majority_threshold=1,
                                             foreground_value=1,
                                             background_value=0,
                                             input_type=None, **kwargs):
    '''
    Fills in holes and cavities by iteratively applying a voting operation.

    Parameters
    ----------
    image: itk.Image
        input binary image
    radius: int or list of int
        radius of the neighborhood used to compute the median
    max_number_of_iterations: int
        maximum number of iterations to perform
    majority_threshold: int
        number of pixels over 50% that will decide whether an OFF pixel will
        become ON or not
    foreground_value: int
        value associated with the Foreground(object) of the binary image
    background_value: int
        value associated with the Background of the binary image
    input_type: itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    cahnger: itk.VotingBinaryIterativeHoleFillingImageFilter
        filter is updated by default.
        To not update the instance pecify update=False as kwargs.
    '''
    logging.debug(f'Voting Binary Hole filling: \
    majority_threshold={majority_threshold},\
     max_number_of_iterations={max_number_of_iterations},\
      foreground_value={foreground_value},\
      background_value={background_value},\
      radius={radius}')

    InputType = infer_itk_image_type(image, input_type)
    filter_ = itk.VotingBinaryIterativeHoleFillingImageFilter[InputType].New()

    _ = filter_.SetInput(image)
    _ = filter_.SetMajorityThreshold(majority_threshold)
    _ = filter_.SetMaximumNumberOfIterations(max_number_of_iterations)
    _ = filter_.SetForegroundValue(foreground_value)
    _ = filter_.SetBackgroundValue(background_value)
    _ = filter_.SetRadius(radius)

    return filter_


@update
def itk_cast(image, new_type=itk.UC, **kwargs):
    '''
    Cast image voxel type to new_type. Preserve image dimensions

    Parameters
    ----------
    image: itk.Image
        Image to cast
    new_type: itk voxel type (i.e. itk.UC)
        new voxel type
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    cast: itk.CastImageFilter
        filter is updated by default.
        To not update the instance pecify update=False as kwargs.
    '''

    pixel_type, dimension = itk.template(image)[1]
    logging.debug(f'Casting image from {pixel_type} to {new_type}')
    InputType = itk.Image[pixel_type, dimension]
    OutputType = itk.Image[new_type, dimension]

    cast = itk.CastImageFilter[InputType, OutputType].New()
    _ = cast.SetInput(image)

    return cast

"""
@update
def itk_flip_image(
                    image: itk.Image,
                    flip_axes: tuple = (True, False),
                    input_type=None,
                    **kwargs) -> itk.FlipImageFilter:
    '''
    Description...

    Parameters
    ----------
    image: itk.Image
        Image to flip
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    flipped: itk.FlipImageFilter
        filter is updated by default.
        To not update the instance pecify update=False as kwargs.
    '''
    ImageType = infer_itk_image_type(image, input_type)

    flipped = itk.FlipImageFilter[ImageType].New()
    _ = flipped.SetInput(image)
    _ = flipped.SetFlipAxes(flip_axes)

    return flipped
"""

@update
def itk_sigmoid(image, alpha=1., beta=0., output_min=0.0, output_max=1., **kwargs):
    '''
    Computes the sigmoid function pixel-wise.

    A linear transformation is applied first on the argument of the sigmoid
    function.
    :math:`(output_max - output_min) * \\frac{1}{1 - e^{-\\frac{X - \\beta}{\\alpha}}} + output_min`

    '''
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    SigmoidFilterType = itk.SigmoidImageFilter[ImageType, ImageType].New()
    sigmoid = SigmoidFilterType.New()

    _ = sigmoid.SetOutputMinimum(output_min)
    _ = sigmoid.SetOutputMaximum(output_max)
    _ = sigmoid.SetAlpha(alpha)
    _ = sigmoid.SetBeta(beta)
    _ = sigmoid.SetInput(image)

    return sigmoid


@update
def itk_geodesic_active_contour(input_image,
                                feature_image,
                                propagation_scaling=1.,
                                curvature_scaling=1.,
                                advection_scaling=1.0,
                                max_RMS_error=0.02,
                                number_of_iterations=1, **kwargs):
    '''
    '''
    PixelType, Dimension = itk.template(input_image)[1]
    ImageType = itk.Image[PixelType, Dimension]
    geodesic_ac = itk.GeodesicActiveContourLevelSetImageFilter[ImageType,
                                                               ImageType,
                                                               itk.F].New()
    _ = geodesic_ac.SetPropagationScaling(propagation_scaling)
    _ = geodesic_ac.SetCurvatureScaling(curvature_scaling)
    _ = geodesic_ac.SetAdvectionScaling(advection_scaling)
    _ = geodesic_ac.SetMaximumRMSError(max_RMS_error)
    _ = geodesic_ac.SetNumberOfIterations(number_of_iterations)
    _ = geodesic_ac.SetInput(input_image)
    _ = geodesic_ac.SetFeatureImage(feature_image)

    return geodesic_ac


@update
def itk_signed_maurer_distance_map(image, inside_positive=False, squared_distance=False, **kwargs):

    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]
    OutputType = itk.Image[itk.F, Dimension]
    dist = itk.SignedMaurerDistanceMapImageFilter[ImageType, OutputType].New()
    _ = dist.SetInput(image)
    _ = dist.SetInsideIsPositive(inside_positive)
    _ = dist.SetSquaredDistance(squared_distance)

    return dist

@update
def itk_image_gradient_magnitude_recursive(image, sigma, **kwargs):

    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    gm = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(
    )
    _ = gm.SetInput(image)
    _ = gm.SetSigma(sigma)

    return gm

#
# Logical Operations
#

@update
def itk_or(image1, image2, input1_type=None, input2_type=None, output_type=None, **kwargs):
    '''
    '''
    Input1Type = infer_itk_image_type(image1, input1_type)
    Input2Type = infer_itk_image_type(image2, input2_type)
    OutputType = infer_itk_image_type(image1, output_type)

    or_ = itk.OrImageFilter[Input1Type, Input2Type, OutputType].New()
    _ = or_.SetInput(0, image1)
    _ = or_.SetInput(1, image2)

    return or_


