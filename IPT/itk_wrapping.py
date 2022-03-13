import itk
import logging
# import numpy as np

from IPT.decorators import update
from IPT.utils import infer_itk_image_type


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['itk_add', 'itk_subtract', 'itk_multiply', 'itk_mask',
           'itk_mask_volume', 'itk_median']

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
         input1 image type. If not specified it is inferred from the input image1
    input2_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input2 image type. If not specified it is inferred from the input image2
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image2

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
         input1 image type. If not specified it is inferred from the input image1
    input2_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input2 image type. If not specified it is inferred from the input image2
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image2

    Results
    -------
    multiply_image: itk.MultiplyImageFilter
        itk.MultiplyImageFilter instance.  As default the instance is updated.
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


#
# Algebric Operations
#

#
# Image Functions
#

@update
def itk_mask():
    """
    """
    logging.debug("")
    pass


@update
def itk_mask_volume():
    """
    """
    pass

#
# Smoothing
#


@update
def itk_median(image, radius=1, image_type=None, output_type=None):
    """
    """
    logging.debug("")
    pass

#
# Binary Images
#
