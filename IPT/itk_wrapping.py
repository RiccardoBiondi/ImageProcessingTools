# import itk
import logging
# import numpy as np

from IPT.decorators import update


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['itk_add', 'itk_subtract', 'itk_multiply', 'itk_mask',
           'itk_mask_volume', 'itk_median']

#
# Aritmetic Operators
#


@update
def itk_add(image1, image2,
            image1_type=None, image2_type=None, output_type=None):
    """
    """
    logging.debug("")
    pass


@update
def itk_subtract(image1, image2,
                 image1_type=None, image2_type=None, output_type=None):
    """
    """
    logging.debug("")
    pass


@update
def itk_multiply(image1, image2,
                 image1_type=None, image2_type=None, output_type=None):
    """
    """
    pass


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
