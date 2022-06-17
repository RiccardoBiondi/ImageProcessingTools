import itk
import logging
import numpy as np
import matplotlib.pyplot as plt

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = ['plt_view', 'plt_image_histogram']


def plt_view(image, idx=(0, 0, 0), label=None, cmap='gray',
             l_cmap='cool', contours='red'):
    '''
    '''
    # first of sll ore the RIS orientation
    if isinstance(image, np.ndarray):
        im = image.copy()
        spacing = (1., 1., 1.)
    else:
        # TODO: force the orthogonal direction
        im = itk.GetArrayFromImage(image)
        # Get the spacing uset to compute the aspect ratio
        spacing = image.GetSacing()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))


def plt_image_histogram():
    pass
