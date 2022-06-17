#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import itk
import numpy as np
##import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck as HC

import strategies as cst  # testing strategies

from IPT.utils import itk_constant_image_from_reference

from IPT.io import itk_image_file_reader
from IPT.io import itk_image_file_writer


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']



@given(cst.path2image_strategy(), cst.random_image_strategy())
@settings(max_examples=20, deadline=None,
          suppress_health_check=(HC.too_slow, ))
def test_read_and_write_image_file(filename, image):
    '''
    Given:
        - valid filename
        - random image
    Then:
        - write the image
        - read the image
    Assert:
        - red image is equal to the input one
    '''

    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    _ = itk_image_file_writer(filename=filename, image=image)
    reader = itk_image_file_reader(filename=filename, image_type=ImageType)

    out = reader.GetOutput()

    out_array = itk.GetArrayFromImage(out)
    in_array = itk.GetArrayFromImage(image)

    _ = os.remove(filename)

    # TODO add more assertion -> also spatial informtion must be preserved

    assert np.all(out_array == in_array)
    assert out.GetSpacing() == image.GetSpacing()
    assert out.GetOrigin() == image.GetOrigin()
    assert out.GetDirection() == image.GetDirection()
