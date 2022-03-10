import itk
import logging
from decorators import update
__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def infer_itk_image_type(image, desidered_type=None):
    logging.debug('Inferring image type')
    
    pass

@update
def itk_orient_image_to_axial(image):
    """
    Change the image orientation to axial one, closer to RIS
    """

    logging.debug('Orienting image to axia(RIS)')
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    orienter = itk.OrientImageFilter[ImageType, ImageType].New()

    _ = orienter.UseImageDirectionOn()
    _ = orienter.SetDesiredCoordinateOrientationToAxial()
    _ = orienter.SetInput(image)

    return orienter
