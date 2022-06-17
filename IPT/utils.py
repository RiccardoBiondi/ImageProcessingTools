import itk
import logging
from IPT.decorators import update

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def infer_itk_image_type(image, desidered_type=None):
    '''
    Infer the desidered image type: if default type is None, will return the
    type of the specified image, otherwise will return desidered_type
s
    Parameters
    ----------
    image: itk.Image
        itk Image from which infer the type

    desidered_type: itk ImageType
        type to return instead of the one of image. Default: None

    Return
    ------
    image_type: itk Image type (i.e. itk.Image[itk.UC, 2])
        inferred image type
    '''
    logging.debug('Inferring image type')

    if desidered_type is not None:
        return desidered_type

    pixel_type, dimension = itk.template(image)[1]
    image_type = itk.Image[pixel_type, dimension]

    return image_type


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


def itk_constant_image_from_reference(reference_image, value=0):
    '''
    Create an image with constant voxel value. The image is created starting
    from a reference image

    Parameters
    ----------
    reference_image: itk.Image
        reference image: the output image will match the reference on all the
        physical properties.
    value: PixelType
        constant value of the image

    Return
    ------
    const : itk.Image
        constant image
    '''
    PixelType, Dimension = itk.template(reference_image)[1]
    const = itk.Image[PixelType, Dimension].New()
    _ = const.SetRegions(reference_image.GetLargestPossibleRegion())
    _ = const.Allocate()
    _ = const.FillBuffer(value)

    _ = const.SetSpacing(reference_image.GetSpacing())
    _ = const.SetDirection(reference_image.GetDirection())
    _ = const.SetOrigin(reference_image.GetOrigin())

    return const
