import os
import itk
import logging

from IPT.decorators import update

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['itk_image_file_reader', 'itk_image_file_writer']

@update
def itk_image_file_reader(filename: str, image_type: itk.Image, **kwargs):
    '''
    Read a medical image in a format supported by itk.
    Create a new instance of itk.ImageFileReader.

    Parameters
    ----------
    filename : str
        input filename
    image_type : itk.Image
        desidered input image type
    Return
    ------
        reader: itk.ImageFileReader
            new and initialized instancce of itk.ImagerReader.
            The instance is updated by default
    '''

    if os.path.exists(filename):

        logging.debug(f"Reading image from: {filename}")

        reader = itk.ImageFileReader[image_type].New()
        _ = reader.SetFileName(filename)

        return reader

    else:
        logging.error(f'The specified path: {filename} does not exists')

        return None


@update
def itk_image_file_writer(filename: str, image: itk.Image,
                          create=False, **kwargs):
    '''
    Create a New instance of itk.ImageWriter, The writer is only initialized
    but not updated. The writer type is inferred from the image

    Parameters
    ----------
    filename : str
        output filename(i.e. ./path/to/my_image.nii)
    image: itk.Image
        image to write.

    Return
    ------
    writer: itk.ImageWriter
        new and initialized instancce of itk.ImagerWriter. The writer is NOT
        updated
    '''
    logging.debug("Writing Image to: {}".format(filename))

    # first of all infer the image type
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    writer = itk.ImageFileWriter[ImageType].New()
    _ = writer.SetInput(image)
    _ = writer.SetFileName(filename)

    return writer


@update
def itk_dicom_series_reader(foldername: str, image_type,
                            keep='largest', **kwargs):
    pass
