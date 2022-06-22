import os
import itk
import hypothesis.strategies as st

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

"""
This file contains the implementation of custom hypothesis strategies to use
during the property based test. I have implemented here all the strategies to
avoid redundant code
"""

legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                 min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)

medical_image_format = ['nrrd', 'nii']
pixel_types = [itk.UC, itk.SS]
itk_types = [itk.UC, itk.UC, itk.SS, itk.F, itk.D]

@st.composite
def random_image_strategy(draw):
    '''
    Generate a White Noise Image
    '''

    # TODO add also a random generation of spatial information:
    # - direction - origin - spacing
    PixelType = draw(st.sampled_from(pixel_types))

    ImageType = itk.Image[PixelType, 3]

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(200)
    rndImage.Update()

    return rndImage.GetOutput()


@st.composite
def path2image_strategy(draw):
    '''
    Generate a valid filename
    '''
    image_name = draw(text_strategy)
    image_format = draw(st.sampled_from(medical_image_format))

    # where I am
    cwd = os.getcwd()

    path = '{}/test/test_images/{}.{}'.format(cwd, image_name, image_format)

    return path


@st.composite
def image_type_strategy(draw):
    '''
    Generate an itk image type obj
    '''
    dimension = draw(st.integers(2, 3))
    pixel_type = draw(st.sampled_from(pixel_types))

    return itk.Image[pixel_type, dimension]
