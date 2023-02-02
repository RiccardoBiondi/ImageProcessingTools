import itk
import logging
import functools


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['update', 'slice_by_slice', 'timing']

#
# Some useful internal functions
#


def _extract(image, region):
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    OutputType = itk.Image[PixelType, Dimension - 1]

    extraction = itk.ExtractImageFilter[ImageType, OutputType].New()

    _ = extraction.SetDirectionCollapseToSubmatrix()
    _ = extraction.SetInput(image)
    _ = extraction.SetExtractionRegion(region)
    _ = extraction.Update()

    return extraction


def _join_series(image_series, spacing=1., origin=0., direction=None):

    PixelType, Dimension = itk.template(image_series[0])[1]
    ImageType = itk.Image[PixelType, Dimension]
    OutputType = itk.Image[PixelType, Dimension + 1]

    joiner = itk.JoinSeriesImageFilter[ImageType, OutputType].New()
    _ = joiner.SetSpacing(spacing)
    _ = joiner.SetOrigin(origin)

    for i, im in enumerate(image_series):
        _ = joiner.SetInput(i, im)

    return joiner


def update(func):
    """
    Decorator to automatically update an itk pipeline. The pipeline must be
    initlaized with the input/s images as *args and other as kwargs.
    The pipeline must return an itk filter, not an image.

    To deactivate the usage of the decorator, simply specify: upadte=False
    as kwargs in the function.

    Example
    -------
    >>> import itk
    >>> from ipt.decorators import update
    >>>
    >>> # Create a decorated function containing the pipeline to update
    >>>
    >>> @update
    >>> def pipeline(image, radius=1, **kwargs):
    >>>   median_filter = itk.MedianImageFilter[type(image), type(image)].New()
    >>>   _ = median_filter.SetInput(image)
    >>>   _ = median_filter.SetRadius(radius)
    >>>
    >>>   return median_filter
    >>>
    >>> def main():
    >>>
    >>>   image = itk.imread('path/to/input/image')
    >>>   filtered = median_filter(image)
    >>>   _ = itk.imwrite('path/to/output', filtered.GetOutput())
    >>>
    >>> if __name__ == '__main__':
    >>>   main()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        pipeline = func(*args, **kwargs)

        if kwargs.get('update', True):
            logging.debug('Updating {}'.format(func.__name__))

            _ = pipeline.Update()

        return pipeline
    return wrapper


def slice_by_slice(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        if kwargs.get('slice_by_slice', False):

            logging.debug('Apply the filter slice by slice')
            # qui comincia l'implementazione di slice by slice
            # assume that the first passed argument is the input image
            region = args[0].GetBufferedRegion()
            # TODO: allow the user to choose the index
            nslices = region.GetSize()[2]
            start = region.GetIndex()  # starting indexes
            size = region.GetSize()  # region size
            size[2] = 0  # because I want to collapse a 3d image to 2d
            region.SetSize(size)
            res = []  # to store the single slice results
            # now process each slice separately
            for s in range(nslices):
                start[2] = s
                region.SetIndex(start)

                imgs = [_extract(i, region).GetOutput() for i in args]
                im = func(*imgs, **kwargs)
                _ = im.Update()

                if kwargs.get('get_output', True):
                    _ = res.append(im.GetOutput())
                else:
                    _ = res.append(im)
            # now join the results -> the joining image filter, not its output!
            return _join_series(res, args[0].GetSpacing()[2], args[0].GetOrigin()[2])

        else:
            return func(*args, **kwargs)
    return wrapper


def timing(func):
    """
    Perform a time benchmark of the decorated pipeline.
    The whole pipeline must be wrapped inside a function, and the update must
    be executed before perform the timing(TODO find a better expression)
    """
    logging.debug(f'Timing for {func.__name__}')

    @functools.wrap(func)
    def wrapper(*args, **kwargs):

        return None
    return wrapper
