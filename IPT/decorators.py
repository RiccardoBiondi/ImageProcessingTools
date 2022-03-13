import logging
import functools


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['update', 'slice_by_slice', 'timing']


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
    """
    """

    logging.debug(f"Iterate Slice by Slice {func.__name__}")
    pass


def timing(func):
    """
    Perform a time benchmark of the decorated pipeline.
    The whole pipeline must be wrapped inside a function, and the update must
    be executed before perform the timing(TODO find a better expression)
    """
    logging.debug(f'Timing for {func.__name__}')

    @functools.wrap(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
