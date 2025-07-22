from itertools import chain
from collections.abc import Iterable


def recursive_flatten(iterable):
    """
    Recursively flattens an iterable, yielding each item in a single flat sequence.
    
    Args:
        iterable (Iterable): The iterable to flatten, which can contain nested iterables.
        
    Yields:
        Each item from the iterable, flattened into a single sequence.
    """
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from recursive_flatten(item)
        else:
            yield item
def flatten_series(series):
    """
    Flattens a series of nested iterables into a single flat list.
    
    Args:
        series (Iterable): The iterable to flatten, which can contain nested iterables.
        
    Returns:
        list: A flat list containing all items from the input iterable.
    """
    return recursive_flatten(series.dropna().tolist())



