"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import numpy as np
import os
import pyro
import random
import torch

def set_seed(seed=9):
    """Sets a fixed environment-wide random state.
    
    Parameters
    ----------
    seed : int, default=9
        Random state seed.

    Returns
    -------
    None
    """

    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        pyro.set_rng_seed(seed)

def itemize(length, *items):
    """Converts each item to a collection of the specified length, truncating if 
    the item is a tuple or list of longer length and repeating the last value if 
    the item is a tuple or list of shorter length.
    
    Parameters
    ----------
    length : int
        Length of item list.
    items : any
        Items to convert to lists.

    Yields
    ------
    tuple | list
        Collection of items.
    """

    for item in items:
        if isinstance(item, (tuple, list)):
            yield item[:length] + item[-1:]*(length - len(item))
        else:
            yield [item,]*length
        