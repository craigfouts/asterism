Examples
========

.. toctree::

Usage
-----

.. code-block:: python

    from asterism import ATLAS
    from asterism.utils.data import make_dataset
    from asterism.utils.plots import show_comparison

    data, locs, labels = make_dataset(wiggle=.2, mix=.2, return_tensor=True, seed=0)
    topics = ATLAS(seed=0).fit_predict(data, locs, labels)
    show_comparison(locs, labels, topics)

.. image:: ../assets/images/atlas.png
