.. _install:

Installation
==========================

This part of the documentation covers the installation of Decanter AI Core SDK.


Python Version
---------------

We recommend using the latest version of Python. Decanter AI Core SDK supports
Python 3.7 and newer.


Dependencies
-------------

These distributions will be installed automatically when installing Decanter
AI Core SDK.

*   `requests`_ is an elegant and simple HTTP library for Python, handles apis
    requests and responses.
*   `pandas`_ is a powerful data analysis and manipulation tool, used to better
    display data and handle data format in Decanter AI Core SDK.
*   `matplotlib`_ is a library for creating visualizations in Python, used to
    plot chart with informations.
*   `tqdm`_ decorate an iterable object, provides the progress bar to show
    decanter's progress.
*   `pyzipper`_ read and write AES encrypted zip files, handles the model
    downloading.
*   `numpy`_ the fundamental package for scientific computing with Python,
    support plotting in Decanter AI Core SDK.

.. _requests: https://requests.readthedocs.io/en/master/#
.. _pandas: https://pandas.pydata.org/
.. _matplotlib: https://matplotlib.org/
.. _tqdm: https://tqdm.github.io/docs/tqdm/
.. _pyzipper: https://palletsprojects.com/p/click/
.. _numpy: https://numpy.org/


Virtual environments
---------------------

Use a virtual environment to manage the dependencies for your project,
both in development and in production.


.. _install-create-env:

Create an environment
~~~~~~~~~~~~~~~~~~~~~

Use a virtual environment to manage the dependencies for your project,
both in development and in production.

Install `Anaconda`_ or `Virtualenv`_.

Conda

.. code-block:: sh

    $ conda create -n myenv python=3.7
    $ conda activate myenv

.. _Anaconda: https://www.anaconda.com/products/individual#macos
.. _Virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Vitualenv

.. code-block:: sh

    $ virtualenv myenv
    $ source myenv/bin/activate


Install Decanter AI Core SDK
------------------------------

Within the activated environment, use the following command to install
Decanter AI Core SDK:

.. code-block:: sh

    $ pip install decanter-ai-core-sdk

Decanter AI Core SDK is now installed. Check out the :doc:`quickstart` or go
to the :doc:`Documentation Overview </index>`.
