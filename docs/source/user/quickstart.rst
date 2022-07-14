.. _quickstart:

Quickstart
==========

This page gives a good introduction to Decanter AI Core SDK. It assumes you
already have Decanter AI Core SDK installed. Follow :doc:`install` to set up
a project and install Decanter AI Core SDK first.


Clone the repository from GitLab:

.. code-block:: sh

    $ git clone https://github.com/MoBagel/decanter-ai-core-sdk.git
    $ cd decanter-ai-core-sdk


.. _python:

Python Script
--------------

Set the username, password, and host at function ``core.Context.create()``
in file ``examples/example.py``

.. code-block:: python

    core.Context.create(
        username='{usr}', password='{pwd}', host='{http://host:port}')


Run the command below: sh

.. code-block:: sh

    $ python -m examples.example


.. _jupyter:

Jupyter
---------

Jupyter Notebook
~~~~~~~~~~~~~~~~~

Install `Jupyter Notebook <https://jupyter.readthedocs.io/en/latest/install.html>`_.

.. code-block:: sh

    $ pip install jupyter notebook


Install `ipywidgets` for progress bar.

.. code-block:: sh

    $ pip install ipywidgets
    $ jupyter nbextension enable --py widgetsnbextension


Add virtual environment to Jupyter Notebook. Make sure ipykernel is
installed in the virtual environment.

.. code-block:: sh

    $ pip install --user ipykernel
    $ python -m ipykernel install --user --name=myenv
    # following output
    # Installed kernelspec myenv in
    # /home/user/.local/share/jupyter/kernels/myenv


Open jupyter notebook

.. code-block:: sh

    $ jupyter notebook


Open the notebook in ``examples/example.ipynb``, and select the kernel
of your environment in the above tool bar `Kernal/Change kernel/myenv`

Jupyter Lab
~~~~~~~~~~~~~~~~~

Install `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`_.


.. code-block:: sh

    $ pip install jupyterlab


Install extension for
`Progress Bar <https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension>`_


.. code-block:: sh

    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager


Open Jupyter Lab

.. code-block:: sh

    $ jupyter lab

