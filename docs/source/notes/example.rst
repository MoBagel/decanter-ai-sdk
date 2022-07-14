.. _example:

Example
~~~~~~~~~~~~~~~~~~~~~~~~

Upload, Setup, Train, Predict
*****************************

``Import Packages``:

.. code-block:: bash

    pip install decanter-ai-core-sdk

.. code-block:: python

	import os
	import asyncio

	from decanter import core
	from decanter.core.core_api import TrainInput, PredictInput, SetupInput
	from decanter.core.extra.utils import check_response, gen_id
	from decanter.core.enums.algorithms import Algo
	from decanter.core.enums.evaluators import Evaluator

``Create Context will set the connection to decanter core server, and create an event loop. Since Jupyter already have an event loop, SDK will just use the current event loop. In Jupyter, it will initially exist a running event loop.``:

.. code-block:: python

	loop = asyncio.get_running_loop()
	loop.is_running()


``CoreClient handles the actions of calling api and getting the results,``:
``When initializing, need to set the usr, pwd, host to create Context.``:

.. code-block:: python

	# enable default logger
	core.enable_default_logger()
	# set the username, password, host
	client = core.CoreClient(
	        username=???, password=???, host='http://host:port')


``Open train & test file``:

.. code-block:: python

	train_file_path = '/data/train.csv'
	test_file_path = '/data/test.csv'
	train_file = open(train_file_path , 'rb')
	test_file = open(test_file_path , 'rb')

``Upload data to CoreX``:

.. code-block:: python

	train_data = client.upload(file=train_file, name="train_data")
	test_data = client.upload(file=test_file, name="test_data")

``Setup data to CoreX``:

.. code-block:: python

	setup_input = SetupInput(
        data = train_data,
        data_source=train_data.accessor,
        data_columns=[
            {
                'id': 'Pclass',
                'data_type': 'categorical'
            }])
    train_data = client.setup(setup_input=setup_input, name='setup_data')


``Set train parameters train model``:

.. code-block:: python

	train_input = TrainInput(data=train_data, target='Survived', algos=[Algo.XGBoost], max_model=2, tolerance=0.9)
	exp = client.train(train_input=train_input, select_model_by=Evaluator.mean_per_class_error, name='myexp')

``Set predict parameters and predict result``:

.. code-block:: python

	predict_input = PredictInput(data=test_data, experiment=exp)
	pred_res = client.predict(predict_input=predict_input, name='mypred')

``Show the predict result``:

.. code-block:: python

	pred_res.show_df()


How to Save Model
******************

``Getting Mojo model zip file from decanter.core server and download to local.``

.. code-block:: python
	
    from decanter.core.core_api import Model
    model = Model()
    
    """
    save the model as zip file
    	model_id (str): ObjectId in 24 hex digits.
    	model_path (str): Path to store zip mojo file.
    """
    model_id = {model_id}
    model_path = './tmp/model.zip'
    model.download_by_id(model_id, model_path)
