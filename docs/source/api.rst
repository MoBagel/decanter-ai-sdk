.. _api:

Developer Interface
===================
This part of the documentation covers all the interfaces of Decanter  Core SDK.


Main Interface
--------------
Decanter AI Core SDK's main functionality can be accessed by the bellow
Interfaces.


Connection and Funcional Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: decanter.core.context
   :members:
   :undoc-members:

Client for Decanter Core API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: decanter.core.client
   :members:
   :undoc-members:

Plot
~~~~~~
.. autofunction:: decanter.core.plot.show_model_attr

Prompt Info
~~~~~~~~~~~~
.. autofunction:: decanter.core.enable_default_logger


Jobs
--------------
Introduce all the Jobs handling different kinds of actions, and the
relation between :class:`~decanter.core.jobs.job.Job` and
:class:`~decanter.core.jobs.task.Task`


Task
~~~~~~
.. automodule:: decanter.core.jobs.task
   :members:
   :member-order: bysource
   :show-inheritance:

Jobs
~~~~~~
.. automodule:: decanter.core.jobs.job
   :members:
   :member-order: bysource

.. autoclass:: decanter.core.jobs.data_upload.DataUpload
   :members:
   :show-inheritance:

.. automodule:: decanter.core.jobs.experiment
   :members:
   :show-inheritance:

.. automodule:: decanter.core.jobs.predict_result
   :members:
   :show-inheritance:


Core Api Interface
------------------------
The API interfaces of Decanter, mainly handles the request and response body.


Train Input
~~~~~~~~~~~~
.. automodule:: decanter.core.core_api.train_input
   :members:
   :undoc-members:
   :show-inheritance:

Setup Input
~~~~~~~~~~~~
.. automodule:: decanter.core.core_api.setup_input
   :members:
   :undoc-members:
   :show-inheritance:


Predict Input
~~~~~~~~~~~~~~
.. automodule:: decanter.core.core_api.predict_input
   :members:
   :undoc-members:
   :show-inheritance:

Model
~~~~~~
.. automodule:: decanter.core.core_api.model
   :members:
   :undoc-members:
   :show-inheritance:

Enum
------------------------
Return the machine learning algorithm and evaluator supported by the current Decanter in the form of enumerate object.


Algorithms
~~~~~~~~~~~~
.. automodule:: decanter.core.enums.algorithms
   :members:


Evaluators
~~~~~~~~~~~~
.. automodule:: decanter.core.enums.evaluators
   :members:

