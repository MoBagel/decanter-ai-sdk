[![Coverage Status](https://coveralls.io/repos/github/MoBagel/decanter-ai-sdk/badge.svg?branch=coveralls)](https://coveralls.io/github/MoBagel/decanter-ai-sdk?branch=coveralls)
[![tests](https://github.com/MoBagel/decanter-ai-sdk/workflows/main/badge.svg)](https://github.com/MoBagel/decanter-ai-sdk)
[![PyPI version](https://badge.fury.io/py/decanter-ai-sdk.svg)](https://badge.fury.io/py/decanter-ai-sdk)
# Mobagel decanter ai sdk

Decanter AI is a powerful AutoML tool which enables everyone to build ML models and make predictions without data science background. With Decanter AI SDK, you can integrate Decanter AI into your application more easily with Python.

It supports actions such as data uploading, model training, and prediction to run in a more efficient way and access results more easily.

To know more about Decanter AI and how you can be benefited with AutoML, visit [MoBagel website](https://mobagel.com/tw/) and contact us to try it out!

## How it works

- Upload train and test files in both csv and pandas dataframe.
- Setup different standards and conduct customized experiments on uploaded data.
- Use different models to run predictions
- Get predict data in pandas dataframe form.

## Requirements

- [Python >= 3.8](https://www.python.org/downloads/release/python-380/)
- [poetry](https://python-poetry.org/)

## Usage

### Installation

`pip install decanter-ai-sdk`
### Constructor
To use this sdk, you must first construct a client object.
```python
from decanter_ai_sdk.client import Client
    client = Client(
        auth_key="auth_API_key",
        project_id="project_id",
        host="host_url",
    )
```

### Upload
After the client is constructed, now you can use it to upload your training and testing files in both csv and pandas dataframe. This function will return uploaded data id in Decanter server.
```python
import os
sys.path.append("..")

current_path = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(current_path, "ts_train.csv")
train_file = open(train_file_path, "rb")
train_id = client.upload(train_file, "train_file")
```

### Experiment
To conduct an experiment, you need to first specify which type of data you are going to use , i.e., iid or ts, then you can input parameters by following our pyhint to customize your experiment.
After the experiment, the function will return an object which you can get experiment attributes from it.
```python
# Training iid data
experiment = client.train_iid(
    experiment_name=exp_name,
    experiment_table_id=train_id,
    target="Survived",
    evaluator=ClassificationMetric.AUC,
    custom_feature_types={
        "Pclass": DataType.categorical,
        "Parch": DataType.categorical,
    },
)
```

```python
# Training ts data
experiment = client.train_ts(
    experiment_name=exp_name,
    experiment_table_id=train_id,
    target="Passengers",
    datetime="Month",
    time_groups=[],
    timeunit=TimeUnit.month,
    groupby_method="sum",
    max_model=5,
    evaluator=RegressionMetric.MAPE,
    custom_feature_types={"Pclass": DataType.numerical},
)
```
To get its attributes, you can either extract them by simply using dot or its functions.
```python
# Experiment object usage
best_model = experiment.get_best_model()
model_list = experiment.get_model_list()
best_auc_model = experiment.get_best_model_by_metric(ClassificationMetric.AUC)
```
### Prediction
Now you can use model data to run prediction.
```python
# Predicting iid data
predict = client.predict_iid(
    keep_columns=[], 
    non_negative=False, 
    test_table_id=test_id, 
    model=best_model
)
```

```python
# Predicting ts data
predict = client.predict_ts(
    keep_columns=[], 
    non_negative=False, 
    test_table_id=test_id, 
    model=best_model
)
```
To get prediction result, do
```python
predict_data = predict.get_predict_df()
```
## Development

### Installing poetry

1. `pip install poetry poethepoet`
2. `poetry install` #Project setup.
3. `poetry shell` #Start your project in poetry env.

Now you can create your own branch to start developing new feature.

### Testing
To run test, do:
```
poe test
```

### Lint and format
To lint, do:
```
poe lint
```

To reformat, do:
```
poe format
```

## Releasing
1. poetry version [new_version]
2. git commit -m"Bump version"
3. git push origin main
4. create new release on github.
5. Create release off main branch, auto generate notes, and review release note.
6. Publish release

## Enums
#TODO

## License
#TODO

## TODO
#TODO



