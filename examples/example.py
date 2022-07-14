# pylint: disable=unsubscriptable-object
"""
Run: python -m examples.example
"""
import os

import pandas as pd

from src.decanter import core
from src.decanter.core.core_api import TrainInput, PredictInput, SetupInput
from src.decanter.core.enums.algorithms import Algo
from src.decanter.core.enums.evaluators import Evaluator


def main():
    """
    Example of training titanic dataset.
    """

    # Enable default logger for logging message
    core.enable_default_logger()

    # The Main object handles the calling of Decanter's API.
    # Create connection to Decanter server, and set up basic settings.
    # Logger message:
    #   "[Context] connect healty :)" if success.
    client = core.CoreClient(username="gp", password="gp-admin", host="http://192.168.2.18:2999")

    current_path = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(current_path, 'data/train.csv')
    test_file_path = os.path.join(current_path, 'data/test.csv')
    train_file = open(train_file_path, 'rb')
    test_df = pd.read_csv(test_file_path)

    # Upload data to Decanter server. Will Get the DataUpload result.
    train_data = client.upload(file=train_file, name='upload_train_data')
    test_data = client.upload(file=test_df, name='upload_test_data')

    # Set up data to change data type. Will Get the DataSetup result
    setup_input = SetupInput(
        data=train_data,
        data_columns=[
            {
                'id': 'Pclass',
                'data_type': 'categorical'
            }])
    train_data = client.setup(setup_input=setup_input, name='setup_data')

    # Settings for training model using TrainInput.
    train_input = TrainInput(
        data=train_data, target='Survived',
        algos=[Algo.XGBoost], max_model=2, tolerance=0.9)

    # Start Model Training, get Experiment result in return.
    exp = client.train(
        train_input=train_input, select_model_by=Evaluator.auto,
        name='myexp')

    # Settings for predict model using PredictInput.
    predict_input = PredictInput(data=test_data, experiment=exp)

    # Start model prediction, get PredictResult in return.
    pred_res = client.predict(predict_input=predict_input, name='mypred')

    # Run all the actions above and make sure all is done before continue
    # to next step.
    client.run()
    print("TASKS ARE DONE")

    # Print out the text result of prediction.
    print(pred_res.show())

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Download model zipped file to local.
    exp.best_model.download(model_path='./tmp/mymodel.zip')

    # Dwonload predict results in csv to local.
    pred_res.download_csv(path='./tmp/pred_res.csv')

    # Close context, close event loop and reset connections.
    client.close()


if __name__ == '__main__':
    main()
