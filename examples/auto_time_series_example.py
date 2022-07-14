"""
Run; python -m examples.time_series_example
"""
import os

from decanter import core
from decanter.core.core_api import TrainTSInput, PredictTSInput, SetupInput
from decanter.core.enums.algorithms import Algo
from decanter.core.enums.evaluators import Evaluator
# from decanter.core.jobs import DataUpload, Experiment


def main():
    """
    Example of running time series on decanter.
    """
    # Enable default logger for logging message
    core.enable_default_logger()

    # The Main object handles the calling of Decanter's API.
    # Create connection to Decanter server, and set up basic settings.
    # Logger message:
    #   "[Context] connect healty :)" if success.
    client = core.CoreClient(username=???, password=???, host=???)

    train_file_path = 'data/ts_data/iris_train.csv'
    test_file_path = 'data/ts_data/iris_test.csv'
    train_file = open(train_file_path, 'rb')
    test_file = open(test_file_path, 'rb')

    # Upload data to Decanter server. Will Get the DataUpload result.
    train_data = client.upload(file=train_file, name='upload_train_data')
    test_data = client.upload(file=test_file, name='upload_test_data')

    # train_data = DataUpload.create(data_id = "{data_id}", name="train_data")
    # test_data = DataUpload.create(data_id = "{data_id}", name="test_data")

    # result will be done to further step.
    # Logger message:
    #     Job proccessing: Create a progress bar showing its current process.
    #     Job finished: "[Job] 'name' done status: 'final status' id: 'id'"

    # Settings for time series forecast training.
    train_input = TrainTSInput(
        data=train_data, target='regression', forecast_horizon=7, gap=0, algorithms=["XGBoost"],
        datetime_column='date', max_model=1, evaluator=Evaluator.r2, time_unit='day',
        max_iteration=10, numerical_groupby_method='mean')

    # Start train time series models.
    exp_ts = client.train_ts(train_input=train_input, select_model_by=Evaluator.r2, name='ExpTS')

    # Settings for predict time series model using PredictTSInput.
    predict_ts_input = PredictTSInput(data=test_data, experiment=exp_ts)

    # Start time series model prediction, get PredicTStResult in return.
    pred_ts_res = client.predict_ts(predict_input=predict_ts_input)

    # Use context.run() to block the above instructions, making sure all the
    # result will be done to further step.
    # Logger message:
    #     Job proccessing: Create a progress bar showing its current process.
    #     Job finished: "[Job] 'name' done status: 'final status' id: 'id'"
    client.run()

    # Print out the text result of prediction.
    print(pred_ts_res.show())

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Dwonload predict results in csv to local.
    pred_ts_res.download_csv(path='./tmp/pred_res.csv')

    client.close()


if __name__ == '__main__':
    main()
