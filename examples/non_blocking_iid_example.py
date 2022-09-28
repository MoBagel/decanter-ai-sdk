from time import sleep
from decanter_ai_sdk.non_blocking_client import Client
import os
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms
from decanter_ai_sdk.enums.data_types import DataType


def test_iid():
    auth_key = ""  # TODO fill in real authorization key
    project_id = ""  # TODO fill in real project id
    host = ""  # TODO fill in real host
    print("---From test iid---")

    client = Client(
        auth_key=auth_key, project_id=project_id, host=host, dry_run_type=None
    )

    current_path = os.path.dirname(os.path.abspath(__file__))

    train_file_path = os.path.join(current_path, "../data/train.csv")
    train_file = open(train_file_path, "rb")
    train_id = client.upload(train_file, "train_file")

    while client.check_upload_status(train_id) != "done":
        print(
            "Upload task: ", train_id, " status: ", client.check_upload_status(train_id)
        )
        sleep(3)
    print("Train file uploaded.")

    test_file_path = os.path.join(current_path, "../data/test.csv")
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "test_file")
    while client.check_upload_status(train_id) != "done":
        print(
            "Upload task: ", test_id, " status: ", client.check_upload_status(test_id)
        )
        sleep(3)
    print("Test file uploaded.")

    print("This will show top 2 uploaded table names and ids: \n")

    for count in range(0, 2):
        table = client.get_table_list()[count]
        print(count, "name:", table["name"], ",id:", table["_id"])

    print(
        "\nThis will show the info of the first table:\n",
        client.get_table(client.get_table_list()[0]["_id"]),
    )

    experiment_id = client.train_iid(
        experiment_name="exp_name",
        experiment_table_id=train_id,
        target="Survived",
        evaluator=ClassificationMetric.AUC,
        custom_feature_types={
            "Pclass": DataType.categorical,
            "Parch": DataType.categorical,
        },
        max_model=5,
        algos=["DRF", "GBM", IIDAlgorithms.DRF],
    )

    # print("This will show info of the experiment:\n", experiment.experiment_info())

    while client.check_exp_status(experiment_id) != "done":
        print("Experiment not done yet.")
        sleep(3)

    print("Experiment id:", experiment_id, " done.")

    exp = client.get_exp_result(exp_id=experiment_id)

    m = exp.result.get_best_model()

    print("This will show the best model id:", m.model_id, "name:", m.model_name, "\n")

    model = exp.result.get_best_model_by_metric(ClassificationMetric.MISCLASSIFICATION)
    print(
        "This will show the best model evaluated by misclassification. id:",
        model.model_id,
        "name:",
        model.model_name,
        "\n",
    )

    predict_id = client.predict_iid(
        keep_columns=[], non_negative=False, test_table_id=test_id, model=m
    )
    while client.check_pred_status(predict_id) != "done":
        print("Prediction task not done yet.")
        sleep(3)

    print("Prediction id:", predict_id, " done.")

    prediction = client.get_pred_result(pred_id=predict_id)

    print(
        "This will show the id of tested ../data:",
        prediction.result.attributes["table_id"],
        "\n",
    )
    print(
        "Head of the prediction ../data:\n", prediction.result.get_predict_df().head()
    )


test_iid()
