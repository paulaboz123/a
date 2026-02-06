import os
import time
import json
import pandas as pd
from models.logistic_regression import LogisticRegression  # noqa
from models.transformer import Transformer  # noqa
import pre_processing  # noqa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    global relevance_model
    global cd_logreg_model
    global cd_transformer_model

    logger.info("Initializing model...")
    model = os.getenv("AZUREML_MODEL_DIR")
    print(f"line 23: {os.listdir(model)}")
    print(f"line 24: {model}")

    model = os.path.join(model, "albot")
    print(f"line 27: {os.listdir(model)}")

    relevance_model = LogisticRegression.load(
        os.path.join(model, "logreg_relevance.joblib")
    )

    cd_logreg_model = LogisticRegression.load(
        os.path.join(model, "logreg_cd.joblib")
    )

    transformer = os.path.join(model, "transformer_model")

    model_cd_transformer_path = os.path.join(
        transformer, "transformer"
    )

    model_cd_transformer_le_path = os.path.join(
        transformer, "transformer_le.joblib"
    )

    cd_transformer_model = Transformer.load(
        model_cd_transformer_path,
        model_cd_transformer_le_path
    )

    logger.info("Model initialized successfully.")


def is_relevant_customer_demand(
    text: str,
    relevant_proba: float,
    cd_logreg_proba: float,
    cd_transformer_proba: float,
):
    """
    Determines if a customer demand prediction should be added to the document.
    cd_logreg_proba is just used as an additional simple out of distribution check,
    we don't care about the label being the same.
    """
    return (
        len(text.split(" ")) > 2
        and cd_logreg_proba > 0.1
        and (
            (relevant_proba > 0.65 and cd_transformer_proba > 0.9)
            or cd_transformer_proba > 0.95
        )
    )


def inference(
    document: json,
    num_cd_predictions: int
):
    # load and process document
    start_time = time.time()
    # document = json.loads(document)[0]
    text = pd.Series(
        content["text"] for content in
        document["contentDomain"]["byId"].values()
    )

    text = pre_processing.clean_text(text)
    latency = time.time() - start_time
    print("Time to load and clean document: {}".format(latency))

    # score relevence model
    start_time = time.time()
    all_relevance_predictions = relevance_model.predict_proba(text)[:, 1]
    latency = time.time() - start_time
    print("Time to score relevence model: {}".format(latency))

    # score cd logreg model
    start_time = time.time()
    all_cd_logreg_predictions = (
        cd_logreg_model.predict_top_n_labels_with_proba(
            text, num_cd_predictions
        )
    )
    latency = time.time() - start_time
    print("Time to score cd logreg model: {}".format(latency))

    # score cd transformer
    start_time = time.time()
    all_cd_transformer_predictions = (
        cd_transformer_model.predict_top_n_labels_with_proba(
            text, num_cd_predictions
        )
    )
    latency = time.time() - start_time
    print("Time to score cd transformer model: {}".format(latency))

    # format results
    start_time = time.time()
    document_demand_predictions = set()

    for (
        content,
        relevance_prediction,
        cd_logreg_predictions,
        cd_transformer_predictions,
    ) in zip(
        document["contentDomain"]["byId"].values(),
        all_relevance_predictions,
        all_cd_logreg_predictions,
        all_cd_transformer_predictions,
    ):
        if is_relevant_customer_demand(
            content["text"],
            relevance_prediction,
            cd_logreg_predictions[0]["proba"],
            cd_transformer_predictions[0]["proba"],
        ):
            document_demand_predictions.add(
                cd_transformer_predictions[0]["label"]
            )

        content.update(
            {
                "relevantProba": relevance_prediction,
                "cdLogregPredictions": cd_logreg_predictions,
                "cdTransformerPredictions": cd_transformer_predictions,
            }
        )

    # update document to include document_demand_predictions
    document['documentDemandPredictions'] = list(document_demand_predictions)

    result = document

    latency = time.time() - start_time
    print("Time to format results: {}".format(latency))

    return result


# =========================
# ONLY REQUIRED CHANGE: run()
# =========================
def run(mini_batch):
    """
    Azure ML Batch Endpoint:
      - mini_batch = list of file paths
      - each file contains EXACT SAME JSON as online used to receive in raw_data:
          {"document": {...}, "num_preds": <int> }

    Returns list of per-file results.
    Adds per-file error handling (doesn't crash whole job on one bad file).
    """
    results = []

    for item in mini_batch:
        try:
            # Batch dostarcza ścieżki do plików
            if not isinstance(item, str) or not os.path.exists(item):
                raise ValueError(f"Bad Request: Input is not a valid file path: {item}")

            with open(item, "r", encoding="utf-8") as f:
                raw_data = f.read()

            logger.info(f"Received request with data from file: {item}")

            if not raw_data or raw_data.strip() == "":
                raise ValueError("Bad Request: Request body cannot be empty!")

            # Parse the input data (tak jak w Twoim online)
            try:
                request_data = json.loads(raw_data)
            except json.JSONDecodeError:
                raise ValueError("Bad Request: Invalid JSON format!")

            if "document" not in request_data or "num_preds" not in request_data:
                raise ValueError("Bad Request: Invalid input, expected 'document' and 'num_preds'.")

            document = request_data["document"]
            num_pred = int(request_data["num_preds"])

            response = inference(document, num_pred)

            logger.info(f"Inference response computed for file: {item}")

            # w batch zwracamy listę wyników; zachowujemy wrapper jak w online
            if isinstance(response, dict):
                results.append({"predictions": response})
            elif hasattr(response, "tolist"):
                results.append({"predictions": response.tolist()})
            else:
                results.append({"error": "Unexpected response format!", "input_file": item})

        except ValueError as ve:
            logger.warning(f"Bad request for file {item}: {str(ve)}")
            results.append({"error": str(ve), "input_file": item})

        except Exception as e:
            logger.error(f"An error occurred for file {item}: {str(e)}", exc_info=True)
            # w batch nie przerywamy całości – zwracamy error dla tego pliku
            results.append({"error": "Internal server error", "input_file": item})

    return results

