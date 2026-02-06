# score.py (Batch Endpoint) — minimal changes vs Online
# Zmienione tylko to, co musi być zmienione:
# - run() ma teraz signature run(mini_batch) i czyta JSON z plików
# - dodana obsługa błędów per plik (nie wywala całego joba na 1 złym wejściu)
# Pozostałe funkcje zostają w tej samej logice co online.

import os
import json
import time
import logging

import pandas as pd

# zakładam, że te importy masz w projekcie tak jak wcześniej
import pre_processing
from models import LogisticRegression, Transformer  # jak u Ciebie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

relevance_model = None
cd_logreg_model = None
cd_transformer_model = None


def init():
    """
    init() jest OK dla batch.
    W batch (tak jak w online) modele są montowane i ścieżka jest podawana w env.
    U Ciebie było AZUREML_MODEL_DIR — zostawiam bez zmian.
    """
    global relevance_model
    global cd_logreg_model
    global cd_transformer_model

    logger.info("Initializing model...")

    model_root = os.getenv("AZUREML_MODEL_DIR")
    if not model_root:
        raise RuntimeError("AZUREML_MODEL_DIR is not set")

    # U Ciebie było: join(model, "AL") — zostawiam
    model_path = os.path.join(model_root, "AL")

    if not os.path.isdir(model_path):
        # Minimalna walidacja, żeby błąd był czytelny w job logach
        raise RuntimeError(
            f"Model path not found: {model_path}. "
            f"AZUREML_MODEL_DIR={model_root}, contents={os.listdir(model_root) if os.path.isdir(model_root) else 'N/A'}"
        )

    logger.info("AZUREML_MODEL_DIR=%s", model_root)
    logger.info("Using model path=%s", model_path)

    relevance_model = LogisticRegression.load(
        os.path.join(model_path, "logreg_relevance.joblib")
    )

    cd_logreg_model = LogisticRegression.load(
        os.path.join(model_path, "logreg_cd.joblib")
    )

    transformer = os.path.join(model_path, "transformer_model")
    model_cd_transformer_path = os.path.join(transformer, "transformer")
    model_cd_transformer_le_path = os.path.join(transformer, "transformer_le.joblib")

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
    return (
        len(text.split(" ")) > 2
        and cd_logreg_proba > 0.1
        and (
            (relevant_proba > 0.65 and cd_transformer_proba > 0.9)
            or cd_transformer_proba > 0.95
        )
    )


def inference(document: json, num_cd_predictions: int):
    # load and process document
    start_time = time.time()

    text = pd.Series(
        content["text"]
        for content in document["contentDomain"]["byId"].values()
    )

    text = pre_processing.clean_text(text)
    latency = time.time() - start_time
    print("Time to load and clean document: {}".format(latency))

    # score relevance model
    start_time = time.time()
    all_relevance_predictions = relevance_model.predict_proba(text)
    latency = time.time() - start_time
    print("Time to score relevance model: {}".format(latency))

    # score cd logreg model
    start_time = time.time()
    all_cd_logreg_predictions = cd_logreg_model.predict_top_n_labels_with_proba(
        text, num_cd_predictions
    )
    latency = time.time() - start_time
    print("Time to score cd logreg model: {}".format(latency))

    # score cd transformer
    start_time = time.time()
    all_cd_transformer_predictions = cd_transformer_model.predict_top_n_labels_with_proba(
        text, num_cd_predictions
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
            document_demand_predictions.add(cd_transformer_predictions[0]["label"])

        content.update(
            {
                "relevantProba": relevance_prediction,
                "cdLogregPredictions": cd_logreg_predictions,
                "cdTransformerPredictions": cd_transformer_predictions,
            }
        )

    document["documentDemandPredictions"] = list(document_demand_predictions)

    result = document

    latency = time.time() - start_time
    print("Time to format results: {}".format(latency))

    return result


def run(mini_batch):
    """
    Batch Endpoint contract:
    - mini_batch: lista ścieżek do plików wejściowych
    - każdy plik zawiera JSON w tym samym formacie co online:
        {"document": {...}, "num_preds": 5}

    Zwraca listę wyników (po 1 na plik).
    Dodana obsługa błędów per plik (nie wywala całego joba na 1 złym pliku).
    """
    outputs = []

    for item in mini_batch:
        try:
            # item powinien być ścieżką do pliku
            if not isinstance(item, str) or not os.path.exists(item):
                raise ValueError(f"Input is not a valid file path: {item}")

            with open(item, "r", encoding="utf-8") as f:
                raw_data = f.read()

            logger.info(f"Received request with data (file={item})")

            if not raw_data or raw_data.strip() == "":
                raise ValueError("Bad Request: Request body cannot be empty!")

            try:
                request_data = json.loads(raw_data)
            except json.JSONDecodeError:
                raise ValueError("Bad Request: Invalid JSON format!")

            if "document" not in request_data or "num_preds" not in request_data:
                raise ValueError("Bad Request: Invalid input, expected 'document' and 'num_preds'.")

            document = request_data["document"]
            num_pred = int(request_data["num_preds"])

            response = inference(document, num_pred)

            # Tak jak online: wrapper z predictions
            outputs.append({"predictions": response})

        except Exception as e:
            # Obsługa błędu per plik: zapisujesz error zamiast wywalać cały job
            logger.error(f"Error processing item {item}: {str(e)}", exc_info=True)
            outputs.append(
                {
                    "error": str(e),
                    "input": item,
                }
            )

    return outputs
