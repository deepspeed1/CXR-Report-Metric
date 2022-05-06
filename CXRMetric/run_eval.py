import json
import numpy as np
import os
import pandas as pd
import pickle
import torch

from bert_score import BERTScorer
from fast_bleu import BLEU
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import config
from CXRMetric.radgraph_evaluate_model import run_radgraph

"""Computes 4 individual metrics and a composite metric on radiology reports."""


CHEXBERT_PATH = config.CHEXBERT_PATH
RADGRAPH_PATH = config.RADGRAPH_PATH

REPORT_COL_NAME = "report"
STUDY_ID_COL_NAME = "study_id"
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]
COMPOSITE_METRIC_PATH = "CXRMetric/composite_metric_model.pkl"
NORMALIZER_PATH = "CXRMetric/normalizer.pkl"

cache_path = "cache/"
pred_embed_path = os.path.join(cache_path, "pred_embeddings.pt")
gt_embed_path = os.path.join(cache_path, "gt_embeddings.pt")
weights = {"bigram": (1/2., 1/2.)}
composite_metric_col = "cxr_metric_score"


def prep_reports(reports):
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

# adds a column with the bleu score for each report
def add_bleu_col(gt_df, pred_df):
    pred_df["bleu_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row[REPORT_COL_NAME]])[0]
        pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
        if len(pred_row) == 0:
            print("problem")
            continue
        predicted_report = \
            prep_reports([pred_row[REPORT_COL_NAME].values[0]])[0]
        if len(pred_row) == 1:
            bleu = BLEU([gt_report], weights)
            score = bleu.get_score([predicted_report])["bigram"]
            assert len(score) == 1
            _index = pred_df.index[
                pred_df[STUDY_ID_COL_NAME]==row[STUDY_ID_COL_NAME]].tolist()[0]
            pred_df.at[_index, "bleu_score"] = score[0]
    return pred_df

def add_bertscore_col(gt_df, pred_df):
    test_reports = gt_df[REPORT_COL_NAME].tolist()
    test_reports = [test.lstrip() for test in test_reports]
    method_reports = pred_df[REPORT_COL_NAME].tolist()
    method_reports = [report.lstrip() for report in method_reports]

    scorer = BERTScorer(model_type="distilroberta-base", batch_size=256)
    _, _, f1 = scorer.score(method_reports, test_reports)
    pred_df["bertscore"] = f1
    return pred_df

def add_semb_col(pred_df, semb_path, gt_path):
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    np_label_embeds = torch.stack([*label_embeds.values()], dim=0).numpy()
    np_pred_embeds = torch.stack([*pred_embeds.values()], dim=0).numpy()
    # print(len(pred_df), len(np_pred_embeds), len(np_label_embeds))
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum() / (
            np.linalg.norm(label) * np.linalg.norm(pred))
        scores.append(sim_scores)
    pred_df["semb_score"] = scores
    return pred_df

def add_radgraph_col(pred_df, entities_path, relations_path):
    study_id_to_radgraph = {}
    with open(entities_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            study_id_to_radgraph[int(study_id)] = float(f1)
    with open(relations_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            study_id_to_radgraph[int(study_id)] += float(f1)
            study_id_to_radgraph[int(study_id)] /= float(2)
    radgraph_scores = []
    count = 0
    for i, row in pred_df.iterrows():
        radgraph_scores.append(study_id_to_radgraph[int(row[STUDY_ID_COL_NAME])])
    pred_df["radgraph_combined"] = radgraph_scores
    return pred_df

def calc_metric(gt_csv, pred_csv, out_csv): # TODO: support single metrics at a time
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    # print(gt_csv, pred_csv)
    # take a csv to the eval an gt reports
    gt, pred = pd.read_csv(gt_csv), pd.read_csv(pred_csv)

    # check that the length is the same, assume that the order is the same
    assert len(gt) == len(pred)
    assert (REPORT_COL_NAME in gt.columns) and (REPORT_COL_NAME in pred.columns)

    # add blue column to the eval df
    pred = add_bleu_col(gt, pred)

    # add bertscore column to the eval df
    pred = add_bertscore_col(gt, pred)

    # run encode.py to make the semb column
    # print(os.getcwd())
    os.system(f"mkdir -p {cache_path}")
    os.system(f"python CXRMetric/CheXbert/src/encode.py -c {CHEXBERT_PATH} -d {pred_csv} -o {pred_embed_path}")
    os.system(f"python CXRMetric/CheXbert/src/encode.py -c {CHEXBERT_PATH} -d {gt_csv} -o {gt_embed_path}")
    pred = add_semb_col(pred, pred_embed_path, gt_embed_path)

    # run radgraph to create that column
    entities_path = os.path.join(cache_path, "entities_cache.json")
    relations_path = os.path.join(cache_path, "relations_cache.json")
    run_radgraph(gt_csv, pred_csv, cache_path, RADGRAPH_PATH,
                 entities_path, relations_path)
    pred = add_radgraph_col(pred, entities_path, relations_path)

    # run the linear model
    with open(COMPOSITE_METRIC_PATH, "rb") as f:
        composite_metric_model = pickle.load(f)
    with open(NORMALIZER_PATH, "rb") as f:
        normalizer = pickle.load(f)
    # normalize
    input_data = np.array(pred[COLS])
    norm_input_data = normalizer.transform(input_data)
    # generate new col
    scores = composite_metric_model.predict(norm_input_data)

    # append new column
    pred[composite_metric_col] = scores

    # save results in the out folder
    pred.to_csv(out_csv)
