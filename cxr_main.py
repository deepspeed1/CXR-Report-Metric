import os
import pandas as pd
from CXRMetric.run_eval import calc_metric
from huggingface_hub import hf_hub_download, login, HfApi, snapshot_download


cwd = os.getcwd()

DOWNLOAD_REPO = 'ahmedabdelwahed/RologyVLM-22k'

login('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')

# gt_reports
hf_hub_download(
    filename = 'gt_reports.csv',
    # local_dir = cwd,
    repo_type = 'model',
    repo_id = DOWNLOAD_REPO)

# predicted_reports
hf_hub_download(
    filename = 'predicted_reports.csv',
    # local_dir = cwd,
    repo_type = 'model',
    repo_id = DOWNLOAD_REPO)


gt_reports = f'{cwd}/gt_reports.csv'
predicted_reports = f'{cwd}/predicted_reports.csv'
out_file = f'{cwd}/report_scores.csv'
use_idf = False
calc_metric(gt_reports, predicted_reports, out_file, use_idf)
