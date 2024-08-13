import os
import pandas as pd
from CXRMetric.run_eval import calc_metric
from huggingface_hub import hf_hub_download, login, HfApi, snapshot_download


cwd = os.getcwd()

DOWNLOAD_REPO = input('enter download repo is')

login(input('enter hf token'))

snapshot_download(repo_id = DOWNLOAD_REPO)

gt_reports = f'{cwd}/gt_reports.csv'
predicted_reports = f'{cwd}/predicted_reports.csv'
out_file = f'{cwd}/report_scores.csv'
use_idf = False
calc_metric(gt_reports, predicted_reports, out_file, use_idf)
