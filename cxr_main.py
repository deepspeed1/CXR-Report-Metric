import os
import pandas as pd
from CXRMetric.run_eval import calc_metric
from huggingface_hub import hf_hub_download, login, HfApi, snapshot_download


cwd = os.getcwd()

DOWNLOAD_REPO = input('enter download repo id: ')

login(input('enter hf token: '))

hf_hub_download(repo_id = DOWNLOAD_REPO, filename = 'gt_reports.csv')
hf_hub_download(repo_id = DOWNLOAD_REPO, filename = 'predicted_reports.csv')
gt_reports = f'{cwd}/gt_reports.csv'
predicted_reports = f'{cwd}/predicted_reports.csv'
out_file = f'{cwd}/report_scores.csv'
use_idf = False
calc_metric(gt_reports, predicted_reports, out_file, use_idf)
