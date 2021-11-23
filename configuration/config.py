import logging
import os
from pathlib import Path
import json
import numpy as np
try:
    import pandas as pd
except:
    print("not install pandas")
from tqdm import tqdm
from collections import defaultdict


"""
数据、代码都在workspace里

"""

ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
workspace = Path(ROOT_PATH) / ".."

# data
data_dir = Path(ROOT_PATH) / "data"
model_dir = Path(ROOT_PATH) / "model"

bert_data_path = workspace / 'db__pytorch_pretrained_bert'
tencent_w2v_path = workspace / 'db__word2vec'
common_data_path = workspace / 'db__common_dataset'

# bert
bert_vocab_path = bert_data_path / 'bert-base-chinese' / 'vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese'

# UER
uer_bert_base_model_path = bert_data_path / 'uer-bert-base'
uer_bert_large_model_path = bert_data_path / 'uer-bert-large'

# roberta
roberta_wwm_path = bert_data_path / "chinese_wwm_ext_L-12_H-768_A-12"

roberta_large_path = bert_data_path / 'chinese_Roberta_bert_wwm_large_ext_pytorch'
bert_wwm_pt_path = bert_data_path / "chinese_wwm_ext_pytorch"
roberta_wwm_pt_path = bert_data_path / "chinese_roberta_wwm_ext_pytorch"

# nezha
nezha_pt_path = bert_data_path / "nezha-cn-base"

# t5
mt5_pt_path = bert_data_path / "mt5"
mt5_small_pt_path = bert_data_path / "mt5_small_pt"
mt5_large_pt_path = bert_data_path / "mt5_large_pt"
t5_pegasus_pt_path = bert_data_path / "chinese_t5_pegasus_base_pt"
t5_pegasus_tf_path = bert_data_path / "chinese_t5_pegasus_base"

# wobert
wobert_plus_tf_path = bert_data_path / "chinese_wobert_plus_L-12_H-768_A-12"
wobert_plus_pt_path = bert_data_path / "chinese_wobert_plus_pt"

# bart
bart_pt_path = bert_data_path / "bart"

# cpm
cpm_tf_path = bert_data_path / "cpm_tf"

# gpt2
gpt2_pt_path = bert_data_path / "gpt2_pt"
gpt2_tf_path = bert_data_path / "gpt2_tf"

# simbert
simbert_path = bert_data_path / "chinese_simbert_L-12_H-768_A-12"
simbert_pt_path = bert_data_path / "chinese_simbert_pt"

# cdial-gpt
cdial_gpt_pt_path = bert_data_path / "cdial_gpt"
cdial_gpt_nezha_path = bert_data_path / "nezha_gpt_dialog"
cdial_gpt_nezha_pt_path = bert_data_path / "nezha_gpt_dialog_pt"

# mlm_rc
mlm_rc_model_path = bert_data_path / "mlm_rc_model"
mlm_rc_model_pt_path = bert_data_path / "mlm_rc_model_pt"

# open dataset
open_dataset_path = common_data_path / "open_dataset"


###############################################
# log
###############################################

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'begin progress ...')


