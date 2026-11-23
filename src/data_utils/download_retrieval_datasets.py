import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset

def get_fiqa_dataset(base_dir: str):
    ds = load_dataset("vibrantlabsai/fiqa", "main", cache_dir=base_dir)
    return ds

def get_squad_dataset(base_dir: str):
    ds = load_dataset("rajpurkar/squad", cache_dir=base_dir)
    return ds