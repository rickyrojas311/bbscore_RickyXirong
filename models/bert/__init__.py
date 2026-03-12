from models import MODEL_REGISTRY
from .bert import BERT

MODEL_REGISTRY["bert_base"] = {
    "class": BERT, "model_id_mapping": "BERT-Base"}

MODEL_REGISTRY["bert_large"] = {
    "class": BERT, "model_id_mapping": "BERT-Large"}
