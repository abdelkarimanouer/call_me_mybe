from typing import Dict
from llm_sdk import Small_LLM_Model
from json import load


def get_id_token(model: Small_LLM_Model) -> Dict:
    vocab_path = model.get_path_to_vocab_file()
    vocab_dict: Dict = {}

    with open(vocab_path, "r") as f:
        vocab_dict = load(f)

    id_token = {v: k for k, v in vocab_dict.items()}
    return id_token
