from json import load
from llm_sdk import Small_LLM_Model
from typing import Dict


def get_id_token(model: Small_LLM_Model) -> Dict:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as vf:
        token_id = load(vf)

    id_token = {v: k for k, v in token_id.items()}
    return id_token
