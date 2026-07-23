from json import load
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import Dict, List, Set


class Vocab:
    """
    Manages vocabulary operations.
    Handles token lookups and ID mappings for the model's vocabulary.
    """

    @staticmethod
    def get_id_token(model: Small_LLM_Model) -> Dict[int, str]:
        """
        Loads the vocabulary file for the model.
        Returns a mapping from token IDs to token strings.
        """
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, "r") as vf:
            token_id = load(vf)

        id_token = {v: k for k, v in token_id.items()}
        return id_token

    @staticmethod
    def build_token_lookup(id_token: Dict[int, str]) -> Dict[str, int]:
        """
        Builds a reverse mapping dictionary.
        Maps token strings to their corresponding token IDs.
        """
        return {token: tid for tid, token in id_token.items()}

    @staticmethod
    def find_tokens_for_exact_string(
        target: str,
        token_lookup: Dict[str, int]
    ) -> List[int]:
        """
        Finds token IDs that exactly match a target string.
        Checks the token lookup and returns matching IDs.
        """
        results: List[int] = []
        if target in token_lookup:
            results.append(token_lookup[target])
        return results

    @staticmethod
    def find_tokens_starting_with(
        prefix: str,
        id_token: Dict[int, str]
    ) -> Set[int]:
        """
        Finds all token IDs whose text starts with a prefix.
        Returns a set of matching token IDs.
        """
        return {
            tid for tid, token in id_token.items()
            if token.startswith(prefix)
        }
