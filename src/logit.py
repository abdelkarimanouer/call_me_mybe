from typing import List, Set
import numpy as np


class Logit:
    """
    Handles logit array operations.
    Provides methods for masking logits and selecting the best token.
    """

    @staticmethod
    def mask_logits(
        logits: List[float],
        allowed_ids: Set[int]
    ) -> List[float]:
        """
        Masks logit values.
        Sets all logits to -inf except for those in allowed_ids.
        """
        masked = [float('-inf')] * len(logits)
        for tid in allowed_ids:
            if 0 <= tid < len(logits):
                masked[tid] = logits[tid]
        return masked

    @staticmethod
    def select_best_token(logits: List[float]) -> int:
        """
        Selects the best token ID.
        Returns the index of the highest-scoring token from the logits.
        """
        return int(np.argmax(logits))
