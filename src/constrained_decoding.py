from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import List, Dict, Set, Tuple
from .vocab import Vocab
from .logit import Logit


class ConstrainedDecoding:
    """
    Manages constrained decoding.
    Guides the model output to generate structured JSON values token-by-token.
    """

    @staticmethod
    def generate_string_value(
        model: Small_LLM_Model,
        input_ids: List[int],
        id_token: Dict[int, str],
        token_lookup: Dict[str, int]
    ) -> Tuple[str, List[int]]:
        """
        Generates a JSON string value.
        Uses constrained decoding to emit characters until closing quote.
        """
        quote_ids = Vocab.find_tokens_for_exact_string('"', token_lookup)  # noqa: E501
        if not quote_ids:
            quote_ids = Vocab.find_tokens_for_exact_string('â\x80\x9c', token_lookup)  # noqa: E501
        if quote_ids:
            input_ids = input_ids + [quote_ids[0]]

        result_chars: List[str] = []
        max_tokens = 150

        for _ in range(max_tokens):
            logits = model.get_logits_from_input_ids(input_ids)

            allowed: Set[int] = set()

            for tid, token_str in id_token.items():
                if '\n' in token_str:
                    continue
                allowed.add(tid)

            masked = Logit.mask_logits(logits, allowed)
            chosen = Logit.select_best_token(masked)
            chosen_str = id_token.get(chosen, '')

            if not chosen_str:
                break

            if '"' in chosen_str:
                clean_str = chosen_str.split('"')[0].replace('Ġ', ' ')
                result_chars.append(clean_str)
                input_ids = input_ids + [chosen]
                break

            result_chars.append(chosen_str.replace('Ġ', ' '))
            input_ids = input_ids + [chosen]

        return ''.join(result_chars), input_ids

    @staticmethod
    def generate_number_value(
        model: Small_LLM_Model,
        input_ids: List[int],
        id_token: Dict[int, str]
    ) -> Tuple[str, List[int]]:
        """
        Generates a JSON number value.
        Allows digits, signs, decimals, and scientific notation constraints.
        """
        number_chars: List[str] = []
        max_tokens = 30
        has_dot = False
        has_digit = False

        for _ in range(max_tokens):
            logits = model.get_logits_from_input_ids(input_ids)
            allowed: Set[int] = set()

            for tid, token_str in id_token.items():
                valid = True
                temp_dot = has_dot
                temp_digit = has_digit

                for ch in token_str:
                    if ch.isdigit():
                        temp_digit = True
                    elif ch == '.' and not temp_dot:
                        temp_dot = True
                    elif ch == '-' and not number_chars and not temp_digit:
                        pass
                    elif ch == ' ' and not number_chars:
                        pass
                    else:
                        valid = False
                        break

                stripped = token_str.replace(' ', '')
                if valid and stripped:
                    allowed.add(tid)

            if has_digit:
                best_unconstrained = Logit.select_best_token(logits)
                best_str = id_token.get(best_unconstrained, '')
                stripped_best = best_str.strip()
                is_numeric_continuation = all(
                    c.isdigit() or c == '.' or c == '-' or c == 'e' or c == 'E'
                    for c in stripped_best
                ) if stripped_best else False

                if not is_numeric_continuation:
                    break

            if not allowed:
                break

            masked = Logit.mask_logits(logits, allowed)
            chosen = Logit.select_best_token(masked)
            chosen_str = id_token.get(chosen, '')

            clean = chosen_str.lstrip(' ')
            number_chars.append(clean)
            if '.' in clean:
                has_dot = True
            if any(c.isdigit() for c in clean):
                has_digit = True

            input_ids = input_ids + [chosen]

        return ''.join(number_chars), input_ids

    @staticmethod
    def generate_boolean_value(
        model: Small_LLM_Model,
        input_ids: List[int],
        id_token: Dict[int, str]
    ) -> Tuple[str, List[int]]:
        """
        Generates a JSON boolean value.
        Constrains output to exactly true or false literals.
        """
        allowed: Set[int] = set()
        for tid, token_str in id_token.items():
            clean = token_str.lstrip(' ')
            if clean and ('true'.startswith(clean) or
                          'false'.startswith(clean)):
                allowed.add(tid)

        logits = model.get_logits_from_input_ids(input_ids)
        masked = Logit.mask_logits(logits, allowed)
        chosen = Logit.select_best_token(masked)
        chosen_str = id_token.get(chosen, '').lstrip(' ')

        if chosen_str.startswith('t'):
            target = 'true'
        else:
            target = 'false'

        generated = chosen_str
        input_ids = input_ids + [chosen]

        while len(generated) < len(target):
            remaining = target[len(generated):]
            allowed_next: Set[int] = set()
            for tid, token_str in id_token.items():
                clean = token_str.lstrip(' ')
                if clean and remaining.startswith(clean):
                    allowed_next.add(tid)

            if not allowed_next:
                break

            logits = model.get_logits_from_input_ids(input_ids)
            masked = Logit.mask_logits(logits, allowed_next)
            chosen = Logit.select_best_token(masked)
            chosen_str = id_token.get(chosen, '').lstrip(' ')
            generated += chosen_str
            input_ids = input_ids + [chosen]

        return target, input_ids
