from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import List, Dict, Any
from .logit import Logit


class FunName:
    """
    Manages function name resolution.
    Handles matching user prompts to function names and definitions.
    """

    @staticmethod
    def get_full_prompt_for_name(
        prompt: str,
        parse_fun_def: Any
    ) -> str:
        """
        Builds a prompt for function name selection.
        Presents the function list and asks for the best match.
        """
        function_names = "\n".join(
            f"- {fun['name'] if isinstance(fun, dict) else fun.name}: "
            f"{fun['description'] if isinstance(fun, dict) else fun.description}"  # noqa: E501
            for fun in parse_fun_def
        )

        return (
            "Choose the single best matching function "
            "from the list below.\n"
            "Return ONLY the function name.\n"
            "Do not return anything else.\n"
            "If none match, return ONLY: NONE.\n\n"
            f"Functions:\n{function_names}\n\n"
            f"User request:\n{prompt}\n\n"
            "Function name: "
        )

    @staticmethod
    def get_fun_name(
        model: Small_LLM_Model,
        prompt: str,
        funs_definitions: List[Any],
        id_token: Dict[int, str]
    ) -> str:
        """
        Selects the best matching function name.
        Uses the model with constrained logit generation.
        """
        prompt_name = FunName.get_full_prompt_for_name(
            prompt, funs_definitions
        )
        ids = model.encode(prompt_name)[0].tolist()

        names = [
            f['name'] if isinstance(f, dict) else f.name
            for f in funs_definitions
        ]
        names.append("NONE")

        text = ""
        for _ in range(100):
            logits = model.get_logits_from_input_ids(ids)

            allowed = set()
            for tid, token_str in id_token.items():
                clean_str = token_str.lstrip(' ')
                if not clean_str:
                    continue

                new_text = text + clean_str
                if any(name.startswith(new_text) for name in names):
                    allowed.add(tid)

            if not allowed:
                break

            masked = Logit.mask_logits(logits, allowed)
            chosen = Logit.select_best_token(masked)
            chosen_str = id_token.get(chosen, '').lstrip(' ')

            text += chosen_str
            ids.append(chosen)

            if text in names:
                return str(text)

        if text in names:
            return str(text)
        return str(names[0])

    @staticmethod
    def find_function_def(
        func_name: str,
        funs_def: List[Any]
    ) -> Dict[str, Any]:
        """
        Finds the function definition for a given name.
        Iterates over definitions and returns the matching dictionary.
        """
        for f in funs_def:
            f_dict = f if isinstance(f, dict) else f.model_dump()
            if f_dict['name'] == func_name:
                return f_dict
        raise ValueError(f"Function '{func_name}' not found in definitions")
