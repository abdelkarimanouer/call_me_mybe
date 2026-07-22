from llm_sdk import Small_LLM_Model
from typing import List, Dict
import numpy as np


class Generate:

    @staticmethod
    def get_full_prompt_for_name(prompt: str, parse_fun_def: Dict) -> str:
        function_names = "\n".join(
            f"- {fun['name']}"
            for fun in parse_fun_def
        )

        return (
            "Choose the single best matching function from the list below.\n"
            "Return ONLY the function name.\n"
            "Do not return anything else.\n"
            "If none match, return ONLY: NONE.\n\n"
            f"Functions:\n{function_names}\n\n"
            f"User request:\n{prompt}"
        )

    @staticmethod
    def get_fun_name(model: Small_LLM_Model, prompt: str,
                     funs_defintions: List) -> str:
        prompt_name = Generate.get_full_prompt_for_name(prompt,
                                                        funs_defintions)
        ids = model.encode(prompt_name)[0].tolist()
        text = ""

        while True:
            logits = model.get_logits_from_input_ids(ids)
            masked = [float('-inf') for _ in logits]
            for id in ids:
                masked[id] = logits[id]
            bst = np.argmax(logits)
            ids.append(bst)
            text += model.decode(bst)
            for f in funs_defintions:
                if f['name'] in text:
                    return f['name']

    @staticmethod
    def run_generate(input_tests: List, funs_def: List) -> None:
        model: Small_LLM_Model = Small_LLM_Model()

        steps = {
            1: model.encode('{\n\t"prompt": ')[0].tolist(),
            2: [model.encode(f'"{p}",')[0].tolist() for p in input_tests],
            3: model.encode('\n\t"name": ')[0].tolist(),
        }

        l_prompts = len(input_tests)
        for i in range(l_prompts):
            full_ids = []
            for s in range(1, len(steps) + 1):
                if s == 1:
                    prompt_ids = steps[s]
                    full_ids.extend(prompt_ids)
                elif s == 2:
                    ids = steps[s][i]
                    full_ids.extend(ids)
                elif s == 3:
                    name_ids = steps[s]
                    full_ids.extend(name_ids)

            print(model.decode(full_ids))
