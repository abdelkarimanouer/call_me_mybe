from llm_sdk import Small_LLM_Model
from typing import List


class Generate:
    @staticmethod
    def run_generate(input_tests: List, funs_defintions: List) -> None:
        model: Small_LLM_Model = Small_LLM_Model()

        steps = {
            1: model.encode('{"prompt": ')[0].tolist(),
            2: [model.encode(f'"{p}",')[0].tolist() for p in input_tests]
        }

        l_prompts = len(input_tests)
        for i in range(l_prompts):
            full_ids = []
            for s in range(1, len(steps) + 1):
                if s == 1:
                    static_ids = steps[s]
                    full_ids.extend(static_ids)
                elif s == 2:
                    ids = steps[s][i]
                    full_ids.extend(ids)
            print(model.decode(full_ids))
