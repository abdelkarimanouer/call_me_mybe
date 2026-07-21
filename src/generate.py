from llm_sdk import Small_LLM_Model
from typing import List


class Generate:

    @staticmethod
    def get_fun_name(prompt: str, funs_defintions: List) -> str:
        fun_name = ""
        

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
                    fun_name = Generate.get_fun_name(input_tests[i], funs_def)
                    fun_name = f'"{fun_name}",'
                    fun_name_ids = model.encode(fun_name)[0].tolist()
                    full_ids.extend(fun_name_ids)

            print(model.decode(full_ids))
