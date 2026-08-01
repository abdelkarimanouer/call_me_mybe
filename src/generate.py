from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import List, Dict, Any
from .vocab import Vocab
from .parameters import Parameters
from .fun_name import FunName
import json
import os


class Generate:
    """
    Orchestrates function call generation.
    Main entry point for converting prompts into function calls.
    """

    @staticmethod
    def save_json_output(output_path: str, results: List[Dict]) -> None:
        """Save the result ouput in the output json file"""

        print(f"\nSaving {len(results)} results to {output_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print("Done!")

    @staticmethod
    def process_single_prompt(
        model: Small_LLM_Model,
        prompt: str,
        funs_def: List[Any],
        id_token: Dict[int, str],
        token_lookup: Dict[str, int]
    ) -> str:
        """
        Processes a single user prompt.
        Identifies the function and extracts its parameters.
        """
        print(f"  Processing: {prompt}")

        func_name = FunName.get_fun_name(model, prompt, funs_def, id_token)
        print(f"  → Function: {func_name}")

        func_def = FunName.find_function_def(func_name, funs_def)

        real_json = Parameters.extract_all_parameters(
            model, prompt, func_name, func_def,
            id_token, token_lookup
        )
        return str(real_json)

    @staticmethod
    def run_generate(
        input_tests: List[str],
        funs_def: List[Any],
        output_path: str
    ) -> None:
        """
        Runs the full generation pipeline.
        Iterates over inputs, processes them, and saves results.
        """
        print("Loading model...")
        model: Small_LLM_Model = Small_LLM_Model()

        id_token = Vocab.get_id_token(model)
        token_lookup = Vocab.build_token_lookup(id_token)

        results: List[Dict] = []

        print(f"Processing {len(input_tests)} prompts...")
        for i, prompt in enumerate(input_tests):
            print(f"\n[{i + 1}/{len(input_tests)}]")
            real_json = Generate.process_single_prompt(
                model, prompt, funs_def,
                id_token, token_lookup
            )

            my_result = json.loads(real_json)
            print(f"  → Output: {my_result}")
            results.append(my_result)

        Generate.save_json_output(output_path, results)
