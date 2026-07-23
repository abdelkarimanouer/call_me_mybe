"""
Constrained decoding module for function calling.
Generates structured JSON function calls via token masking.
"""

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import List, Dict, Any
from .vocab import Vocab
from .save_result import save_results, Result
from .parameters import Parameters
from .fun_name import FunName


class Generate:
    """
    Orchestrates function call generation.
    Main entry point for converting prompts into function calls.
    """

    @staticmethod
    def process_single_prompt(
        model: Small_LLM_Model,
        prompt: str,
        funs_def: List[Any],
        id_token: Dict[int, str],
        token_lookup: Dict[str, int]
    ) -> Result:
        """
        Processes a single user prompt.
        Identifies the function and extracts its parameters.
        """
        print(f"  Processing: {prompt}")

        func_name = FunName.get_fun_name(model, prompt, funs_def, id_token)
        print(f"  → Function: {func_name}")

        if func_name == "NONE":
            return Result(prompt=prompt, name="NONE", parameters={})

        func_def = FunName.find_function_def(func_name, funs_def)

        parameters = Parameters.extract_all_parameters(
            model, prompt, func_name, func_def,
            id_token, token_lookup
        )
        print(f"  → Parameters: {parameters}")

        return Result(prompt=prompt, name=func_name, parameters=parameters)

    @staticmethod
    def run_generate(
        input_tests: List[str],
        funs_def: List[Any],
        output_path: str = "data/output/function_calls.json"
    ) -> None:
        """
        Runs the full generation pipeline.
        Iterates over inputs, processes them, and saves results.
        """
        print("Loading model...")
        model: Small_LLM_Model = Small_LLM_Model()

        print("Building vocabulary lookup...")
        id_token = Vocab.get_id_token(model)
        token_lookup = Vocab.build_token_lookup(id_token)

        results: List[Result] = []

        print(f"Processing {len(input_tests)} prompts...")
        for i, prompt in enumerate(input_tests):
            print(f"\n[{i + 1}/{len(input_tests)}]")
            try:
                result = Generate.process_single_prompt(
                    model, prompt, funs_def,
                    id_token, token_lookup
                )
                results.append(result)
            except Exception as e:
                print(f"  [WARNING] Failed: {e}")
                names = [
                    f['name'] if isinstance(f, dict) else f.name
                    for f in funs_def
                ]
                results.append(Result(
                    prompt=prompt,
                    name=names[0] if names else "unknown",
                    parameters={}
                ))

        print(f"\nSaving {len(results)} results to {output_path}")
        save_results(results, output_path)
        print("Done!")
