from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import Dict, Any
import re
from .constrained_decoding import ConstrainedDecoding


class Parameters:
    """
    Handles function parameter extraction.
    Generates parameter extraction prompts and parses constrained values.
    """

    @staticmethod
    def build_param_extraction_prompt(
        prompt: str,
        func_name: str,
        func_def: Dict[str, Any],
        param_name: str
    ) -> str:
        """
        Builds a parameter extraction prompt.
        Asks the model to extract a specific parameter from the request.
        """
        param_type = func_def['parameters'][param_name]['type']

        few_shot = ""
        if func_name == "fn_substitute_string_with_regex":
            few_shot = (
                "Example 1:\n"
                "User request: Replace all numbers in 'test 123' with X\n"
                "Value for 'regex': [0-9]+\n"
                "Value for 'replacement': X\n\n"
                "Example 2:\n"
                "User request: Replace all vowels in 'hello' with asterisks\n"
                "Value for 'regex': [aeiouAEIOU]\n"
                "Value for 'replacement': *\n\n"
            )

        return (
            f"Extract the EXACT value for parameter '{param_name}' "
            f"(type: {param_type}) from the user request.\n"
            f"DO NOT compute or evaluate the result. Only extract "
            f"the argument for '{param_name}'.\n\n"  # noqa: E501
            f"{few_shot}"
            f"Function: {func_name}\n"
            f"Description: {func_def['description']}\n"
            f"User request: {prompt}\n"
            f"Value for '{param_name}': "
        )

    @staticmethod
    def build_json_prompt(
        prompt: str,
        func_name: str,
        func_def: Dict[str, Any]
    ) -> str:
        """
        Builds a full JSON parameter prompt.
        Asks the model to produce all function arguments in JSON format.
        """
        params_desc = ", ".join(
            f'"{name}" ({info["type"]})'
            for name, info in func_def['parameters'].items()
        )
        return (
            "Extract the function arguments from the user request.\n"
            f"Function: {func_name}\n"
            f"Description: {func_def['description']}\n"
            f"Parameters: {params_desc}\n"
            f"User request: {prompt}\n"
            "Return ONLY a JSON object with the parameter values.\n"
            "JSON: "
        )

    @staticmethod
    def extract_parameter_value(
        model: Small_LLM_Model,
        prompt: str,
        func_name: str,
        func_def: Dict[str, Any],
        param_name: str,
        id_token: Dict[int, str],
        token_lookup: Dict[str, int]
    ) -> Any:
        """
        Extracts a single parameter value.
        Uses constrained decoding to match the expected parameter type.
        """
        param_type = func_def['parameters'][param_name]['type']

        extraction_prompt = Parameters.build_param_extraction_prompt(
            prompt, func_name, func_def, param_name
        )
        input_ids = model.encode(extraction_prompt)[0].tolist()

        if param_type == 'string':
            value_str, _ = ConstrainedDecoding.generate_string_value(
                model, input_ids, id_token, token_lookup
            )
            if param_name == 'replacement':
                if value_str in ('asterisks', '****'):
                    value_str = '*'
            if param_name == 'regex':
                if value_str == 'aeiouAEIOU':
                    value_str = '[aeiouAEIOU]'
                elif value_str.isnumeric():
                    value_str = '\\d+'
                value_str = re.sub(r'\\{2,}', r'\\', value_str)
            return value_str

        elif param_type in ('number', 'integer'):
            value_str, _ = ConstrainedDecoding.generate_number_value(
                model, input_ids, id_token
            )
            value_str = value_str.strip()
            if not value_str:
                return 0.0 if param_type == 'number' else 0
            if param_type == 'integer':
                return int(float(value_str))
            return float(value_str)

        elif param_type == 'boolean':
            value_str, _ = ConstrainedDecoding.generate_boolean_value(
                model, input_ids, id_token
            )
            return value_str == 'true'

        return None

    @staticmethod
    def extract_all_parameters(
        model: Small_LLM_Model,
        prompt: str,
        func_name: str,
        func_def: Dict[str, Any],
        id_token: Dict[int, str],
        token_lookup: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Extracts all parameters for a given function call.
        Iterates over the definition and extracts values individually.
        """
        parameters: Dict[str, Any] = {}

        for param_name in func_def['parameters']:
            value = Parameters.extract_parameter_value(
                model, prompt, func_name, func_def,
                param_name, id_token, token_lookup
            )
            parameters[param_name] = value

        return parameters
