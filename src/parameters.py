from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import Dict, Any, List
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
        parameters: Dict,
    ) -> str:
        """
        Builds a parameter extraction prompt.
        Asks the model to extract a specific parameter from the request.
        Includes previously extracted parameters for context.
        """

        params_detail = "\n".join(
                f"- {name} (type: {spec['type']})"
                for name, spec in parameters.items()
            )
        return (
            f"You are a parameter extraction assistant.\n"
            f"Extract parameter values from the user request "
            f"and output them as a JSON object.\n\n"
            f"Function: {func_name}\n"
            f"Description: {func_def['description']}\n\n"
            f"Parameters:\n{params_detail}\n\n"
            f"User request: {prompt}\n\n"
            f"Rules:\n"
            f"- Output a single JSON object\n"
            f"- No explanation, no extra text\n\n"
            f"Output:"
        )

    @staticmethod
    def extract_parameter_value(
        model: Small_LLM_Model,
        input_ids: List[int],
        param_type: str,
        id_token: Dict[int, str],
        token_lookup: Dict[str, int],
    ) -> Any:
        """
        Extracts a single parameter value.
        Uses constrained decoding to match the expected parameter type.
        """

        if param_type == 'string':
            value_str, _ = ConstrainedDecoding.generate_string_value(
                model, input_ids, id_token, token_lookup
            )
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
    ) -> str:
        """
        Extracts all parameters for a given function call.
        Iterates over the definition and extracts values individually.
        """
        parameters: Dict[str, Any] = {}
        prompt = prompt.replace("\"", "\\\"")
        real_json = '{' + '"prompt":"' + prompt + '","name":"' + func_name
        real_json += '","parameters": {'

        extraction_prompt = Parameters.build_param_extraction_prompt(
            prompt, func_name, func_def, func_def['parameters'])
        input_ids = model.encode(extraction_prompt)[0].tolist()

        for param_name in func_def['parameters']:

            real_json += '"' + param_name + '": '

            param_type = func_def['parameters'][param_name]['type']

            input_ids += model.encode(real_json)[0].tolist()

            value = Parameters.extract_parameter_value(model, input_ids,
                                                       param_type,
                                                       id_token, token_lookup)
            if param_type == "string":
                real_json += '"' + value + '",'
            else:
                real_json += str(value) + ","
            parameters[param_name] = value
        real_json = real_json[:-1]
        real_json += "}}"

        return real_json
