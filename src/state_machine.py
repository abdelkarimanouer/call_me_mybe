from typing import Dict


def get_full_prompt(prompt: str, parse_fun_def: Dict) -> str:
    instruction = (
        'You are a function-calling assistant. Based on the user\'s question, '
        'choose the most appropriate function from the list below '
        'and provide the correct arguments. '
        'Respond only with a JSON object in this exact shape: '
        '{"name": "<function name>", "parameters": {...}}.'
    )

    functions_text = ""
    for fun in parse_fun_def:
        params_text = ", ".join(
            f"{param_name} ({param_info['type']})"
            for param_name, param_info in fun["parameters"].items()
        )
        functions_text += (f"- {fun['name']}: {fun['description']}\n  "
                           f"Parameters: {params_text}\n\n")

    full_prompt = (
        f"{instruction}\n\n"
        f"Available functions:\n\n{functions_text}\n"
        f"User question: {prompt}\n\n"
        f"Answer: {{"
    )

    return full_prompt
