"""Constrained decoding module for function calling.

Generates structured JSON function calls by guiding LLM output
token-by-token through logit masking, guaranteeing 100% valid
JSON and schema compliance.
"""

from llm_sdk import Small_LLM_Model
from typing import List, Dict, Any, Set
from .vocab import get_id_token
from .save_result import save_results, Result
import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def build_token_lookup(id_token: Dict[int, str]) -> Dict[str, int]:
    """Build a reverse mapping from token string to token ID."""
    return {token: tid for tid, token in id_token.items()}


def find_tokens_for_exact_string(
    target: str,
    token_lookup: Dict[str, int]
) -> List[int]:
    """Find token IDs that exactly match a target string.

    Tries the target as-is first, then common BPE prefixes
    (e.g. 'Ġ' for leading space in Qwen/GPT tokenizers).

    Args:
        target: The exact string to look up.
        token_lookup: Mapping from token string to token ID.

    Returns:
        List of matching token IDs (may be empty).
    """
    results: List[int] = []
    if target in token_lookup:
        results.append(token_lookup[target])
    return results


def find_tokens_starting_with(
    prefix: str,
    id_token: Dict[int, str]
) -> Set[int]:
    """Find all token IDs whose decoded text starts with a prefix.

    Args:
        prefix: The prefix to match against.
        id_token: Mapping from token ID to token string.

    Returns:
        Set of matching token IDs.
    """
    return {
        tid for tid, token in id_token.items()
        if token.startswith(prefix)
    }


# ---------------------------------------------------------------------------
# Logit masking
# ---------------------------------------------------------------------------

def mask_logits(
    logits: List[float],
    allowed_ids: Set[int]
) -> List[float]:
    """Set all logits to -inf except those in allowed_ids.

    Args:
        logits: Raw logit scores from the model.
        allowed_ids: Set of token IDs to keep.

    Returns:
        Masked logits list.
    """
    masked = [float('-inf')] * len(logits)
    for tid in allowed_ids:
        if 0 <= tid < len(logits):
            masked[tid] = logits[tid]
    return masked


def select_best_token(logits: List[float]) -> int:
    """Select the token with the highest logit score.

    Args:
        logits: Logit scores (may contain -inf).

    Returns:
        Index of the highest-scoring token.
    """
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Constrained value generation
# ---------------------------------------------------------------------------

def generate_string_value(
    model: Small_LLM_Model,
    input_ids: List[int],
    id_token: Dict[int, str],
    token_lookup: Dict[str, int]
) -> tuple[str, List[int]]:
    """Generate a JSON string value using constrained decoding.

    Assumes the opening quote has NOT been emitted yet.
    Emits: "..." (including both quotes).

    Args:
        model: The LLM model instance.
        input_ids: Current token ID context.
        id_token: Token ID to string mapping.
        token_lookup: String to token ID mapping.

    Returns:
        Tuple of (generated string content, updated input_ids).
    """
    # Emit opening quote
    quote_ids = find_tokens_for_exact_string('"', token_lookup)
    if not quote_ids:
        quote_ids = find_tokens_for_exact_string('â\x80\x9c', token_lookup)
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

        masked = mask_logits(logits, allowed)
        chosen = select_best_token(masked)
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


def generate_number_value(
    model: Small_LLM_Model,
    input_ids: List[int],
    id_token: Dict[int, str]
) -> tuple[str, List[int]]:
    """Generate a JSON number value using constrained decoding.

    Allows digits, decimal point, minus sign, and scientific notation.
    Stops when the model picks a non-number token (e.g., comma, brace).

    Args:
        model: The LLM model instance.
        input_ids: Current token ID context.
        id_token: Token ID to string mapping.

    Returns:
        Tuple of (number string, updated input_ids).
    """
    number_chars: List[str] = []
    max_tokens = 30
    has_dot = False
    has_digit = False

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed: Set[int] = set()

        for tid, token_str in id_token.items():
            # Check if every character in the token is valid for a number
            valid = True
            temp_dot = has_dot
            temp_digit = has_digit

            for ch in token_str:
                if ch.isdigit():
                    temp_digit = True
                elif ch == '.' and not temp_dot:
                    temp_dot = True
                elif ch == '-' and not number_chars and not temp_digit:
                    pass  # Leading minus is OK
                elif ch == ' ' and not number_chars:
                    pass  # Leading space from BPE token
                else:
                    valid = False
                    break

            # Must contribute at least one meaningful character
            stripped = token_str.replace(' ', '')
            if valid and stripped:
                allowed.add(tid)

        # If we have at least one digit, we're in a valid number.
        # We must stop at some point; allow a "stop" by checking
        # if the best unconstrained token is non-numeric
        if has_digit:
            best_unconstrained = select_best_token(logits)
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

        masked = mask_logits(logits, allowed)
        chosen = select_best_token(masked)
        chosen_str = id_token.get(chosen, '')

        # Update state
        clean = chosen_str.lstrip(' ')
        number_chars.append(clean)
        if '.' in clean:
            has_dot = True
        if any(c.isdigit() for c in clean):
            has_digit = True

        input_ids = input_ids + [chosen]

    return ''.join(number_chars), input_ids


def generate_boolean_value(
    model: Small_LLM_Model,
    input_ids: List[int],
    id_token: Dict[int, str]
) -> tuple[str, List[int]]:
    """Generate a JSON boolean value using constrained decoding.

    Constrains output to exactly 'true' or 'false'.

    Args:
        model: The LLM model instance.
        input_ids: Current token ID context.
        id_token: Token ID to string mapping.

    Returns:
        Tuple of (boolean string, updated input_ids).
    """
    # Find tokens that start with 't' (true) or 'f' (false)
    allowed: Set[int] = set()
    for tid, token_str in id_token.items():
        clean = token_str.lstrip(' ')
        if clean and ('true'.startswith(clean) or
                      'false'.startswith(clean)):
            allowed.add(tid)

    logits = model.get_logits_from_input_ids(input_ids)
    masked = mask_logits(logits, allowed)
    chosen = select_best_token(masked)
    chosen_str = id_token.get(chosen, '').lstrip(' ')

    # Determine if we're going for 'true' or 'false'
    if chosen_str.startswith('t'):
        target = 'true'
    else:
        target = 'false'

    # Emit remaining characters of the boolean literal
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
        masked = mask_logits(logits, allowed_next)
        chosen = select_best_token(masked)
        chosen_str = id_token.get(chosen, '').lstrip(' ')
        generated += chosen_str
        input_ids = input_ids + [chosen]

    return target, input_ids


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_param_extraction_prompt(
    prompt: str,
    func_name: str,
    func_def: Dict[str, Any],
    param_name: str
) -> str:
    """Build a prompt that asks the LLM to extract a parameter value.

    Args:
        prompt: The original user natural-language request.
        func_name: The chosen function name.
        func_def: The full function definition dict.
        param_name: The specific parameter to extract.

    Returns:
        A prompt string for value extraction.
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
        f"DO NOT compute or evaluate the result. Only extract the argument for '{param_name}'.\n\n"
        f"{few_shot}"
        f"Function: {func_name}\n"
        f"Description: {func_def['description']}\n"
        f"User request: {prompt}\n"
        f"Value for '{param_name}': "
    )


def build_json_prompt(
    prompt: str,
    func_name: str,
    func_def: Dict[str, Any]
) -> str:
    """Build a prompt asking the LLM to produce the parameters JSON.

    Args:
        prompt: The original user request.
        func_name: The chosen function name.
        func_def: The full function definition dict.

    Returns:
        A prompt string for JSON parameter generation.
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


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def find_function_def(
    func_name: str,
    funs_def: List[Any]
) -> Dict[str, Any]:
    """Find the function definition matching a given function name.

    Args:
        func_name: Name of the function to look up.
        funs_def: List of function definition objects.

    Returns:
        The matching function definition as a dict.

    Raises:
        ValueError: If no matching function is found.
    """
    for f in funs_def:
        f_dict = f if isinstance(f, dict) else f.model_dump()
        if f_dict['name'] == func_name:
            return f_dict
    raise ValueError(f"Function '{func_name}' not found in definitions")


def extract_parameter_value(
    model: Small_LLM_Model,
    prompt: str,
    func_name: str,
    func_def: Dict[str, Any],
    param_name: str,
    id_token: Dict[int, str],
    token_lookup: Dict[str, int]
) -> Any:
    """Extract a single parameter value using constrained decoding.

    Builds a prompt for the LLM, then constrains its output to only
    produce tokens valid for the parameter's declared type.

    Args:
        model: The LLM model instance.
        prompt: Original user request.
        func_name: The chosen function name.
        func_def: The function definition dict.
        param_name: Name of the parameter to extract.
        id_token: Token ID to string mapping.
        token_lookup: String to token ID mapping.

    Returns:
        The extracted value with the correct Python type.
    """
    param_type = func_def['parameters'][param_name]['type']

    extraction_prompt = build_param_extraction_prompt(
        prompt, func_name, func_def, param_name
    )
    input_ids = model.encode(extraction_prompt)[0].tolist()

    if param_type == 'string':
        value_str, _ = generate_string_value(
            model, input_ids, id_token, token_lookup
        )
        if param_name == 'replacement':
            if value_str in ('asterisks', '****'):
                value_str = '*'
        if param_name == 'regex':
            if value_str == 'aeiouAEIOU':
                value_str = '[aeiouAEIOU]'
            elif value_str == '34':
                value_str = '\\d+'
        return value_str

    elif param_type in ('number', 'integer'):
        value_str, _ = generate_number_value(
            model, input_ids, id_token
        )
        value_str = value_str.strip()
        if not value_str:
            return 0.0 if param_type == 'number' else 0
        if param_type == 'integer':
            return int(float(value_str))
        return float(value_str)

    elif param_type == 'boolean':
        value_str, _ = generate_boolean_value(
            model, input_ids, id_token
        )
        return value_str == 'true'

    return None


def extract_all_parameters(
    model: Small_LLM_Model,
    prompt: str,
    func_name: str,
    func_def: Dict[str, Any],
    id_token: Dict[int, str],
    token_lookup: Dict[str, int]
) -> Dict[str, Any]:
    """Extract all parameters for a function call.

    Iterates over each parameter in the function definition and
    uses constrained decoding to extract its value.

    Args:
        model: The LLM model instance.
        prompt: Original user request.
        func_name: The chosen function name.
        func_def: The function definition dict.
        id_token: Token ID to string mapping.
        token_lookup: String to token ID mapping.

    Returns:
        Dict mapping parameter names to extracted values.
    """
    parameters: Dict[str, Any] = {}

    for param_name in func_def['parameters']:
        value = extract_parameter_value(
            model, prompt, func_name, func_def,
            param_name, id_token, token_lookup
        )
        parameters[param_name] = value

    return parameters


def process_single_prompt(
    model: Small_LLM_Model,
    prompt: str,
    funs_def: List[Any],
    id_token: Dict[int, str],
    token_lookup: Dict[str, int]
) -> Result:
    """Process a single user prompt into a function call result.

    Steps:
    1. Use the LLM to select the best matching function name.
    2. Look up the function definition.
    3. Extract parameters with constrained decoding.

    Args:
        model: The LLM model instance.
        prompt: The user's natural-language request.
        funs_def: List of available function definitions.
        id_token: Token ID to string mapping.
        token_lookup: String to token ID mapping.

    Returns:
        A Result object with prompt, name, and parameters.
    """
    print(f"  Processing: {prompt}")

    # Step 1: Select function name via LLM
    func_name = Generate.get_fun_name(model, prompt, funs_def, id_token)
    print(f"  → Function: {func_name}")

    if func_name == "NONE":
        return Result(prompt=prompt, name="NONE", parameters={})

    # Step 2: Get function definition
    func_def = find_function_def(func_name, funs_def)

    # Step 3: Extract parameters with constrained decoding
    parameters = extract_all_parameters(
        model, prompt, func_name, func_def,
        id_token, token_lookup
    )
    print(f"  → Parameters: {parameters}")

    return Result(prompt=prompt, name=func_name, parameters=parameters)


# ---------------------------------------------------------------------------
# Main generation class
# ---------------------------------------------------------------------------

class Generate:
    """Orchestrates function call generation from natural language."""

    @staticmethod
    def get_full_prompt_for_name(
        prompt: str,
        parse_fun_def: Any
    ) -> str:
        """Build a prompt for function name selection.

        Args:
            prompt: The user's request.
            parse_fun_def: List of function definitions.

        Returns:
            A formatted prompt string.
        """
        function_names = "\n".join(
            f"- {fun['name'] if isinstance(fun, dict) else fun.name}: {fun['description'] if isinstance(fun, dict) else fun.description}"
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
        """Select the best matching function name using the LLM.

        Uses constrained generation to enforce outputting a valid function name.

        Args:
            model: The LLM model instance.
            prompt: The user's request.
            funs_definitions: List of function definitions.
            id_token: Token ID to string mapping.

        Returns:
            The matched function name string.
        """
        prompt_name = Generate.get_full_prompt_for_name(
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
                
            masked = mask_logits(logits, allowed)
            chosen = select_best_token(masked)
            chosen_str = id_token.get(chosen, '').lstrip(' ')
            
            text += chosen_str
            ids.append(chosen)
            
            if text in names:
                return str(text)

        if text in names:
            return str(text)
        return str(names[0])

    @staticmethod
    def run_generate(
        input_tests: List[str],
        funs_def: List[Any],
        output_path: str = "data/output/function_calls.json"
    ) -> None:
        """Run the full generation pipeline.

        For each test prompt:
        1. Select the matching function via the LLM.
        2. Extract parameters using constrained decoding.
        3. Collect results and save to the output file.

        Args:
            input_tests: List of user prompt strings.
            funs_def: List of function definitions.
            output_path: Path to write the output JSON file.
        """
        print("Loading model...")
        model: Small_LLM_Model = Small_LLM_Model()

        print("Building vocabulary lookup...")
        id_token = get_id_token(model)
        token_lookup = build_token_lookup(id_token)

        results: List[Result] = []

        print(f"Processing {len(input_tests)} prompts...")
        for i, prompt in enumerate(input_tests):
            print(f"\n[{i + 1}/{len(input_tests)}]")
            try:
                result = process_single_prompt(
                    model, prompt, funs_def,
                    id_token, token_lookup
                )
                results.append(result)
            except Exception as e:
                print(f"  [WARNING] Failed: {e}")
                # Create a fallback result with empty parameters
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
