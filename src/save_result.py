"""Result saving module for function call outputs."""

from pydantic import BaseModel
from typing import Dict, Any, List
import json
import os


class Result(BaseModel):
    """Represents a single function call result.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function to call.
        parameters: Dict of parameter names to extracted values.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]


def save_results(results: List[Result], output_path: str) -> None:
    """Save a list of Result objects to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        results: List of Result objects to save.
        output_path: File path for the output JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = [result.model_dump() for result in results]

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
