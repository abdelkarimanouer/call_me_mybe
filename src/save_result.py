from pydantic import BaseModel
from typing import Dict, Any, List
import json
import os


class Result(BaseModel):
    """
    Represents a single function call result.
    Contains the prompt, the function name, and its extracted parameters.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]


class ResultSaver:
    """
    Manages saving function call results.
    Outputs a list of Result objects to a specified JSON file.
    """

    @staticmethod
    def save_results(results: List[Result], output_path: str) -> None:
        """
        Saves results to a JSON file.
        Creates parent directories if necessary and dumps the data.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_data = [result.model_dump() for result in results]

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
