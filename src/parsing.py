from typing import List, Dict, Literal, Any
from pydantic import BaseModel, ConfigDict, ValidationError
import json
import argparse
import sys


class InputTest(BaseModel):
    """
    Represents an input test case.
    Validates the user prompt structure.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str


class ParamType(BaseModel):
    """
    Represents a parameter type definition.
    Validates allowed types for function parameters.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """
    Represents a function definition schema.
    Validates function name, description, parameters, and returns.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, ParamType]
    returns: ParamType


class Parsing:
    """
    Handles command-line arguments and JSON file parsing.
    Loads and validates input tests and function definitions.
    """

    def parse_args(self) -> Dict[str, str]:
        """
        Parses command-line arguments.
        Returns a dictionary containing input, output, and definition paths.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition",
                            default="data/input/functions_definition.json")
        parser.add_argument("--input",
                            default="data/input/function_calling_tests.json")
        parser.add_argument("--output",
                            default="data/output/function_calls.json")

        args = parser.parse_args()

        arguments: Dict[str, str] = {}
        arguments['fun_def'] = args.functions_definition
        arguments['input'] = args.input
        arguments['output'] = args.output
        return arguments

    def __load_json_list(self, path_file: str) -> List[Any]:
        """
        Loads a JSON file containing a list of objects.
        Exits the program if an error occurs.
        """
        try:
            with open(path_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR]: {e}")
            exit(1)
        except (FileNotFoundError, PermissionError) as e:
            print(f"[ERROR]: {e}")
            exit(1)
        except Exception as e:
            print(f"[ERROR]: {e}")
            exit(1)

        if not isinstance(data, list) or len(data) == 0:
            print(f"[ERROR]: {path_file} must contain a non-empty list",
                  file=sys.stderr)
            exit(1)

        return data

    def __build_input_test(self, item: Dict[str, Any]) -> str:
        """
        Builds and validates an InputTest object.
        Returns the parsed prompt string.
        """
        try:
            test = InputTest(**item)
        except ValidationError as e:
            print(f"[ERROR]: {e}", file=sys.stderr)
            exit(1)

        prompt = test.prompt.strip()
        if not prompt:
            print("[ERROR]: prompt is empty or whitespace only",
                  file=sys.stderr)
            exit(1)

        return prompt

    def __build_fun_def(self, item: Dict[str, Any]) -> FunctionDefinition:
        """
        Builds and validates a FunctionDefinition object.
        Ensures valid naming and parameter structure.
        """
        try:
            fun_def = FunctionDefinition(**item)
        except ValidationError as e:
            print(f"[ERROR]: {e}", file=sys.stderr)
            exit(1)

        if not fun_def.name.isidentifier() or not fun_def.description.strip():
            print("[ERROR]: Invalid name/description",
                  file=sys.stderr)
            exit(1)

        for param_name in fun_def.parameters:
            if not param_name.strip():
                print("[ERROR]: parameter name is empty or whitespace only",
                      file=sys.stderr)
                exit(1)

        return fun_def

    def get_input_tests(self, path_file: str) -> List[str]:
        """
        Loads and retrieves input tests from a file.
        Returns a list of prompt strings.
        """
        raw_data = self.__load_json_list(path_file)
        return [self.__build_input_test(item) for item in raw_data]

    def get_funs_definition(self, path_file: str) -> List[FunctionDefinition]:
        """
        Loads and retrieves function definitions from a file.
        Returns a list of FunctionDefinition objects.
        """
        raw_data = self.__load_json_list(path_file)
        return [self.__build_fun_def(item) for item in raw_data]
