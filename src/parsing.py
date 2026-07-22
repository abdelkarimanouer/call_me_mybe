from typing import List, Dict, Literal
from pydantic import (BaseModel, RootModel, ConfigDict,
                      field_validator, ValidationError)
import json
import argparse
import sys


class InputTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("prompt is empty or whitespace only")
        return stripped


class InputTestsList(RootModel[List[InputTest]]):
    pass


class ParamType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, ParamType]
    returns: ParamType

    @field_validator("name", "description")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("field must not be empty")
        return v

    @field_validator("parameters")
    @classmethod
    def param_names_not_blank(cls, v: Dict[str, ParamType]
                              ) -> Dict[str, ParamType]:
        for param_name in v:
            if not param_name:
                raise ValueError("parameter name must not be empty")
        return v


class FunctionsDefinitionList(RootModel[List[FunctionDefinition]]):
    pass


class Parsing:

    def parse_args(self) -> Dict:

        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition",
                            default="data/input/functions_definition.json")
        parser.add_argument("--input",
                            default="data/input/function_calling_tests.json")
        parser.add_argument("--output",
                            default="data/output/function_calls.json")

        args = parser.parse_args()

        arguments = {}
        arguments['fun_def'] = args.functions_definition
        arguments['input'] = args.input
        arguments['output'] = args.output
        return arguments

    def get_input_tests(self, path_file: str) -> List:
        try:
            with open(path_file, "r") as f:
                raw_data = json.load(f)
                parsed = InputTestsList.model_validate(raw_data)
                return [item.prompt for item in parsed.root]
        except ValidationError as e:
            print(f"[ERROR]: {e}", file=sys.stderr)
            exit()
        except json.JSONDecodeError as e:
            print(f"[ERROR]: {e}")
            exit()
        except (FileNotFoundError, PermissionError) as e:
            print(f"[ERROR]: {e}")
            exit()
        except Exception as e:
            print(f"[ERROR]: {e}")
            exit()

    def get_funs_definition(self, path_file: str) -> List:
        try:
            with open(path_file, "r") as f:
                raw_data = json.load(f)
                parsed = FunctionsDefinitionList.model_validate(raw_data)
                if len(parsed.root) == 0:
                    print("[ERROR]: functions definition file is empty.",
                          file=sys.stderr)
                    exit()
                return parsed.root
        except ValidationError as e:
            print(f"[ERROR]: {e}", file=sys.stderr)
            exit()
        except json.JSONDecodeError as e:
            print(f"[ERROR]: {e}")
            exit()
        except (FileNotFoundError, PermissionError) as e:
            print(f"[ERROR]: {e}")
            exit()
        except Exception as e:
            print(f"[ERROR]: {e}")
            exit()
