from typing import List, Dict
import json
import argparse


class Parsing:

    @staticmethod
    def parse_args() -> Dict:

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

    @staticmethod
    def get_input_tests(path_file: str) -> List:
        try:
            with open(path_file, "r") as f:
                tests = json.load(f)
                input_tests = []
                for t in tests:
                    input_tests.append(t["prompt"])
                return input_tests
        except json.JSONDecodeError as e:
            print(f"[ERROR]: {e}")
            exit()
        except (FileNotFoundError, PermissionError) as e:
            print(f"[ERROR]: {e}")
            exit()
        except Exception as e:
            print(f"[ERROR]: {e}")
            exit()

    @staticmethod
    def get_funs_definition(path_file: str) -> List:
        try:
            with open(path_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR]: {e}")
            exit()
        except (FileNotFoundError, PermissionError) as e:
            print(f"[ERROR]: {e}")
            exit()
        except Exception as e:
            print(f"[ERROR]: {e}")
            exit()
