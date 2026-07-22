from typing import List, Dict, Any
import json
import argparse
import sys
import regex


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
    def check_input_tests(t: Any) -> bool:
        if not isinstance(t, dict) or len(t) != 1:
            return False

        for key in t:
            if bool(regex.match("^prompt$", str(key))) and t[key]:
                return True
            else:
                return False

    @staticmethod
    def get_input_tests(path_file: str) -> List:
        try:
            with open(path_file, "r") as f:
                tests = json.load(f)
                input_tests = []
                for t in tests:
                    if Parsing.check_input_tests(t) is False:
                        print("""[ERROR]: Invalid test input. \
Please enter it like this {\"Prompt\": \"<your prompt>\"} \
""", file=sys.stderr)
                        exit()
                    else:
                        for k in t:
                            input_tests.append(t[k])
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
    def check_funs_def(funs_def: Any) -> bool:
        ...

    @staticmethod
    def get_funs_definition(path_file: str) -> List:
        try:
            with open(path_file, "r") as f:
                funs_def = json.load(f)
                if Parsing.check_funs_def(funs_def):
                    return funs_def
        except json.JSONDecodeError as e:
            print(f"[ERROR]: {e}")
            exit()
        except (FileNotFoundError, PermissionError) as e:
            print(f"[ERROR]: {e}")
            exit()
        except Exception as e:
            print(f"[ERROR]: {e}")
            exit()
