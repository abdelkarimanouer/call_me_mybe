import argparse
from src.parsing import Parsing


def main() -> None:
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

    parse_fun_def = Parsing.load_json_data(arguments['fun_def'])
    parse_input_tests = Parsing.load_json_data(arguments['input'])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"[ERROR]: {e}")
    except Exception as e:
        print(f"[ERROR]: {e}")
