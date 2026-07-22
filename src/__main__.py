from .parsing import Parsing
from .generate import Generate
from typing import List


def main() -> None:
    parser = Parsing()
    arguments = parser.parse_args()

    parse_input_tests: List = parser.get_input_tests(arguments['input'])
    parse_fun_def: List = parser.get_funs_definition(arguments['fun_def'])

    # Generate.run_generate(parse_input_tests, parse_fun_def)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"[ERROR]: {e}")
    except Exception as e:
        print(f"[ERROR]: {e}")
