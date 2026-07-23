from .parsing import Parsing
from .generate import Generate
from typing import List, Any
import time


class Main:
    """
    Main entry point for the application.
    Orchestrates parsing and generation workflows.
    """

    @staticmethod
    def main() -> None:
        """
        Executes the main pipeline.
        Parses arguments and runs the generation process.
        """
        start_time = time.time()
        parser = Parsing()
        arguments = parser.parse_args()

        parse_input_tests: List[str] = parser.get_input_tests(
            arguments['input']
        )
        parse_fun_def: List[Any] = parser.get_funs_definition(
            arguments['fun_def']
        )

        Generate.run_generate(
            parse_input_tests, parse_fun_def, arguments['output']
        )
        end_time = time.time()

        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"time:{minutes}m{seconds}s")


if __name__ == "__main__":
    try:
        Main.main()
    except KeyboardInterrupt as e:
        print(f"[ERROR]: {e}")
    except Exception as e:
        print(f"[ERROR]: {e}")
