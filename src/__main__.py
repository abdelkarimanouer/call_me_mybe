from src.vocab import get_id_token
from src.parsing import Parsing

from llm_sdk import Small_LLM_Model


def main() -> None:
    arguments = Parsing.parse_args()

    parse_fun_def = Parsing.load_json_data(arguments['fun_def'])
    parse_input_tests = Parsing.load_json_data(arguments['input'])

    # model: Small_LLM_Model = Small_LLM_Model()

    # id_token = get_id_token(model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"[ERROR]: {e}")
    except Exception as e:
        print(f"[ERROR]: {e}")
