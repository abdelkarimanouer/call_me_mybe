from src.parsing import Parsing
from llm_sdk import Small_LLM_Model


def main() -> None:
    arguments = Parsing.parse_args()

    parse_input_tests = Parsing.get_input_tests(arguments['input'])
    parse_fun_def = Parsing.get_funs_definition(arguments['fun_def'])

    # model: Small_LLM_Model = Small_LLM_Model()

    # prompt = "What is the sum of 2 and 3?"
    # steps = {
    #     1: model.encode('{"prompt": '),
    #     2: model.encode(prompt)
    # }
    # full_ids = []
    # l_steps = len(steps)
    # for i in range(1, l_steps + 1):
    #     full_ids.extend(steps[i][0].tolist())
    # print(model.decode(full_ids))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"[ERROR]: {e}")
    except Exception as e:
        print(f"[ERROR]: {e}")
