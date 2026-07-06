from typing import List
import json


class Parsing:
    @staticmethod
    def load_json_data(path_file: str) -> List:
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
