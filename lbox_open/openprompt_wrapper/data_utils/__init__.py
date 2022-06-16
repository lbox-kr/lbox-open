import json

from openprompt import data_utils


class InputExampleWrapper(data_utils.InputExample):
    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        # return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
