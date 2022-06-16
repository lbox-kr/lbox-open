# Wonseok add FastTokenizerWrapper. The class inherit OpenPrompt-v1.0.0 TokenizerWrapper class.

import warnings

import numpy as np
from openprompt import plms


class FastTokenizerWrapper(plms.utils.TokenizerWrapper):
    def add_special_tokens(self, encoder_inputs):
        # add special tokens
        for key in encoder_inputs:
            if key == "input_ids":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encoder_inputs[
                        key
                    ] = self.tokenizer.build_inputs_with_special_tokens(
                        encoder_inputs[key]
                    )
            else:
                # special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(encoder_inputs[key], already_has_special_tokens=True))
                special_tokens_mask = np.array([0] * len(encoder_inputs[key]))
                with_special_tokens = np.array(
                    self.tokenizer.build_inputs_with_special_tokens(encoder_inputs[key])
                )
                if key in ["soft_token_ids"]:  # TODO maybe more than this
                    encoder_inputs[key] = (
                        (1 - special_tokens_mask) * with_special_tokens
                    ).tolist()  # use 0 as special
                else:
                    encoder_inputs[key] = (
                        (1 - special_tokens_mask) * with_special_tokens
                        - special_tokens_mask * 100
                    ).tolist()  # use -100 as special
        return encoder_inputs
