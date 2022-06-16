import json
from pathlib import Path

from openprompt import plms
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    MT5Config,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .lm import LMTFastokenizerWrapperCustom


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]


MT5TokenizerWrapper = plms.T5TokenizerWrapper

_MODEL_CLASSES = {
    "mt5": plms.ModelClass(
        **{
            "config": MT5Config,
            "tokenizer": MT5Tokenizer,
            "model": MT5ForConditionalGeneration,
            "wrapper": MT5TokenizerWrapper,
        }
    ),
    "kogpt2": plms.ModelClass(
        **{
            "config": GPT2Config,
            "tokenizer": PreTrainedTokenizerFast,
            "model": GPT2LMHeadModel,
            "wrapper": LMTFastokenizerWrapperCustom,
        }
    ),
    "legal-gpt": plms.ModelClass(
        **{
            "config": GPT2Config,
            "tokenizer": AutoTokenizer,
            "model": GPT2LMHeadModel,
            "wrapper": LMTFastokenizerWrapperCustom,
        }
    ),
}


def load_plm_wrapper(
    model_name,
    model_path,
    specials_to_add=None,
    revision=None,
    do_not_load_pretrained_weight=False,
    use_custom_loader=False,
):
    if not use_custom_loader:
        return plms.load_plm(model_name, model_path, specials_to_add)
    else:
        model_class = get_model_class(plm_type=model_name)
        wrapper = model_class.wrapper
        if model_name in ["kogpt2"]:
            model_config = model_class.config.from_pretrained(
                model_path, revision=revision
            )
            if do_not_load_pretrained_weight:
                model = model_class.model(
                    config=model_config,
                )
            else:
                model = model_class.model.from_pretrained(
                    model_path, revision=revision, config=model_config
                )

            tokenizer = model_class.tokenizer.from_pretrained(
                model_path,
                bos_token="</s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
            )
        elif model_name in ["legal-gpt"]:
            model_config = model_class.config.from_pretrained(
                model_path, revision=revision
            )
            if do_not_load_pretrained_weight:
                model = model_class.model(
                    config=model_config,
                )
            else:
                model = model_class.model.from_pretrained(
                    model_path, revision=revision, config=model_config
                )
            tokenizer = model_class.tokenizer.from_pretrained(
                model_path,
                bos_token="[BOS]",
                unk_token="[UNK]",
                pad_token="[PAD]",
                mask_token="[MASK]",
            )

        else:
            model_config = model_class.config.from_pretrained(
                model_path, revision=revision
            )
            if do_not_load_pretrained_weight:
                model = model_class.model(
                    config=model_config,
                )
            else:

                model = model_class.model.from_pretrained(
                    model_path, revision=revision, config=model_config
                )

            if "gpt" in model_name:  # add pad token for gpt
                specials_to_add = ["<pad>"]

            tokenizer = model_class.tokenizer.from_pretrained(model_path)
            model, tokenizer = plms.add_special_tokens(
                model, tokenizer, specials_to_add=specials_to_add
            )

        if model_name in ["mt5"]:
            _path = (
                Path(__file__).parent.resolve() / "mt5_additional_special_tokens.json"
            )
            with open(_path) as f:
                mt5_additional_special_tokens = json.load(f)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": mt5_additional_special_tokens[
                        "additional_special_tokens"
                    ]
                }
            )

        return model, tokenizer, model_config, wrapper
