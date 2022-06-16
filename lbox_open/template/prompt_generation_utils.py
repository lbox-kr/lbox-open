# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC-ND 4.0

from openprompt import prompts

from lbox_open.template import prompt_templates

from ..constants import ENG_TO_KOR_PARSE_NAMES_LJP_CRIMINAL


def gen_template(task, key, type, plm, tokenizer):
    mytemplate = prompts.MixedTemplate(
        model=plm,
        tokenizer=tokenizer,
        text=prompt_templates.gen_input_template_str(task, key, type),
    )

    return mytemplate


def gen_output_template(
    task,
    key,
    sub_keys,
    label,
    parse_sep_token,
):
    """ """
    # todo: move template part to ./template.py

    if task == "ljp_criminal":
        if key == "fine_imprisonment_lvs":
            label_dict = label
            out = ""
            for key in sub_keys:
                key_kor = ENG_TO_KOR_PARSE_NAMES_LJP_CRIMINAL[key]
                out += f"{key_kor}{label_dict[key]}{parse_sep_token} "
            out = out.strip(f"{parse_sep_token} ")

        else:
            raise NotImplementedError

    elif task == "ljp_civil":
        if key == "claim_acceptance_lv":
            out = str(label)
        else:
            raise NotImplementedError

    elif task == "casename_classification":
        if key == "casename_classification":
            out = str(label)
        else:
            raise NotImplementedError
    elif task == "statute_classification":
        assert isinstance(label, list)
        if key == "statute_classification":
            out = f"{parse_sep_token} ".join(label)
        else:
            raise NotImplementedError
    elif task == "summarization":
        if key == "summarization":
            out = str(label)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return out
