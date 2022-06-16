# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC-ND 4.0

import re


def sep_token_parser_baseline(
    parse_sep_token, value_sep_token, empty_token, sub_parse_keys, text
):
    parse = {}
    char_comma = ","

    # filter ',' inside of number
    ms_money = re.finditer("\d[\d|,|.]+\d", text)
    ms_comma = re.finditer(char_comma, text)

    idxs_comma = [m.start() for m in ms_comma]
    idxs_comma_I = []
    for ms_money in ms_money:
        st = ms_money.start()
        ed = ms_money.end()
        for idx_comma in idxs_comma:
            if idx_comma >= st and idx_comma <= ed:
                idxs_comma_I.append(idx_comma)

    text_copy = ""
    rpl_sym = "★"
    for i, c in enumerate(text):
        if i in idxs_comma_I:
            text_copy += rpl_sym
        else:
            text_copy += c

    values = text_copy.split(parse_sep_token)

    for i, k in enumerate(sub_parse_keys):
        if i <= len(values) - 1:
            if empty_token in values[i]:
                vals = empty_token
            else:
                parse_values_before_split = values[i]
                parse_values = parse_values_before_split.split(value_sep_token)
                vals = []
                for val in parse_values:
                    v = val.replace(rpl_sym, char_comma).strip()
                    v = re.sub("\s", "", v)
                    vals.append(v)
        else:
            vals = None
        parse[k] = vals
    return parse


def sep_token_based_parser(
    target_parse, parse_sep_token, value_sep_token, empty_token, keys, text
):
    if target_parse in [
        "fine_imprisonment_lvs",
        "claim_acceptance_lv",
        "casename_classification",
        "statute_classification",
    ]:
        # print(text)
        parse = sep_token_parser_baseline(
            parse_sep_token, value_sep_token, empty_token, keys, text
        )
    else:
        raise NotImplementedError

    return parse
