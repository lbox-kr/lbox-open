# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC 4.0


def gen_input_template_str(task, key, type):
    if key == "fine_imprisonment_lvs":
        if type == 0:
            input_template_str = (
                '{"placeholder":"text_a"} 형사사건에 대하여 순서대로 벌금, 징역, 금고 레벨을 쓰시오. {"mask"}'
            )
        else:
            raise NotImplementedError
    elif key == "claim_acceptance_lv":
        if type == 0:
            input_template_str = (
                '{"placeholder":"text_a"} 주어진 사실관계, 청구 취지를 읽고, 주장 인정율을 예측하시오. {"mask"}'
            )
        else:
            raise NotImplementedError
    elif key == "casename_classification":
        if type == 0:
            input_template_str = (
                '{"placeholder":"text_a"} 주어진 사실관계를 읽고, 사건명을 예측하시오. {"mask"}'
            )
        else:
            raise NotImplementedError
    elif key == "statute_classification":
        if type == 0:
            input_template_str = (
                '{"placeholder":"text_a"} 주어진 사실관계를 읽고, 적용될 형법 조항들을 예측하시오. {"mask"}'
            )
        else:
            raise NotImplementedError
    elif key == "summarization":
        if type == 0:
            input_template_str = '{"placeholder":"text_a"}\n요약하시오.\n{"mask"}'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return input_template_str
