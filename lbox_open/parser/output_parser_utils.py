# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC 4.0

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import lbox_open.utils.general_utils as gu
from lbox_open.metric.exact_match import ExactMatch

from .output_parser import sep_token_based_parser


def text_to_parse_separator_based(
    target_parse,
    parse_sep_token,
    value_sep_token,
    empty_token,
    target_sub_parses,
    texts,
):
    return list(
        map(
            lambda x: sep_token_based_parser(
                target_parse,
                parse_sep_token,
                value_sep_token,
                empty_token,
                target_sub_parses,
                x,
            ),
            texts,
        )
    )


def get_parses_from_eval_results(
    infer_param,
    target_parses_dict,
    doc_ids,
    gt_texts,
    pr_texts,
):
    parses = defaultdict(dict)
    for target_parse, target_sub_parses in target_parses_dict.items():
        gt_parses = text_to_parse_separator_based(
            target_parse,
            infer_param.parse_sep_token,
            infer_param.value_sep_token,
            infer_param.empty_token,
            target_sub_parses,
            gt_texts[target_parse],
        )

        pr_parses = text_to_parse_separator_based(
            target_parse,
            infer_param.parse_sep_token,
            infer_param.value_sep_token,
            infer_param.empty_token,
            target_sub_parses,
            pr_texts[target_parse],
        )

        # insert doc_ids
        for doc_id, gt_parse, pr_parse in zip_longest(
            doc_ids[target_parse], gt_parses, pr_parses
        ):
            gt_parse["doc_id"] = doc_id
            pr_parse["doc_id"] = doc_id

        parses[target_parse]["gt_parses"] = gt_parses
        parses[target_parse]["pr_parses"] = pr_parses

    return parses


def cal_em_from_parses(
    infer_param,
    target_parses_dict,
    parses,
    verbose=False,
    save=False,
    output_save_dir=None,
    confidences=None,
    threshold=0.0,
    input_texts=None,
):
    em_scores_full = {}
    for target_parse, target_sub_parses in target_parses_dict.items():

        gt_parses = parses[target_parse]["gt_parses"]
        pr_parses = parses[target_parse]["pr_parses"]

        if confidences is None:
            _confs = [1.0] * len(gt_parses)
        else:
            _confs = confidences[target_parse]

        exact_match = ExactMatch(
            list(gt_parses[0].keys()), empty_value=infer_param.empty_token
        )

        (
            f1_all,
            cnt_tp_all,
            cnt_fp_all,
            cnt_fn_all,
            cnt_tn_all,
            th_recall,
        ) = exact_match.compare_parses(gt_parses, pr_parses, _confs, threshold)

        if verbose:
            print(f"Target_parse: {target_parse} with th-recall: {th_recall}")
            print("tp-------------------")
            print(cnt_tp_all)
            print("fp-------------------")
            print(cnt_fp_all)
            print("fn-------------------")
            print(cnt_fn_all)
            print("tn-------------------")
            print(cnt_tn_all)
            print("f1-------------------")
            print(f1_all)

        score = {
            "f1": f1_all,
            "tp": cnt_tp_all,
            "fp": cnt_fp_all,
            "fn": cnt_fn_all,
            "tn": cnt_tn_all,
            "th_recall": th_recall,
        }
        em_scores_full[target_parse] = score

        if save:
            if output_save_dir is not None:
                if "path_eval_result" in infer_param:
                    print("path_eval_result is ignored!!!")
            else:
                output_save_dir = infer_param.path_eval_result

            # path_save_dir = os.path.dirname(output_save_dir)
            path_save_dir = output_save_dir
            path_save = Path(path_save_dir) / f"eval_parse_{target_parse}.json"
            gu.save_json(path_save, parses)

            path_save = Path(path_save_dir) / f"score_exact_match_{target_parse}.json"
            gu.save_json(path_save, score)

    return em_scores_full
