# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC-ND 4.0

from collections import defaultdict


class ExactMatch:
    def __init__(self, parse_keys, empty_value):

        if "doc_id" in parse_keys:
            parse_keys.remove("doc_id")

        self.parse_keys = parse_keys

        self.empty_value = empty_value

    def is_empty(self, value):
        return (str(value) == str(self.empty_value)) or (value is None)

    def compare_parse(self, gt_parse, pr_parse):
        cnt_tp = defaultdict(int)  # both exsit and pr is correct
        cnt_fn = defaultdict(int)  # gt exists but pr is empty
        cnt_fp = defaultdict(
            int
        )  # [gt empty but pr exists] or [gt exists yet pr is wrong]
        cnt_tn = defaultdict(int)  # gt & pr both empty

        for key in self.parse_keys:
            gt_val = gt_parse[key]
            pr_val = pr_parse[key]

            if self.is_empty(gt_val):
                if self.is_empty(pr_val):
                    cnt_tn[key] += 1
                else:
                    cnt_fp[key] += 1
            else:
                if self.is_empty(pr_val):
                    cnt_fn[key] += 1
                else:
                    if str(gt_val) == str(pr_val):
                        cnt_tp[key] += 1
                    else:
                        cnt_fp[key] += 1

        return (cnt_tp, cnt_fp, cnt_fn, cnt_tn)

    def imp_fill_cnt(self, cnt_all, cnt):
        for key in self.parse_keys:
            cnt_all[key] += cnt[key]

    def calculate_micro_f1(self, cnt_tp_all, cnt_fp_all, cnt_fn_all):
        f1_all = {}
        for key in self.parse_keys:
            tp = cnt_tp_all[key]
            fp = cnt_fp_all[key]
            fn = cnt_fn_all[key]

            p = tp / (tp + fp + 1e-5)
            r = tp / (tp + fn + 1e-5)
            f1 = 2 * p * r / (p + r + 1e-5)

            f1_all[key] = f1

        return f1_all

    def compare_parses(self, gt_parses, pr_parses, confidences=None, threshold=0.0):
        cnt_tp_all = defaultdict(int)
        cnt_fn_all = defaultdict(int)
        cnt_fp_all = defaultdict(int)
        cnt_tn_all = defaultdict(int)
        if confidences is None:
            confidences = [1.0] * len(gt_parses)
            assert threshold == 0.0
        cnt = 0
        for gt_parse, pr_parse, confidence in zip(gt_parses, pr_parses, confidences):
            if confidence < threshold:
                continue
            cnt += 1
            (cnt_tp, cnt_fp, cnt_fn, cnt_tn) = self.compare_parse(gt_parse, pr_parse)

            self.imp_fill_cnt(cnt_tp_all, cnt_tp)
            self.imp_fill_cnt(cnt_fp_all, cnt_fp)
            self.imp_fill_cnt(cnt_fn_all, cnt_fn)
            self.imp_fill_cnt(cnt_tn_all, cnt_tn)

        f1_all = self.calculate_micro_f1(cnt_tp_all, cnt_fp_all, cnt_fn_all)
        th_recall = cnt / len(confidences)

        return (
            f1_all,
            cnt_tp_all,
            cnt_fp_all,
            cnt_fn_all,
            cnt_tn_all,
            th_recall,
        )
