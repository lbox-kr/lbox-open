# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC-ND 4.0

import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from pprint import pprint


import datasets
import pytorch_lightning as pl
import torch
from openprompt.utils.metrics import generation_metric
from transformers.generation_utils import GenerationMixin
from rouge_score import rouge_scorer
import numpy as np

import lbox_open.utils.general_utils as gu
from lbox_open import openprompt_wrapper
from lbox_open.model.model_optimizer import get_lr_dict, get_optimizer
from lbox_open.parser.output_parser_utils import (
    cal_em_from_parses,
    get_parses_from_eval_results,
)
from lbox_open.metric import rouge_metric_utils


class GenerativeParser(pl.LightningModule, GenerationMixin):
    def __init__(self, cfg, plm, plm_tokenizer, input_templates):
        super().__init__()
        self.task = cfg.model.task
        self.mparam = cfg.model
        self.tparam = cfg.train
        self.iparam = cfg.infer
        self.cfg_name = cfg.name
        self.target_parses_dict = cfg.model.target_parses_dict

        self.prompt_models = {}
        self.plm = plm
        for target_parse, target_sub_parses in cfg.model.target_parses_dict.items():
            # keep them for just in case we tune plm
            prompt_model = openprompt_wrapper.PromptForGenerationCustom(
                plm=plm,
                template=input_templates[target_parse],
                freeze_plm=cfg.model.plm.freeze,
                tokenizer=plm_tokenizer,
                plm_eval_mode=cfg.model.plm.eval_mode,
            )

            self.prompt_models[target_parse] = prompt_model

        self.prompt_models = torch.nn.ModuleDict(self.prompt_models)

        # if self.plm.config.is_encoder_decoder:
        self.generation_arguments = {
            "max_length": cfg.infer.max_length,
            "max_new_tokens": cfg.infer.get("max_new_tokens", None),
            "min_length": cfg.infer.min_length,
            "temperature": cfg.infer.temperature,
            "do_sample": cfg.infer.do_sample,
            "top_k": cfg.infer.top_k,
            "top_p": cfg.infer.top_p,
            "repetition_penalty": cfg.infer.repetition_penalty,
            "num_beams": cfg.infer.num_beams,
            "bad_words_ids": cfg.infer.bad_words_ids,
            "use_cache": True,
        }

        if plm.config.is_encoder_decoder:
            # remove max_new_tokens
            print(f"The model is of is_encoder_decoder. Thus we remove max new tokens.")
            self.generation_arguments.pop("max_new_tokens")
        else:
            if cfg.infer.get("max_new_tokens", None):
                print(
                    f"Max length in generation option shall be ignored as max_new_tokens presents."
                )
                self.generation_arguments["max_length"] = None

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], tokenizer=rouge_metric_utils.WhiteSpaceTokenizer()
        )
    def forward(self, target_parse, batch):
        loss = self.prompt_models[target_parse](batch[target_parse])
        return loss

    def training_step(self, batch, batch_idx):
        n_keys = len(self.target_parses_dict)
        loss = 0
        for i_target, (target_parse, _) in enumerate(self.target_parses_dict.items()):
            loss += self.forward(target_parse, batch)
        return {"loss": loss / n_keys}

    def training_epoch_end(self, outputs):

        loss_all = torch.stack(self.gather_loss(outputs))
        ave_loss = torch.mean(loss_all)
        self.log("training__ave_loss", ave_loss)

    def gather_loss(self, outputs):
        loss_all = []
        for output in outputs:
            loss_all.append(output["loss"])

        return loss_all

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        (
            eval_score,
            doc_ids_all,
            pr_texts_all,
            gt_texts_all,
            confidences_all,
        ) = self._eval_epoch_end(outputs)
        print("\nValidation!-----------------------------------------")
        pprint(eval_score)
        pprint(f"GT: {gt_texts_all[self.tparam.validation_target_parse][0:2]}")
        pprint(f"PR: {pr_texts_all[self.tparam.validation_target_parse][0:2]}")

        if self.tparam.validation_metric in ["sentence_bleu"]:
            validation_score = eval_score[self.tparam.validation_target_parse]

        elif self.tparam.validation_metric in ["rougeL"]:
            validation_score = eval_score[self.tparam.validation_target_parse]

        elif self.tparam.validation_metric in ["em"]:
            if self.tparam.validation_sub_param.method == "single_parse":
                sub_parse_name = self.tparam.validation_sub_param.target_sub_parse
                validation_score = eval_score[self.tparam.validation_target_parse][
                    "f1"
                ][sub_parse_name]
            elif self.tparam.validation_sub_param.method == "average":
                validation_score = 0
                cnt = 0
                for sub_parse_name, score in eval_score[
                    self.tparam.validation_target_parse
                ]["f1"].items():
                    validation_score += score
                    cnt += 1
                validation_score /= cnt
            elif self.tparam.validation_sub_param.method == "text_em":
                validation_score = eval_score[self.tparam.validation_target_parse][
                    "text_em"
                ]
            else:
                raise ValueError
            for sub_parse_name, score in eval_score[
                self.tparam.validation_target_parse
            ]["f1"].items():
                self.log(sub_parse_name, score)
            self.log(
                f"{self.tparam.validation_target_parse}_text_em",
                eval_score[self.tparam.validation_target_parse]["text_em"],
            )
        else:
            raise ValueError

        self.log(
            f"{self.tparam.validation_metric}_{self.tparam.validation_sub_param.method}",
            validation_score,
        )

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        output_save_dir = (
            Path(self.tparam.weight.path).parent / "analysis" / self.cfg_name
        )
        os.makedirs(output_save_dir, exist_ok=True)
        (
            eval_score,
            doc_ids_all,
            pr_texts_all,
            gt_texts_all,
            confidences_all,
        ) = self._eval_epoch_end(
            outputs, save=True, output_save_dir=output_save_dir, verbose=True
        )
        print("Test!-----------------------------------------------")
        print(eval_score)

        output_save_path_eval_score = output_save_dir / "eval_score.json"
        gu.save_json(output_save_path_eval_score, eval_score)

        eval_result = {
            "doc_ids": doc_ids_all,
            "pr_texts": pr_texts_all,
            "gt_texts": gt_texts_all,
        }
        output_save_path_eval_result = output_save_dir / "eval_result.json"
        gu.save_json(output_save_path_eval_result, eval_result)

        output_save_path_confidences = output_save_dir / "confidences.json"
        gu.save_json(output_save_path_confidences, confidences_all)

        # add doc_ids to confidences_all
        confidences_all_with_doc_ids = {}
        for key_target_parse, confidences in confidences_all.items():
            c_with_ids = [
                (doc_id, c)
                for doc_id, c in zip_longest(doc_ids_all[key_target_parse], confidences)
            ]
            confidences_all_with_doc_ids[key_target_parse] = c_with_ids

        output_save_path_confidences_with_doc_ids = (
            output_save_dir / "confidences_with_doc_ids.json"
        )
        gu.save_json(
            output_save_path_confidences_with_doc_ids, confidences_all_with_doc_ids
        )

    def _eval_step(self, batch, batch_idx):

        out = defaultdict(dict)
        for target_parse, _ in self.target_parses_dict.items():
            _prs, _gts, confidences = self.evaluate(target_parse, batch)

            # add confidences as a saved output.
            out[target_parse]["pr_texts"] = _prs
            out[target_parse]["gt_texts"] = _gts
            out[target_parse]["doc_ids"] = batch[target_parse]["guid"]
            out[target_parse]["confidences"] = confidences

        return out

    def _eval_epoch_end(self, outputs, save=False, output_save_dir=None, verbose=False):
        # outputs = [list of each step outputs]
        pr_texts_all = self.gather_step_outputs("pr_texts", outputs)
        gt_texts_all = self.gather_step_outputs("gt_texts", outputs)
        doc_ids_all = self.gather_step_outputs("doc_ids", outputs)
        confidences_all = self.gather_step_outputs("confidences", outputs)

        eval_score = self.cal_score(
            doc_ids_all,
            pr_texts_all,
            gt_texts_all,
            save=save,
            output_save_dir=output_save_dir,
            confidences=confidences_all,
            threshold=0.0,
            verbose=False,
        )

        return eval_score, doc_ids_all, pr_texts_all, gt_texts_all, confidences_all

    def cal_score(
        self,
        doc_ids_all,
        pr_texts_all,
        gt_texts_all,
        save=False,
        output_save_dir=None,
        confidences=None,
        threshold=0.0,
        verbose=False,
        input_texts=None,
    ):

        if self.tparam.validation_metric == "sentence_bleu":
            eval_score = {}
            for target_parse, _ in self.target_parses_dict.items():
                groundtruth_sentence = gt_texts_all[target_parse]
                generated_sentence = pr_texts_all[target_parse]
                eval_score[target_parse] = generation_metric(
                    generated_sentence, groundtruth_sentence, "sentence_bleu"
                )
        elif self.tparam.validation_metric == "rougeL":
            eval_score = {}
            for target_parse, _ in self.target_parses_dict.items():
                pr_texts = pr_texts_all[target_parse]
                gt_texts = gt_texts_all[target_parse]
                target_scores = []
                for pr_text, gt_text in zip_longest(pr_texts, gt_texts):
                    r_score = self.rouge_scorer.score(
                        prediction=pr_text, target=gt_text
                    )

                    target_scores.append(
                        r_score[self.tparam.validation_metric].fmeasure
                    )

                eval_score[target_parse] = np.mean(
                    target_scores
                )
                print(eval_score)

        elif self.tparam.validation_metric == "em":
            # EM score
            parses = get_parses_from_eval_results(
                self.iparam,
                self.target_parses_dict,
                doc_ids_all,
                gt_texts_all,
                pr_texts_all,
            )

            # analysis
            eval_score = cal_em_from_parses(
                self.iparam,
                self.target_parses_dict,
                parses,
                verbose=verbose,
                save=save,
                output_save_dir=output_save_dir,
                input_texts=input_texts,
                confidences=confidences,
                threshold=threshold,
            )

            # text exact matching
            for target_parse, target_sub_parses in self.target_parses_dict.items():
                gt_texts = gt_texts_all[target_parse]
                pr_texts = pr_texts_all[target_parse]
                corrects = [str(x) == str(y) for x, y in zip(gt_texts, pr_texts)]
                text_em_score = sum(corrects) / len(corrects)
                eval_score[target_parse]["text_em"] = text_em_score

        else:
            raise ValueError
        return eval_score

    def gather_step_outputs(self, key, outputs):
        outputs_all = defaultdict(list)

        for target_parse, _ in self.target_parses_dict.items():
            for output in outputs:
                outputs_all[target_parse] += output[target_parse][key]

        return outputs_all

    def configure_optimizers(self):
        optimizer = get_optimizer(self.mparam, self.tparam, self)
        lr_dict = get_lr_dict(optimizer, self.tparam, "prompt")

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def evaluate(self, target_parse, batch):
        generated_sentence = []
        groundtruth_sentence = []

        seqs, output_sentence, confidences = self.prompt_models[target_parse].generate(
            batch[target_parse], **self.generation_arguments
        )
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(batch[target_parse]["tgt_text"])

        return generated_sentence, groundtruth_sentence, confidences
