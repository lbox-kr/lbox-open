# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC 4.0

import datasets
import pytorch_lightning as pl
from openprompt import PromptDataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from lbox_open import openprompt_wrapper
from lbox_open.template import prompt_generation_utils


class PrecedentData(object):
    def __init__(self, cfg, mode, target_parse, target_sub_parses, raw_data):
        assert mode in ["train", "valid", "test", "predict"]
        self.cfg = cfg
        self.mode = mode
        self.target_parse = target_parse
        self.label_key = self._get_label_key(target_parse)
        self.target_sub_parses = target_sub_parses
        self.data_aug_param = cfg.train.get("data_aug_param", None)
        self.doc_id_key = self._get_doc_id(cfg.model.task)
        if raw_data is not None:
            self.features = self._gen_input_features(raw_data)

    def __getitem__(self, idx):
        return self.features[idx]

    def get_text_a(self, raw_data1):
        if isinstance(self.cfg.model.target_field, list):
            text_a = ""
            if self.cfg.model.task == "ljp_civil":
                for i, k in enumerate(self.cfg.model.target_field):
                    if k == "facts":
                        text_a += f"사실관계: {raw_data1[k]}\n"
                    elif k == "claim":
                        text_a += f"청구취지: {raw_data1[k]['text']}\n"
                    else:
                        raise NotImplementedError
                text_a = text_a.strip()
            else:
                for i, k in enumerate(self.cfg.model.target_field):
                    text_a += f"{raw_data1[k]}\n"
                text_a = text_a.strip()

        else:
            text_a = raw_data1[self.cfg.model.target_field]
        return text_a

    def _get_label_key(self, target_parse):
        if target_parse in ["claim_acceptance_lv"]:
            label_key = "claim_acceptance_lv"
        elif target_parse in ["casename_classification"]:
            label_key = "casename"
        elif target_parse in ["statute_classification"]:
            label_key = "statutes"
        elif target_parse in ["summarization"]:
            label_key = "summary"
        else:
            label_key = "label"

        return label_key

    def _gen_input_features(self, raw_data):
        features = []

        for i, raw_data1 in enumerate(raw_data):
            try:
                text_a = self.get_text_a(raw_data1)

                if self.label_key in raw_data1:
                    tgt_text = prompt_generation_utils.gen_output_template(
                        self.cfg.model.task,
                        self.target_parse,
                        self.target_sub_parses,
                        raw_data1[self.label_key],
                        self.cfg.infer.parse_sep_token,
                    )
                else:
                    assert self.mode == "predict"
                    tgt_text = "This is a dummy text."

                feature = openprompt_wrapper.InputExampleWrapper(
                    text_a=text_a,
                    text_b="",
                    tgt_text=tgt_text,
                    guid=str(raw_data1[self.doc_id_key]),
                )
            except Exception as e:
                print(f"doc_id: {self.doc_id_key}")
                print(repr(e))
                raise e
            features.append(feature)
        return features

    def __len__(self):
        if self.mode != "predict":
            return len(self.features)
        else:
            return 0

    def __iter__(self):
        self.features.__iter__()

    def _get_doc_id(self, task):

        if task in [
            "ljp_civil",
            "ljp_criminal",
            "casename_classification",
            "statute_classification",
            "summarization",
        ]:
            doc_id_key = "id"
        else:
            raise NotImplementedError
        return doc_id_key


class PrecedentDataModule(pl.LightningDataModule):
    def __init__(
        self, cfg, plm_tokenizer, TokenizerWrapper, input_templates, raw_data=None
    ):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.model.task
        self.raw_data = raw_data

        self.plm_tokenizer = plm_tokenizer
        self.TokenizerWrapperClass = TokenizerWrapper

        self.data_ts = {}
        self.data_vs = {}
        self.data_es = {}

        self.input_templates = input_templates
        self.target_parses_dict = cfg.model.target_parses_dict
        if len(self.target_parses_dict) > 1:
            raise Exception("Multitask learning is currently not supported!")

        self.use_local_data = cfg.data.use_local_data
        self.dataset_card = cfg.data.dataset_card

        self.training_set_name = cfg.data.training_set_name
        self.validation_set_name = cfg.data.validation_set_name
        self.test_set_name = cfg.data.test_set_name

    def setup(self, stage):
        if not self.use_local_data:
            assert self.raw_data is None
            self.raw_data = datasets.load_dataset(self.dataset_card, self.task)

        # Assign train/val datasets for use in dataloaders
        if stage in ["fit", "test"] or stage is None:
            for target_parse, target_sub_parses in self.target_parses_dict.items():
                self.data_ts[target_parse] = PrecedentData(
                    self.cfg,
                    "train",
                    target_parse,
                    target_sub_parses,
                    self.raw_data[self.training_set_name],
                ).features
                self.data_vs[target_parse] = PrecedentData(
                    self.cfg,
                    "valid",
                    target_parse,
                    target_sub_parses,
                    self.raw_data[self.validation_set_name],
                ).features
                if "test" in self.raw_data:
                    self.data_es[target_parse] = PrecedentData(
                        self.cfg,
                        "test",
                        target_parse,
                        target_sub_parses,
                        self.raw_data[self.test_set_name],
                    ).features

    def train_dataloader(self):
        data_loaders = {}
        for target_parse, target_sub_parses in self.target_parses_dict.items():
            data_loaders[target_parse] = PromptDataLoader(
                dataset=self.data_ts[target_parse],
                template=self.input_templates[target_parse],
                tokenizer=self.plm_tokenizer,
                tokenizer_wrapper_class=self.TokenizerWrapperClass,
                max_seq_length=self.cfg.model.max_seq_length,
                decoder_max_length=self.cfg.model.decoder_max_length,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                teacher_forcing=True,
                predict_eos_token=True,
                truncate_method="head",
            ).dataloader

        return data_loaders

    def val_dataloader(self):
        data_loaders = {}

        for target_parse, target_sub_parses in self.target_parses_dict.items():
            data_loaders[target_parse] = PromptDataLoader(
                dataset=self.data_vs[target_parse],
                template=self.input_templates[target_parse],
                tokenizer=self.plm_tokenizer,
                tokenizer_wrapper_class=self.TokenizerWrapperClass,
                max_seq_length=self.cfg.model.max_seq_length,
                decoder_max_length=self.cfg.model.decoder_max_length,
                batch_size=self.cfg.train.batch_size_prediction,
                shuffle=False,
                teacher_forcing=False,
                predict_eos_token=True,
                truncate_method="head",
            ).dataloader

        data_loaders = CombinedLoader(data_loaders)

        return data_loaders

    def test_dataloader(self):
        data_loaders = {}
        for target_parse, target_sub_parses in self.target_parses_dict.items():
            data_loaders[target_parse] = PromptDataLoader(
                dataset=self.data_es[target_parse],
                template=self.input_templates[target_parse],
                tokenizer=self.plm_tokenizer,
                tokenizer_wrapper_class=self.TokenizerWrapperClass,
                max_seq_length=self.cfg.model.max_seq_length,
                decoder_max_length=self.cfg.model.decoder_max_length,
                batch_size=self.cfg.train.batch_size_prediction,
                shuffle=False,
                teacher_forcing=False,
                predict_eos_token=True,
                truncate_method="head",
            ).dataloader

        data_loaders = CombinedLoader(data_loaders)

        return data_loaders
