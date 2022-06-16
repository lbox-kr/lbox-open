# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC-ND 4.0

import json

import datasets

_CASENAME_CLASSIFICATION_FEATURES = {
    "id": datasets.Value("int64"),
    "casetype": datasets.Value("string"),
    "casename": datasets.Value("string"),
    "facts": datasets.Value("string"),
}

_STATUTE_CLASSIFICATION_FEATURES = {
    "id": datasets.Value("int64"),
    "casetype": datasets.Value("string"),
    "casename": datasets.Value("string"),
    "statutes": datasets.features.Sequence(datasets.Value("string")),
    "facts": datasets.Value("string"),
}

_LJP_CRIMINAL = {
    "id": datasets.Value("int64"),
    "casetype": datasets.Value("string"),
    "casename": datasets.Value("string"),
    "facts": datasets.Value("string"),
    "reason": datasets.Value("string"),
    "label": {
        "text": datasets.Value("string"),
        "fine_lv": datasets.Value("int64"),
        "imprisonment_with_labor_lv": datasets.Value("int64"),
        "imprisonment_without_labor_lv": datasets.Value("int64"),
    },
    "ruling": {
        "text": datasets.Value("string"),
        "parse": {
            "fine": {
                "type": datasets.Value("string"),
                "unit": datasets.Value("string"),
                "value": datasets.Value("int64"),
            },
            "imprisonment": {
                "type": datasets.Value("string"),
                "unit": datasets.Value("string"),
                "value": datasets.Value("int64"),
            },
        },
    },
}

_LJP_CIVIL = {
    "id": datasets.Value("int64"),
    "casetype": datasets.Value("string"),
    "casename": datasets.Value("string"),
    "facts": datasets.Value("string"),
    "claim_acceptance_lv": datasets.Value("int64"),
    "gist_of_claim": {
        "text": datasets.Value("string"),
        "money": {
            "provider": datasets.Value("string"),
            "taker": datasets.Value("string"),
            "unit": datasets.Value("string"),
            "value": datasets.Value("int64"),
        },
    },
    "ruling": {
        "text": datasets.Value("string"),
        "money": {
            "provider": datasets.Value("string"),
            "taker": datasets.Value("string"),
            "unit": datasets.Value("string"),
            "value": datasets.Value("int64"),
        },
        "litigation_cost": datasets.Value("float32"),
    },
}

_SUMMARIZATION_CLASSIFICATION_FEATURES = {
    "id": datasets.Value("int64"),
    "summary": datasets.Value("string"),
    "precedent": datasets.Value("string"),
}

_PRECEDENT_CORPUS_FEATURES = {
    "id": datasets.Value("int64"),
    "precedent": datasets.Value("string"),
}


class LBoxOpenConfig(datasets.BuilderConfig):
    """BuilderConfig for OpenLBox."""

    def __init__(
        self,
        features,
        data_url,
        citation,
        url,
        label_classes=("False", "True"),
        **kwargs,
    ):
        # Version history:
        # 0.1.0: Initial version.
        super(LBoxOpenConfig, self).__init__(
            version=datasets.Version("0.2.0"), **kwargs
        )
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class LBoxOpen(datasets.GeneratorBasedBuilder):
    """The Legal AI Benchmark dataset from Korean Legal Cases."""

    BUILDER_CONFIGS = [
        LBoxOpenConfig(
            name="casename_classification",
            description="",
            features=_CASENAME_CLASSIFICATION_FEATURES,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/casename_classification/v0.1.2/",
            # data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/casename_classification/v0.1.0/",
            citation="",
            url="lbox.kr",
        ),
        LBoxOpenConfig(
            name="statute_classification",
            description="",
            features=_STATUTE_CLASSIFICATION_FEATURES,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/statute_classification/v0.1.2/",
            # data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/statute_classification/v0.1.0/",
            citation="",
            url="lbox.kr",
        ),
        LBoxOpenConfig(
            name="ljp_criminal",
            description="",
            features=_LJP_CRIMINAL,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/judgement_prediction/v0.1.2/criminal/",
            citation="",
            url="lbox.kr",
        ),
        LBoxOpenConfig(
            name="ljp_civil",
            description="",
            features=_LJP_CIVIL,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/judgement_prediction/v0.1.2/civil/",
            citation="",
            url="lbox.kr",
        ),
        LBoxOpenConfig(
            name="summarization",
            description="",
            features=_SUMMARIZATION_CLASSIFICATION_FEATURES,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/summarization/v0.1.0/",
            citation="",
            url="lbox.kr",
        ),
        LBoxOpenConfig(
            name="precedent_corpus",
            description="",
            features=_PRECEDENT_CORPUS_FEATURES,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/case_corpus/v0.1.0/",
            citation="",
            url="lbox.kr",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(self.config.features),
            homepage=self.config.url,
            citation="",
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "precedent_corpus":
            dl_dir = {
                "train": dl_manager.download_and_extract(
                    f"{self.config.data_url}case_corpus-150k.jsonl"
                )
                or "",
            }

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": dl_dir["train"],
                        "split": datasets.Split.TRAIN,
                    },
                )
            ]

        elif self.config.name in [
            "casename_classification",
            "statute_classification",
            "ljp_criminal",
            "ljp_civil",
        ]:
            dl_dir = {
                "train": dl_manager.download_and_extract(
                    f"{self.config.data_url}train.jsonl"
                )
                or "",
                "valid": dl_manager.download_and_extract(
                    f"{self.config.data_url}valid.jsonl"
                )
                or "",
                "test": dl_manager.download_and_extract(
                    f"{self.config.data_url}test.jsonl"
                )
                or "",
                "test2": dl_manager.download_and_extract(
                    f"{self.config.data_url}test2.jsonl"
                )
                or "",
            }

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": dl_dir["train"],
                        "split": datasets.Split.TRAIN,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": dl_dir["valid"],
                        "split": datasets.Split.VALIDATION,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": dl_dir["test"],
                        "split": datasets.Split.TEST,
                    },
                ),
                datasets.SplitGenerator(
                    name="test2",
                    gen_kwargs={
                        "data_file": dl_dir["test2"],
                        "split": "test2",
                    },
                ),
            ]
        else:
            dl_dir = {
                "train": dl_manager.download_and_extract(
                    f"{self.config.data_url}train.jsonl"
                )
                or "",
                "valid": dl_manager.download_and_extract(
                    f"{self.config.data_url}valid.jsonl"
                )
                or "",
                "test": dl_manager.download_and_extract(
                    f"{self.config.data_url}test.jsonl"
                )
                or "",
            }

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": dl_dir["train"],
                        "split": datasets.Split.TRAIN,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": dl_dir["valid"],
                        "split": datasets.Split.VALIDATION,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": dl_dir["test"],
                        "split": datasets.Split.TEST,
                    },
                ),
            ]

    def _generate_examples(self, data_file, split):
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                yield row["id"], row
