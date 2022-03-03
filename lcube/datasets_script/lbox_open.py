# lbox_open
# Copyright 2022-present LBox Co. Ltd.
# Licensed under the CC BY-NC-ND 4.0
import json
import datasets

_CASENAME_CLASSIFICATION_FEATURES = {
    "id": datasets.Value("int32"),
    "casetype": datasets.Value("string"),
    "casename": datasets.Value("string"),
    "facts": datasets.Value("string"),
}

_STATUTE_CLASSIFICATION_FEATURES = {
    "id": datasets.Value("int32"),
    "casetype": datasets.Value("string"),
    "casename": datasets.Value("string"),
    "statutes": datasets.features.Sequence(datasets.Value("string")),
    "facts": datasets.Value("string"),
}

_SUMMARIZATION_CLASSIFICATION_FEATURES = {
    "id": datasets.Value("int32"),
    "summary": datasets.Value("string"),
    "precedent": datasets.Value("string"),
}

_CASE_CORPUS_FEATURES = {
    "id": datasets.Value("int32"),
    "precedent": datasets.Value("string"),
}


class LBoxOpenConfig(datasets.BuilderConfig):
    """BuilderConfig for OpenLBox."""

    def __init__(self, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        # Version history:
        # 0.1.0: Initial version.
        super(LBoxOpenConfig, self).__init__(version=datasets.Version("0.1.0"), **kwargs)
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
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/casename_classification/v0.1.0/",
            citation="",
            url="lbox.kr",
        ),
        LBoxOpenConfig(
            name="statute_classification",
            description="",
            features=_STATUTE_CLASSIFICATION_FEATURES,
            data_url="https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/statute_classification/v0.1.0/",
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
            name="case_corpus",
            description="",
            features=_CASE_CORPUS_FEATURES,
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
        if self.config.name == "case_corpus":
            dl_dir = {
                "train": dl_manager.download_and_extract(f"{self.config.data_url}case_corpus-150k.jsonl") or "",
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

        else:
            dl_dir = {
                "train": dl_manager.download_and_extract(f"{self.config.data_url}train.jsonl") or "",
                "valid": dl_manager.download_and_extract(f"{self.config.data_url}valid.jsonl") or "",
                "test": dl_manager.download_and_extract(f"{self.config.data_url}test.jsonl") or "",
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
