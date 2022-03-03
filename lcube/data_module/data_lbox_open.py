from typing import List

import torch
from datasets import load_dataset
import pytorch_lightning as pl


class LBoxOpenData:
    def __init__(self, task, tokenizer, max_input_len, max_target_len, features):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.input_key = get_input_key(task)
        self.label_key = get_label_key(task)

        self.features = []
        for feature in features:
            if self.label_key == "statutes":
                statutes_text = ", ".join(feature["statutes"])  # List[str] -> str
                feature["statutes"] = statutes_text
            else:
                pass
            self.features.append(feature)

        # self.padding = "max_length"
        self.padding = True
        self.truncation = True


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        input = feature[self.input_key]
        if self.label_key in feature:
            label = feature[self.label_key]
        else:
            label = None

        model_inputs = self.tokenizer(
            input,
            max_length=self.max_input_len,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors='pt',
        )

        if label is not None:

            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer(
                    label,
                    max_length=self.max_target_len,
                    padding=self.padding,
                    truncation=self.truncation,
                    return_tensors='pt',
                )

            model_inputs["labels"] = label["input_ids"]

        model_inputs.update(
            {
                "id": feature["id"],
                "input_text": feature[self.input_key],
                "label_text": feature[self.label_key],
            }
        )
        return model_inputs

    def collate_fn(self, batch):
        # batch = List[Dict]

        keys = list(batch[0].keys())
        batch_model_inputs = {}
        batch_others = {}

        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                batch_model_inputs[k] = torch.nn.utils.rnn.pad_sequence(
                    [feature[k][0] for feature in batch],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
            else:
                batch_others[k] = [feature[k] for feature in batch]

        return batch_model_inputs, batch_others


def get_input_key(task):
    if task == "casename_classification":
        key = "facts"
    elif task == "statute_classification":
        key = "facts"
    elif task == "summarization":
        key = "precedent"
    else:
        raise ValueError

    return key


def get_label_key(task):
    if task == "casename_classification":
        key = "casename"
    elif task == "statute_classification":
        key = "statutes"
    elif task == "summarization":
        key = "summary"
    else:
        raise ValueError

    return key


class LBoxOpenDataModule(pl.LightningDataModule):
    def __init__(self, dataset_card, task, tokenizer, max_input_len, max_target_len, batch_size, batch_size_eval):
        super().__init__()
        assert dataset_card == "lbox/lbox_open"
        self.dataset_card = dataset_card
        self.task = task
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval

    def setup(self, stage=None):
        self.dataset = load_dataset(self.dataset_card, self.task)

    @staticmethod
    def _get_loader(data, batch_size, shuffle):
        return torch.utils.data.DataLoader(
            data,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=data.collate_fn,
        )

    def train_dataloader(self):
        data = LBoxOpenData(
            self.task, self.tokenizer, self.max_input_len, self.max_target_len, self.dataset["train"]
        )
        return self._get_loader(data, self.batch_size, shuffle=True)

    def val_dataloader(self):
        data = LBoxOpenData(
            self.task, self.tokenizer, self.max_input_len, self.max_target_len, self.dataset["validation"]
        )
        return self._get_loader(data, self.batch_size_eval, shuffle=False)

    def test_dataloader(self):
        data = LBoxOpenData(self.task, self.tokenizer, self.max_input_len, self.max_target_len, self.dataset["test"])
        return self._get_loader(data, self.batch_size_eval, shuffle=False)
