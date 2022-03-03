from functools import reduce

import datasets
import pytorch_lightning as pl
import torch.optim


class SeqToSeqBaseline(pl.LightningModule):
    def __init__(self, task, backbone, tokenizer, learning_rate, max_target_len, validation_metric):
        super().__init__()
        self.task = task
        self.model = backbone
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.max_target_len = max_target_len
        self.validation_metric = validation_metric

    def forward(self, batch_model_inputs):
        return self.model(**batch_model_inputs)

    def training_step(self, batch, batch_idx):
        batch_model_inputs, _ = batch
        output = self.forward(batch_model_inputs)
        return {"loss": output["loss"]}

    def validation_step(self, batch, batch_idx):
        batch_model_inputs, batch_others = batch

        return {
            "gts": self.get_gts(batch_others),
            "prs": self.get_prs(batch_model_inputs),
        }

    def training_epoch_end(self, outputs):
        # outputs = List[Dict]
        ave_loss = torch.mean(
            torch.stack([x["loss"] for x in outputs])
        )
        self.log("training__ave_loss", ave_loss)

    def validation_epoch_end(self, outputs):
        f_concat_list = lambda x, y: x + y
        gts = reduce(f_concat_list, [x["gts"] for x in outputs], [])
        prs = reduce(f_concat_list, [x["prs"] for x in outputs], [])

        eval_score_dict = self.evaluate(gts, prs)
        self.log(self.validation_metric, eval_score_dict[self.validation_metric])
        print(f"Validation test")
        print(f"ground truth: {gts[0]}")
        print(f"prediction:   {prs[0]}")

    def configure_optimizers(self):
        param_list = [
            {
                "params": filter(
                    lambda p: p.requires_grad, self.model.parameters()
                ),
                "lr": self.lr,
            }
        ]
        optimizer = torch.optim.AdamW(param_list, lr=self.lr)
        return {"optimizer": optimizer}

    def generate_prediction(self, batch_model_inputs):
        pr_seqs = self.model.generate(batch_model_inputs["input_ids"], max_length=self.max_target_len)
        prs = self.tokenizer.batch_decode(pr_seqs, skip_special_tokens=True)
        return prs

    def get_gts(self, batch_others):
        gts = batch_others["label_text"]  # "casename"
        return gts

    def get_prs(self, batch_model_inputs):
        prs = self.generate_prediction(batch_model_inputs)
        return prs

    def evaluate(self, gts, prs):
        if self.validation_metric == "exact_match":
            corrects = [x == y for x, y in zip(gts, prs)]
            score = sum(corrects) / len(corrects)
        elif "rouge" in self.validation_metric:
            rouge_metric = datasets.load_metric('rouge')
            rouge_score = rouge_metric.compute(predictions=prs, references=gts)
            score = rouge_score[self.validation_metric].mid.fmeasure
        else:
            raise ValueError

        print(f"metric: {self.validation_metric}, score: {score}")

        return {
            self.validation_metric: score
        }
