# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC-ND 4.0

import torch
import transformers

map_optimizers_name_to_type = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def get_optimizer(mparam, tparam, model):
    # todo: plm training part
    _lr_type, lr_param = get_lr_type_and_param(tparam, "prompt")

    # prompt
    optimizer_type = map_optimizers_name_to_type[tparam.optim.prompt.optimizer_type]

    if model.task in [
        "ljp_civil",
        "ljp_criminal",
        "casename_classification",
        "statute_classification",
        "summarization",
    ]:
        optimizer_grouped_parameters = []
        if not mparam.plm.freeze:
            optimizer_grouped_parameters.append(
                {
                    "params": list(
                        filter(lambda p: p.requires_grad, model.plm.parameters())
                    ),
                    "lr": tparam.optim.plm.lr,
                }
            )

        for target_parse, _target_sub_parses in model.target_parses_dict.items():
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.prompt_models[
                            target_parse
                        ].template.named_parameters()
                        if "raw_embedding" not in n
                    ]
                }
            )

        optimizer = optimizer_type(
            optimizer_grouped_parameters, lr=tparam.optim.prompt.lr, weight_decay=0
        )

    else:
        raise NotImplementedError

    return optimizer


def get_lr_type_and_param(tparam, key):
    lr_type = tparam.optim[key].lr_scheduler_type
    lr_param = tparam.optim[key].lr_scheduler_param[lr_type]
    return lr_type, lr_param


def gen_lr_scheduler(tparam, optimizer, lr_type, lr_param):
    if lr_type == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda epoch: 1, lambda epoch: 1], verbose=True
        )
    elif lr_type == "multi_step_lr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_param["milestones"],
            gamma=lr_param["gamma"],
            verbose=True,
        )

    elif lr_type == "warmup_constant":
        lr_scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=lr_param.num_warmup_steps
        )
    elif lr_type == "cos_with_hard_restarts":
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=lr_param.num_warmup_steps,
            num_training_steps=lr_param.num_training_steps,
            num_cycles=lr_param.num_cycles,
        )
    elif lr_type == "linear":
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=lr_param.num_warmup_steps,
            num_training_steps=tparam.max_epochs,
        )

    else:
        raise NotImplementedError
    return lr_scheduler


def get_lr_dict(optimizer, tparam, key):
    lr_type, lr_param = get_lr_type_and_param(tparam, key)
    lr_scheduler = gen_lr_scheduler(tparam, optimizer, lr_type, lr_param)
    lr_dict = {
        "scheduler": lr_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "val_loss",
        "strict": True,
        "name": None,
    }

    return lr_dict
