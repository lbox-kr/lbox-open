# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC 4.0

from pathlib import Path

import pytorch_lightning as pl
import torch

from lbox_open import openprompt_wrapper
from lbox_open.data_module.data_precedent import PrecedentDataModule
from lbox_open.model.generative_baseline_model import GenerativeParser
from lbox_open.template import prompt_generation_utils
from lbox_open.utils import general_utils as gu


def get_data_module(
    cfg,
    plm_tokenizer,
    TokenizerWrapper,
    input_templates,
):

    if cfg.data.use_local_data:
        raw_data = {
            "train": gu.load_jsonl(cfg.data.path_train, None),
            "valid": gu.load_jsonl(cfg.data.path_valid, None),
        }
        if cfg.data.path_test is not None:
            raw_data["test"] = gu.load_jsonl(cfg.data.path_test, None)
    else:
        raw_data = None

    if cfg.model.task in [
        "ljp_civil",
        "ljp_criminal",
        "casename_classification",
        "statute_classification",
        "summarization",
    ]:
        data_module = PrecedentDataModule(
            cfg,
            plm_tokenizer,
            TokenizerWrapper,
            input_templates,
            raw_data,
        )
    else:
        raise NotImplementedError

    return data_module


def get_plm(cfg):
    (
        plm,
        plm_tokenizer,
        plm_model_config,
        TokenizerWrapperClass,
    ) = openprompt_wrapper.load_plm_wrapper(
        model_name=cfg.model.plm.name,
        model_path=cfg.model.plm.path,
        revision=cfg.model.plm.revision,
        do_not_load_pretrained_weight=cfg.train.weight.do_not_load_pretrained_weight,
        use_custom_loader=True,
    )
    return plm, plm_tokenizer, plm_model_config, TokenizerWrapperClass


def gen_input_templates(cfg, plm, plm_tokenizer):
    input_templates = {}
    for target_parse, target_sub_parses in cfg.model.target_parses_dict.items():
        input_templates[target_parse] = prompt_generation_utils.gen_template(
            cfg.model.task,
            target_parse,
            cfg.model.input_template_type,
            plm,
            plm_tokenizer,
        )

    return input_templates


def get_model(cfg, plm, plm_tokenizer, input_templates):
    if cfg.model.model_type == "generative":
        model = GenerativeParser(cfg, plm, plm_tokenizer, input_templates)
    else:
        raise NotImplementedError

    if cfg.train.weight.trained:
        path_load = Path(cfg.train.weight.path)

        if cfg.model.task in [
            "ljp_civil",
            "ljp_criminal",
            "casename_classification",
            "statute_classification",
            "summarization",
        ]:
            ckpt = torch.load(path_load)
            if "state_dict" in ckpt:
                ckpt_state_dict = ckpt["state_dict"]
            else:
                ckpt_state_dict = ckpt
            model.load_state_dict(ckpt_state_dict, strict=False)

        else:
            raise NotImplementedError

        print(f"The model weights are loaded from {path_load}.")

    return model


def get_trainer(cfg):
    from pytorch_lightning import loggers as pl_loggers

    tparam = cfg.train
    mparam = cfg.model

    log_dir = Path(cfg.train.log_dir) / cfg.name
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

    pl.utilities.seed.seed_everything(seed=cfg.train.seed, workers=False)

    n_gpus = torch.cuda.device_count()

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=f"{cfg.train.validation_metric}_{cfg.train.validation_sub_param.method}",
            dirpath=gu.get_model_saving_path(tparam.weight.save_path_dir, cfg.name),
            save_top_k=1,
            mode="max",
            save_last=not True,
        )
    ]
    if tparam.optim.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(
                swa_epoch_start=tparam.optim.swa.swa_epoch_start,
                swa_lrs=tparam.optim.swa.lr,
                annealing_epochs=tparam.optim.swa.annealing_epochs,
            )
        )

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=tparam.accelerator,
        strategy=tparam.strategy,
        max_epochs=tparam.max_epochs,
        precision=mparam.precision if torch.cuda.is_available() else 32,
        num_sanity_val_steps=tparam.num_sanity_val_steps,
        gpus=n_gpus,
        check_val_every_n_epoch=tparam.check_val_every_n_epoch,
        gradient_clip_val=tparam.optim.gradient_clip_val,
        gradient_clip_algorithm=tparam.optim.gradient_clip_algorithm,
        accumulate_grad_batches=tparam.accumulate_grad_batches,
        val_check_interval=tparam.val_check_interval,
        profiler=tparam.profiler,
        fast_dev_run=tparam.fast_dev_run,
        callbacks=callbacks,
        limit_train_batches=tparam.get("limit_train_batches", 1.0),
        limit_val_batches=tparam.get("limit_val_batches", 1.0),
    )
    return trainer


def prepare_modules(mode, cfg):

    # get pretrained language models
    plm, plm_tokenizer, plm_model_config, TokenizerWrapperClass = get_plm(cfg)

    # gen templates
    input_templates = gen_input_templates(cfg, plm, plm_tokenizer)

    # get data module
    data_module = get_data_module(
        cfg, plm_tokenizer, TokenizerWrapperClass, input_templates
    )

    # get model
    model = get_model(cfg, plm, plm_tokenizer, input_templates)

    # get trainer
    trainer = get_trainer(cfg)

    return data_module, model, trainer
