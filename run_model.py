# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC 4.0

import argparse

from lbox_open.pipeline import prepare_modules
from lbox_open.utils import general_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_cfg", default="")
    parser.add_argument("--mode", default="")
    args = parser.parse_args()

    cfg = general_utils.load_cfg(args.path_cfg)

    if args.mode == "train":
        data_module, model, trainer = prepare_modules("train", cfg)
        trainer.fit(model, data_module)

    elif args.mode == "test":
        data_module, model, trainer = prepare_modules("train", cfg)
        trainer.test(model, datamodule=data_module)
    else:
        print(
            f"{args.mode} mode is not supported. The mode should be either 'train' or 'test'."
        )


if __name__ == "__main__":
    main()
