# LBox Open
# Copyright (c) 2022-present LBox Co. Ltd.
# CC BY-NC 4.0

import json
import os
import pickle
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def stop_flag(idx, toy_size):
    # idx + 1 = length
    data_size = idx + 1
    if toy_size is not None:
        if toy_size <= data_size:
            return True
    else:
        return False


def save_pkl(path_save, data):
    with open(path_save, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path_load):
    with open(path_load, "rb") as f:
        data = pickle.load(f)
    return data


def save_json(path_save, data):
    with open(path_save, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)


def save_jsonl(path_save, data):
    with open(path_save, "w") as f:
        for t1 in data:
            f.writelines(json.dumps(t1, ensure_ascii=False))
            f.writelines("\n")


def load_jsonl(fpath, toy_size=None):
    data = []
    with open(fpath) as f:
        for i, line in tqdm(enumerate(f)):
            try:
                data1 = json.loads(line)
            except:
                print(f"{i}th sample failed.")
                print(f"We will wkip this!")
                print(line)
                data1 = None
            if data1 is not None:
                data.append(data1)
            if stop_flag(i, toy_size):
                break

    return data


def my_timeit(func):
    def wrapped_func(*args, **kwargs):
        st = time.time()
        results = func(*args, **kwargs)
        ed = time.time()
        print(f"func {func.__name__} taks {ed - st} sec.")
        return results

    return wrapped_func


def flatten_list(list_):
    out = []
    for x in list_:
        if isinstance(x, list):
            out += flatten_list(x)
        else:
            out += [x]

    return out


def load_cfg(path_cfg):
    import munch
    import yaml

    with open(path_cfg) as f:
        cfg = yaml.full_load(f)
    cfg = munch.munchify(cfg)
    cfg.name = path_cfg.__str__().split("/")[-1]
    return cfg


def get_model_saving_path(save_dir, cfg_name):
    return Path(save_dir) / cfg_name


def download_url(path_save, url):
    p = subprocess.Popen(["wget", "-q", "-O", path_save.__str__(), url])
    sts = os.waitpid(p.pid, 0)


def get_local_rank():
    """
    Pytorch lightning save local rank to environment variable "LOCAL_RANK".
    From rank_zero_only
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank
