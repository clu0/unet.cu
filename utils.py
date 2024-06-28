import os
from typing import Dict, Any, List
import torch
import argparse


def write(tensor: torch.Tensor, handle):
    if tensor.is_cuda:
        handle.write(tensor.cpu().detach().numpy().astype("float32").tobytes())
    else:
        handle.write(tensor.detach().numpy().astype("float32").tobytes())

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict: Dict[str, Any]):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def list_image_files_recursive(data_dir: str) -> List[str]:
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split('.')[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(list_image_files_recursive(full_path))
    return results