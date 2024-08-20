import pickle
import os
import wandb
import torch
import yaml
from ml_collections import config_dict


def generic_file_load(full_path):
    """
    Load file from full path.
    Args:
        full_path (str): Full path to the file.
    """
    if full_path.split(".")[-1] in ["pk", "pkl"]:
        checkpoint = pickle.load(open(full_path, "rb"))
    elif full_path.split(".")[-1] in ["pt", "tch","torch", "t"]:
        checkpoint = torch.load(full_path, map_location='cpu')
    elif full_path.split(".")[-1] in ["info", "yaml"]:
        with open(full_path, "r") as f:
            checkpoint = yaml.safe_load(f)
    else:
        raise ValueError(f"File type {full_path.split('.')[-1]} not recognized."
                            "Supported types are .pt, .pkl, .info, .yaml"
                            "Please add the file type to the supported types in the function.")
    
    # fix str(None) to None
    for key, value in checkpoint.items():
        if value == "None":
            checkpoint[key] = None

    return checkpoint

def load_file_by_priority(base_paths, wandb_paths, file_names_by_priority):
    """
    Load file by priority from base paths and wandb paths.
    Args:
        base_paths (list): List of base paths to look for the file.
        wandb_paths (list): List of wandb paths to look for the file.
        file_names_by_priority (list): List of file names to look for in the base paths and wandb paths.
    """
    # First, try to load from the base paths
    for base_path in base_paths:
        for file_name in file_names_by_priority:
            full_path = os.path.join(base_path, file_name)
            if os.path.exists(full_path):
                return generic_file_load(full_path)
            else:
                print(f"{__file__}: error: file {file_name} does not exist in base path {base_path}.")
                continue

    # If not found in base paths, try downloading from WandB paths
    if wandb_paths: 
        tmp_base_path = "pretrained_models/tmp_model/"
        for wandb_path in wandb_paths:
            # Try loading again from the new model path
            for file_name in file_names_by_priority: 
                full_path = os.path.join(tmp_base_path, file_name)
                try:
                    wandb.Api().run(wandb_path).file(file_name).download(root=tmp_base_path, replace=True)
                    return generic_file_load(full_path)
                except Exception as e:
                    print(f"{__file__}: error: file {file_name} does not exist in wandb path {wandb_path}.\nError: {e}")
                    continue

    raise FileNotFoundError("None of the specified files found in the given base paths or WandB paths.")

def load_args_actor_critic(path, wandb_path=None):
    """
    Load args, actor and critic from path.
    Args:
        path (str): Path to the model.
    """
    previous_args = load_args(path, wandb_path)
    actor_checkpoint = load_file_by_priority([path], [wandb_path], ["actor.pt", "actor_latest.pt"])
    critic_checkpoint = load_file_by_priority([path], [wandb_path], ["critic.pt", "critic_latest.pt"])
    return previous_args, actor_checkpoint, critic_checkpoint

def load_args(path, wandb_path=None):
    """
    Load args from path.
    Args:
        path (str): Path to the model.

    Returns:
        args (config_dict.ConfigDict): Arguments.
    """
    args_dict = load_file_by_priority([path], [wandb_path], ["experiment.info"])
    return config_dict.ConfigDict(args_dict)
