import copy
import os
import yaml
from argparse import ArgumentParser

try:
    from .infer_osp import main as infer_osp_main
except ImportError:
    from infer_osp import main as infer_osp_main


WAN_T2V_MODEL_NAME = "wan_t2v"


def build_want2v_config(config):
    config = copy.deepcopy(config)
    config["model_name"] = WAN_T2V_MODEL_NAME

    # WanT2V does not use OSP-Next skiparse sequence parallelism.
    config["use_sequence_parallel"] = False
    config["use_skiparse_sequence_parallel"] = False
    config["sp_size"] = 1
    config["skiparse_sp_size"] = 1

    model_config = config.setdefault("model_config", {})
    model_config.pop("skiparse_model_type", None)
    model_config.pop("sparse_ratio", None)
    model_config.pop("num_full_blocks", None)
    return config


def main(config):
    infer_osp_main(build_want2v_config(config))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/t2v.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
