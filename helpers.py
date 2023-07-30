from omegaconf import OmegaConf


def _trigger_post_init(cfg):
    try:
        cfg = OmegaConf.to_object(cfg)  # trigger __post_init__ of the dataclasses
        return cfg
    except RuntimeError as e:
        print(e)
        raise e
