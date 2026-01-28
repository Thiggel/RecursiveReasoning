from .common import TransformerConfig
from .drm import DRMModel
from .hrm import HRMModel
from .trm import TRMModel
from .urm import URMModel


_MODEL_REGISTRY = {
    "urm": URMModel,
    "trm": TRMModel,
    "hrm": HRMModel,
    "drm": DRMModel,
}


def get_model_class(name: str):
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return _MODEL_REGISTRY[name]
