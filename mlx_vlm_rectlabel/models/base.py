import inspect
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}
