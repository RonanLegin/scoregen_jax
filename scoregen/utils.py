# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Utility code for generating and saving image grids and checkpointing.

   The `save_image` code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""

import math
from typing import Any, Dict, Optional, TypeVar

import flax
import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)
