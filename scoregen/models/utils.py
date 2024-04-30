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

"""All functions and modules related to model definition.
"""
from typing import Any
import json

import flax
import functools
import jax.numpy as jnp
import jax
import numpy as np
from flax.training import checkpoints
from .model import NCSNpp1D


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(config_path):
  # Load parameters from a json file back into the Config class
  with open(config_path, 'r') as f:
      loaded_config_dict = json.load(f)

  # Convert dictionaries back into Config objects
  for key in loaded_config_dict:
      if isinstance(loaded_config_dict[key], dict):
          loaded_config_dict[key] = Config(**loaded_config_dict[key])

  loaded_config = Config(**loaded_config_dict)

  return loaded_config


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
  step: int
  opt_state: Any
  lr: float
  model_state: Any
  params: Any
  ema_rate: float
  params_ema: Any
  rng: Any


def init_model(rng, config):
  """ Initialize a `flax.linen.Module` model. """
  model_name = config.model.name
  model = NCSNpp1D(config)

  input_shape = (1, config.data.data_size, config.data.num_channels)
  fake_input = jnp.zeros(input_shape, dtype=jnp.float32)
  fake_label = jnp.zeros((1,), dtype=jnp.float32)
  params_rng, dropout_rng = jax.random.split(rng)

  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label)
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  #init_model_state, initial_params = variables.pop('params')
  init_model_state, initial_params = flax.core.pop(variables, 'params')
  return model, init_model_state, initial_params


def get_model_fn(model, params, states, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, t, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = {'params': params, **states}
    if not train:
      return model.apply(variables, x, t, train=False, mutable=False), states
    else:
      rngs = {'dropout': rng}
      return model.apply(variables, x, t, train=True, mutable=list(states.keys()), rngs=rngs)

  return model_fn


def get_score_fn(sde, model, params, states, train=False, return_state=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all other mutable parameters.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    return_state: If `True`, return the new mutable states alongside the model output.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, params, states, train=train)

  def score_fn(x, t, rng=None):
    sigma_t = sde.marginal_prob(jnp.zeros_like(x), t)[1]

    output, state = model_fn(x, t, rng)
    score = output / sigma_t.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
    
    if return_state:
      return score, state
    else:
      return score

  return score_fn


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)

