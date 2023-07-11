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
# pytype: skip-file
"""Various sampling methods."""
import functools

import jax
import jax.numpy as jnp
import jax.random as random
import flax

from .models import utils as mutils
from .utils import batch_mul


# class EulerMaruyamaPredictor():
#   def __init__(self, sde, score_fn):
#     super().__init__(sde, score_fn)
#     self.sde = sde
#     # Compute the reverse SDE/ODE
#     self.rsde = sde.reverse(score_fn)
#     self.score_fn = score_fn

#   def update_fn(self, rng, x, t):
#     dt = -1. / self.rsde.N
#     z = random.normal(rng, x.shape)
#     drift, diffusion = self.rsde.sde(x, t)
#     x_mean = x + drift * dt
#     x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)
#     return x, x_mean



def get_sampler(sde, model, shape, eps=1e-5, dtype=jnp.float32):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    n_steps: An integer. The number of corrector steps per predictor update.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

  Returns:
    A sampling function that takes random states, and a replcated training state and returns samples as well as
    the number of function evaluations during sampling.
  """

  def sampler(rng, state):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples, number of function evaluations
    """

    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape, dtype=dtype)
    timesteps = jnp.linspace(sde.T, eps, sde.N, dtype=dtype)

    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, return_state=False)
    rsde = sde.reverse(score_fn)
    dt = -1. / rsde.N

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0], dtype=dtype) * t
      rng, step_rng = random.split(rng)

      z = random.normal(rng, x.shape, dtype=dtype)
      drift, diffusion = rsde.sde(x, vec_t)
      x_mean = x + drift * dt
      x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)

      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    # Denoising is equivalent to running one predictor step without adding noise.
    return x_mean

  return sampler


