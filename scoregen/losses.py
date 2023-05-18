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

"""All functions related to loss computation and optimization.
"""

import flax
import optax
import jax
import jax.numpy as jnp
import jax.random as random
from .models import utils as mutils
from .sde import VESDE
from .utils import batch_mul





def get_sde_loss_fn(sde, model, train, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """

  def loss_fn(rng, params, states, data):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    model_fn = mutils.get_model_fn(model, params, states, train=train)
    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    output, new_model_state = model_fn(perturbed_data, t, rng=step_rng)
    loss = jnp.sum(jnp.square(output + z)) / data.shape[0]

    return loss, new_model_state

  return loss_fn




def get_step_fn(sde, model, train, optimizer=None):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """

  loss_fn = get_sde_loss_fn(sde, model, train)

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.params
      states = state.model_state
      (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)
      #grad = jax.lax.pmean(grad, axis_name='batch')

      updates, new_opt_state = optimizer.update(grad, state.opt_state, params)
      new_params = optax.apply_updates(params, updates)

      #new_optimizer = optimize_fn(state, grad)
      new_params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, params
      )
      #new_params_ema = jax.tree_multimap(
      #  lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
      #  state.params_ema, new_optimizer.target
      #)
      step = state.step + 1
      new_state = state.replace(
        step=step,
        opt_state=new_opt_state,
        model_state=new_model_state,
        params=new_params,
        params_ema=new_params_ema
      )
    else:
      loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch)
      new_state = state

    #loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn
