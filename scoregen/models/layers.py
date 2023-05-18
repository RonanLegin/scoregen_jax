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
"""Layers for defining NCSN++.
"""
import functools
from typing import Any, Optional, Tuple
from . import up_or_down_sampling_1d as up_or_down_sampling
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


def get_act(config):
  """Get activation functions from the config file."""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.elu
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.relu
  elif config.model.nonlinearity.lower() == 'lrelu':
    return functools.partial(nn.leaky_relu, negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.swish
  else:
    raise NotImplementedError('activation function does not exist!')



def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return jnn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def conv1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """1x1 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(out_planes, kernel_size=(1,),
                   strides=(stride), padding='SAME', use_bias=bias,
                   kernel_dilation=(dilation),
                   kernel_init=default_init(init_scale),
                   bias_init=bias_init)(x)
  return output


def conv3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """3x1 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(
    out_planes,
    kernel_size=(3,),
    strides=(stride),
    padding='SAME',
    use_bias=bias,
    kernel_dilation=(dilation),
    kernel_init=default_init(init_scale),
    bias_init=bias_init)(x)
  return output


class NIN(nn.Module):
  num_units: int
  init_scale: float = 0.1

  @nn.compact
  def __call__(self, x):
    in_dim = int(x.shape[-1])
    W = self.param('W', default_init(scale=self.init_scale), (in_dim, self.num_units))
    b = self.param('b', jnn.initializers.zeros, (self.num_units,))
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + (self.num_units,)
    return y


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  embedding_size: int = 256
  scale: float = 1.0

  @nn.compact
  def __call__(self, x):
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""
  skip_rescale: bool = False
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x):
    B, H, C = x.shape
    h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
    q = NIN(C)(h)
    k = NIN(C)(h)
    v = NIN(C)(h)

    w = jnp.einsum('bhc,bHc->bhH', q, k) * (int(C) ** (-0.5))
    w = jnp.reshape(w, (B, H, H))
    w = jax.nn.softmax(w, axis=-1)
    w = jnp.reshape(w, (B, H, H))
    h = jnp.einsum('bhH,bHc->bhc', w, v)
    h = NIN(C, init_scale=self.init_scale)(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  """ResBlock adapted from BigGAN."""
  act: Any
  up: bool = False
  down: bool = False
  out_ch: Optional[int] = None
  dropout: float = 0.1
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 1)
  skip_rescale: bool = True
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_1d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_1d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_1d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_1d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_1d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_1d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_1d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_1d(x, factor=2)

    h = conv3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, :]

    h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3(h, out_ch, init_scale=self.init_scale)
    if C != out_ch or self.up or self.down:
      x = conv1(x, out_ch)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
