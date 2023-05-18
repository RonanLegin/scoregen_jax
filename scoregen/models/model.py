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

from . import layers
import flax.linen as nn
import functools
import jax.numpy as jnp

ResnetBlockBigGAN = layers.ResnetBlockBigGANpp
conv3x3 = layers.conv3
conv1x1 = layers.conv1
get_act = layers.get_act
default_initializer = layers.default_init


class NCSNpp1D(nn.Module):
  """NCSN++ model"""
  config: dict

  @nn.compact
  def __call__(self, x, time_cond, train=True):
    # config parsing
    config = self.config
    act = get_act(config)

    nf = config.model.nf
    ch_mult = config.model.ch_mult
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    num_resolutions = len(ch_mult)

    conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    skip_rescale = config.model.skip_rescale
    resblock_type = config.model.resblock_type.lower()
    init_scale = config.model.init_scale


    #used_sigmas = time_cond
    temb = layers.GaussianFourierProjection(
      embedding_size=nf,
      scale=config.model.fourier_scale)(time_cond)#jnp.log(used_sigmas))



    if conditional:
      temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
      temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))
    else:
      temb = None

    B, H, C = x.shape

    
    AttnBlock = functools.partial(layers.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                    act=act,
                                    dropout=dropout,
                                    fir=fir,
                                    fir_kernel=fir_kernel,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale)


    # Downsampling block
    hs = [conv3x3(x , nf)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
        if h.shape[1] in attn_resolutions:
          h = AttnBlock()(h)
        hs.append(h)

      if i_level != num_resolutions - 1:
        h = ResnetBlock(down=True)(hs[-1], temb, train)

        hs.append(h)

    h = hs[-1]
    h = ResnetBlock()(h, temb, train)


    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1),
                                                      temb,
                                                      train)

      if h.shape[1] in attn_resolutions:
        h = AttnBlock()(h)

      if i_level != 0:
        h = ResnetBlock(up=True)(h, temb, train)

    assert not hs

    h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = conv3x3(h, x.shape[-1], init_scale=init_scale)
      
    return h
