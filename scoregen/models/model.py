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
from jax import lax

ResnetBlockBigGAN = layers.ResnetBlockBigGANpp
ResnetBlockBigGANOG = layers.ResnetBlockBigGANppOG
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
        res_mult = config.model.res_mult
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        num_resolutions = len(ch_mult)
        
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        skip_rescale = config.model.skip_rescale
        init_scale = config.model.init_scale

        temb = layers.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)(time_cond)
        temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
        temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))

        B, H, C = x.shape

        ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                        act=act,
                                        dropout=dropout,
                                        fir=fir,
                                        fir_kernel=fir_kernel,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale)
        
        sigma_max = config.model.sigma_max
        sigma_min = config.model.sigma_min
        
        sigma_t = sigma_min * (sigma_max/sigma_min)**time_cond
        sigma_t = sigma_t[:, None, None] 
        
        x *= 1/jnp.sqrt(sigma_t**2 + 1.0)
        
        ## Downsampling block
        hs = [conv3x3(x , nf)]
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                hs.append(h)

            if i_level != num_resolutions - 1:
                h = ResnetBlock(down=True, up_down_factor=res_mult[i_level])(hs[-1], temb, train)
                hs.append(h)

        h = hs[-1]
        h = ResnetBlock()(h, temb, train)
        h = ResnetBlock()(h, temb, train)

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1),
                                                         temb,
                                                         train)
            if i_level != 0:
                h = ResnetBlock(up=True, up_down_factor=res_mult[i_level-1])(h, temb, train)

        assert not hs

        h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        h = conv3x3(h, x.shape[-1], init_scale=1.)

        h *= sigma_t/jnp.sqrt(sigma_t**2 + 1.0) # scaling factor of convolved white Gaussian distribution
        return h
    
class NCSNpp1D2(nn.Module):
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
        num_resolutions = len(ch_mult)
        
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        skip_rescale = config.model.skip_rescale
        init_scale = config.model.init_scale

        temb = layers.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)(time_cond)
        temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
        temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))

        B, H, C = x.shape

    
        # AttnBlock = functools.partial(layers.AttnBlockpp,
        #                               init_scale=init_scale,
        #                               skip_rescale=skip_rescale)

        ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                        act=act,
                                        dropout=dropout,
                                        fir=fir,
                                        fir_kernel=fir_kernel,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale)
        sigma_max = config.model.sigma_max
        sigma_min = config.model.sigma_min
        
        sigma_t = sigma_min * (sigma_max/sigma_min)**time_cond
        sigma_t = sigma_t[:, None, None] 
        
        x *= 1/jnp.sqrt(sigma_t**2 + 1.0)
        ## Downsampling block
        hs = [conv3x3(x , nf)]
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                hs.append(h)

            if i_level != num_resolutions - 1:
                h = ResnetBlock(down=True)(hs[-1], temb, train)
                hs.append(h)

        h = hs[-1]
        h = ResnetBlock()(h, temb, train)
        h = ResnetBlock()(h, temb, train)

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1),
                                                         temb,
                                                         train)
            if i_level != 0:
                h = ResnetBlock(up=True)(h, temb, train)

        assert not hs

        h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        h = conv3x3(h, x.shape[-1], init_scale=1.)
        
        
        h *= sigma_t/jnp.sqrt(sigma_t**2 + 1.0) # scaling factor of convolved white Gaussian distribution
        return h
    
# class NCSNpp1Du(nn.Module):
#     """NCSN++ model"""
#     config: dict

#     @nn.compact
#     def __call__(self, x, time_cond, train=True):
#         # config parsing
#         config = self.config
#         act = get_act(config)

#         nf = config.model.nf
#         ch_mult = config.model.ch_mult
#         num_res_blocks = config.model.num_res_blocks
#         attn_resolutions = config.model.attn_resolutions
#         dropout = config.model.dropout
#         num_resolutions = len(ch_mult)
        
#         fir = config.model.fir
#         fir_kernel = config.model.fir_kernel
#         skip_rescale = config.model.skip_rescale
#         init_scale = config.model.init_scale
#         scalelongskip_rescale = config.model.scalelongskip_rescale
        
#         if scalelongskip_rescale:
#             sls_coeff = 2**(-0.5)
#         else:
#             sls_coeff = 1.

#         temb = layers.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)(time_cond)
        
#         #if train:
#         u_sigma = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
#         u_sigma = nn.Dense(1, kernel_init=default_initializer())(act(u_sigma))
        
#         temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
#         temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))
        
#         B, H, C = x.shape


#         # AttnBlock = functools.partial(layers.AttnBlockpp,
#         #                               init_scale=init_scale,
#         #                               skip_rescale=skip_rescale)

#         ResnetBlock = functools.partial(ResnetBlockBigGAN,
#                                         act=act,
#                                         dropout=dropout,
#                                         fir=fir,
#                                         fir_kernel=fir_kernel,
#                                         init_scale=init_scale,
#                                         skip_rescale=skip_rescale)
#         ## Downsampling block
#         hs = [conv3x3(x , nf)]
#         for i_level in range(num_resolutions):
#             for i_block in range(num_res_blocks):
#                 h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
#                 hs.append(h)

#             if i_level != num_resolutions - 1:
#                 h = ResnetBlock(down=True)(hs[-1], temb, train)
#                 hs.append(h)

#         h = hs[-1]
#         h = ResnetBlock()(h, temb, train)
#         h = ResnetBlock()(h, temb, train)

#         # Upsampling block
#         for i_level in reversed(range(num_resolutions)):
#             for i_block in range(num_res_blocks + 1):
                
#                 h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()*sls_coeff], axis=-1),
#                                                          temb,
#                                                          train)
#             if i_level != 0:
#                 h = ResnetBlock(up=True)(h, temb, train)

#         assert not hs

#         h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
#         h = conv3x3(h, x.shape[-1], init_scale=1.)
        
#         if train:
#             return h, u_sigma
#         else:
#             return h
    

    
class WhiteSkip1D(nn.Module):
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
        num_resolutions = len(ch_mult)
        
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        skip_rescale = config.model.skip_rescale
        init_scale = config.model.init_scale
            
        sigma_max = config.model.sigma_max
        sigma_min = config.model.sigma_min
        
        sigma_t = sigma_min * (sigma_max/sigma_min)**time_cond
        sigma_t = sigma_t[:, None, None] 
        sigma_n = 1.
        
        gaussian_score = -(sigma_t * x)/(sigma_t**2 + sigma_n**2)
        
        temb = layers.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)(time_cond)
        temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
        temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))

        B, H, C = x.shape


        # AttnBlock = functools.partial(layers.AttnBlockpp,
        #                               init_scale=init_scale,
        #                               skip_rescale=skip_rescale)

        ResnetBlockWhiteSkip = functools.partial(ResnetWhiteSkip,
                                        act=act,
                                        dropout=dropout,
                                        init_scale=1.,
                                        skip_rescale=skip_rescale)

        ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                act=act,
                                up=False,
                                down=False,
                                dropout=dropout,
                                fir=fir,
                                fir_kernel=fir_kernel,
                                init_scale=init_scale,
                                skip_rescale=skip_rescale)
            
        
        h_gaussian_score = conv1x1(gaussian_score, 2*nf, init_scale=1.)
        h_gaussian_score = ResnetBlockWhiteSkip(out_ch=2*nf)(h_gaussian_score, None, train)
        
        # Downsampling block
        hs = [conv3x3(x , nf)]
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                hs.append(h)

            if i_level != num_resolutions - 1:
                h = ResnetBlock(down=True)(hs[-1], temb, train)
                hs.append(h)

        h = hs[-1]
        h = ResnetBlock()(h, temb, train)
        h = ResnetBlock()(h, temb, train)

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            if i_level == 0:                
                for i_block in range(num_res_blocks + 1):
                    h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop(), h_gaussian_score], axis=-1),
                                                             temb,
                                                             train)
            else:
                for i_block in range(num_res_blocks + 1):
                    h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1),
                                                             temb,
                                                             train)
            if i_level != 0:
                h = ResnetBlock(up=True)(h, temb, train)

        assert not hs

        h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(jnp.concatenate([h, h_gaussian_score], axis=-1)))
        h = conv1x1(h, x.shape[-1], init_scale=1.)

        return h

