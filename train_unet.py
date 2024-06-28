from typing import List, Optional, Tuple
from abc import abstractmethod
from collections import defaultdict
import sys
import datetime
import math
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch._inductor.config as config
import numpy as np

from utils import add_dict_to_argparser, list_image_files_recursive, write


torch.manual_seed(0)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# nn Modules, mostly adapted from https://github.com/openai/guided-diffusion

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.op = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.op(x)
        return x
        

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_scale_shift_norm=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        # make in_layers explicit for manual saving
        self.gn1 = nn.GroupNorm(32, channels)
        self.silu1 = nn.SiLU()
        self.cv3_1 = nn.Conv2d(channels, self.out_channels, 3, padding=1)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels)
            self.x_upd = Upsample(channels)
        elif down:
            self.h_upd = Downsample(channels)
            self.x_upd = Downsample(channels)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # make emb_layers explicit
        self.silu_emb = nn.SiLU()
        self.l_emb = nn.Linear(
            emb_channels,
            2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        )


        # make out_layers explicit
        self.gn2 = nn.GroupNorm(32, self.out_channels)
        self.silu2 = nn.SiLU()
        self.cv3_2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)

        # get rid of the use_conv option that was never used in the original model
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb, debug=False):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.

        NOTE: original version used checkpointing for the forward function,
        but that doesn't change the values
        """
        # first layer
        h = self.gn1(x)
        h = self.silu1(h)
        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)
        h = self.cv3_1(h)

        # emb layer
        emb_out = self.silu_emb(emb)
        emb_out = self.l_emb(emb_out).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # second layer
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.gn2(h)
            h = self.silu2(h)
            h = self.cv3_2(h)

        return self.skip_connection(x) + h

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(ch)
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.n_heads, ch, length),
            k.reshape(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        NH=1,
        HS=32,
        gn_n_groups=32,
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        self.channels = channels
        if HS == -1:
            assert channels % NH == 0
            self.HS = channels // NH
            self.NH = NH
        else:
            assert channels % HS == 0
            self.HS = HS
            self.NH = channels // HS
        self.gn = nn.GroupNorm(gn_n_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.NH)

        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.gn(x)
        gn = h.clone()
        qkv = self.qkv(h)
        h = self.attention(qkv)
        att = h.clone()
        h = self.proj(h)
        proj = h.clone()
        if self.debug:
            return (x + h).reshape(b, c, *spatial), gn, qkv, att, proj
        else:
            return (x + h).reshape(b, c, *spatial)


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability. NOTE: not using this right now
    :param channel_mult: channel multiplier for each level of the UNet.
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        # time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            NH=num_heads,
                            HS=num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                NH=num_heads,
                HS=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            NH=num_heads_upsample,
                            HS=num_head_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(input_ch, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        #emb = self.time_embed(timesteps)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)

# copied from guided diffusion
def timestep_embedding(timesteps, dim, max_period=1000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# ------------------------------------------------------------
# data loader

def load_data(
    data_dir: str,
    batch_size: int,
    class_cond=False,
    randomize=False,
):
    all_img_files = list_image_files_recursive(data_dir)
    classes = None
    if class_cond:
        class_names = [os.path.basename(path).split("_")[0] for path in all_img_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(all_img_files, classes, random_flip=randomize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=randomize, num_workers=2, drop_last=True)
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        classes: Optional[List[str]] = None,
        random_flip = True,
    ):
        self.image_paths = image_paths
        self.classes = classes
        self.random_flip = random_flip
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = np.array(pil_image)
        if self.random_flip and np.random.rand() < 0.5:
            arr = arr[:, ::-1]
            
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.transpose(2, 0, 1)
        
        if self.classes is not None:
            return arr, np.array(self.classes[idx], dtype=np.int32)
        return arr

# ------------------------------------------------------------
# logger

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            self.file = filename_or_file
            self.own_file = False

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writekvs(self, kvs):
        key2str = {}
        for key, val in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if len(key2str) == 0:
            print("WARNING: tried to write empty k-v dict")
            return
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # write data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for key, val in sorted(key2str.items()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        self.file.flush()

    def writeseq(self, seq):
        seq = list(seq)
        for i, elem in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:
                self.file.write(" ")
            self.file.write("\n")
            self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "a+t")

    def writekvs(self, kvs):
        keys = sorted(kvs.keys())
        if self.file.tell() == 0:
            # write header
            self.file.write(",".join(keys) + "\n")

        # write values
        self.file.write(",".join(str(kvs[key]) for key in keys) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class Logger(object):
    def __init__(self, output_formats):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        old_val, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = old_val * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] += 1

    def dumpkvs(self):
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                if isinstance(args, str):
                    fmt.writeseq([args])
                else:
                    fmt.writeseq(map(str, args))

    def close(self):
        for fmt in self.output_formats:
            fmt.close()


# ------------------------------------------------------------
# args
def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=32,
        num_classes=10,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False, # not supporting this option in cuda right now
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res

def create_argparser():
    defaults = dict(
        data_dir=".",
        log_dir="logs",
        model_dir="models",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        iterations=100000,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        randomize=False,
        compile=1,
        init_model_only=False,
        init_model_filename="unet_init.bin"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


# ------------------------------------------------------------
# Model

def create_model(
    num_channels,
    num_res_blocks,
    image_size,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    num_classes=None,
    use_scale_shift_norm=False,
    resblock_updown=False,
    **kwargs,
):
    if channel_mult == "":
        channel_mult = (1, 2, 3, 4)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )

def save_model_params_to_bin(
    model: nn.Module,
    filename: str,
    B: int = 32,
    C_in: int = 3,
    C_model: int = 64,
    C_out: int = 3,
    H: int = 64,
    W: int = 64,
    max_period: int = 1000,
):
    # create model header
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 12345678 # magic number
    header[1] = B
    header[2] = C_in
    header[3] = C_model
    header[4] = C_out
    header[5] = H
    header[6] = W
    header[7] = max_period
    header[8] = 0 # don't save adamw states in python
    header[9] = 0 # don't save rng states in python

    with open(filename, "wb") as f:
        f.write(header.numpy().tobytes())
        for _, param in model.named_parameters():
            write(param, f)

# ------------------------------------------------------------
# diffusion sampler and loss, copied from https://github.com/openai/guided-diffusion

def sample_timesteps(batch_size: int, weights: np.ndarray):
    """
    Importance sampling for timesteps for a batch
    """
    p = weights / weights.sum()
    indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
    indices = torch.from_numpy(indices_np).long().to(DEVICE)
    weights_np = 1 / (len(p) * p[indices_np])
    weights = torch.from_numpy(weights_np).float().to(DEVICE)
    return indices, weights

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def mean_flat(tensor: torch.Tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """
    taken from guided diffusion, currently only supporting constant variance and mse loss
    """
    def __init__(self, betas):
        betas = np.array(betas, dtype=np.float32) # just using float32 everywhere for easier porting to cuda
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start: torch.Tensor, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t.long(), x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.long(), x_start.shape)
            * noise
        )
    
    def mse_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor]=None,
    ):
        """
        compute the mse loss
        
        not learning variance for now
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=DEVICE)
        x_t = self.q_sample(x_start, t, noise)
        out = model(x_t, t)
        loss = mean_flat((out - noise) ** 2)
        return loss
    


# ------------------------------------------------------------

def compute_norms(model: nn.Module) -> Tuple[float, float]:
    weight_norm = 0.0
    grad_norm = 0.0
    for p in model.parameters():
        with torch.no_grad():
            weight_norm += p.norm(p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += p.grad.norm(p=2, dtype=torch.float32).item() ** 2
    return weight_norm, grad_norm


def main():
    args = create_argparser().parse_args()

    if DEVICE != "cpu":
        torch.cuda.reset_peak_memory_stats()

    model = create_model(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    if args.init_model_only:
        print(f"Only initializing model, saving binary weights to {args.init_model_filename}")
        save_model_params_to_bin(model, args.init_model_filename)
        return

    model.train()
    model.to(DEVICE)
    # compile code copied from llm.c
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print("compiling the model...")
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        randomize=args.randomize,
    )
    
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion = GaussianDiffusion(betas)

    # initialize save dir and logger
    log_save_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_save_dir, exist_ok=True)
    output_formats = [
        HumanOutputFormat(sys.stdout),
        HumanOutputFormat(os.path.join(log_save_dir, f"log.txt")),
        CSVOutputFormat(
            os.path.join(log_save_dir, f"progress.csv")
        ),
    ]
    logger = Logger(output_formats)

    logger.log(f"training args: \n{args}")

    model_save_dir = os.path.join(
        args.model_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(model_save_dir, exist_ok=True)

    step = 1
    weights = np.ones(args.diffusion_steps) # uniform weights for now
    logger.log(f"starting trainging at step {step} for {args.iterations} steps")
    #start_time = time()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_time.record()
    torch.cuda.synchronize()
    while (step <= args.iterations):
        logger.logkv("step", step)
        logger.logkv("samples", step * args.batch_size)
        opt.zero_grad()
        x = next(data) # ignore class labels for now
        x = x.to(DEVICE)
        t, _ = sample_timesteps(args.batch_size, weights) # don't keep track of computed weights because using uniform weights only

        loss = diffusion.mse_loss(model, x, t)
        loss = loss.mean()
        loss.backward()
        grad_norm, param_norm = compute_norms(model)
        logger.logkv("loss", loss.item())
        logger.logkv_mean("loss_mean", loss.item())
        logger.logkv_mean("grad_norm_mean", grad_norm)
        logger.logkv("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()

        if step % args.log_interval == 0:
            #logger.log(f"finished step {step} in {time() - start_time:.2f} seconds")
            torch.cuda.synchronize()
            end_time.record()
            torch.cuda.synchronize()
            logger.log(f"finished step {step} in {start_time.elapsed_time(end_time) / 1000} seconds")
            logger.logkv("peak_mem_MiB", torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024)
            logger.dumpkvs()
        if step % args.save_interval == 0:
            logger.log(f"step {step}, saving model")
            filename = f"model_{step}.pt"
            save_path = os.path.join(model_save_dir, filename)
            torch.save(model.state_dict(), save_path)
        step += 1
    if step % args.save_interval != 0:
        logger.log(f"step {step}, saving model")
        filename = f"model_{step}.pt"
        save_path = os.path.join(model_save_dir, filename)
        torch.save(model.state_dict(), save_path)



if __name__ == "__main__":
    main()
