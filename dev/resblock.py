"""
Residual block from guided-diffusion
"""
from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import write
from tqdm import tqdm

torch.manual_seed(0)


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
        # dropout, # not using dropout right now
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
        h_gn1 = h.clone()
        h = self.silu1(h)
        h_silu1 = h.clone()
        h_ud = None
        if self.updown:
            h = self.h_upd(h)
            h_ud = h.clone()
            x = self.x_upd(x)
        h = self.cv3_1(h)

        h_1 = h.clone()
        x_1 = x.clone()

        # emb layer
        emb_out = self.silu_emb(emb)
        emb_out = self.l_emb(emb_out).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        emb_1 = emb_out.clone()
        emb_broad = emb_1.expand(emb_1.shape[0], emb_1.shape[1], h.shape[2], h.shape[3])
        
        # second layer
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h_plus_emb = h.clone()
            h = self.gn2(h)
            h_gn2 = h.clone()
            h = self.silu2(h)
            h_silu2 = h.clone()
            h = self.cv3_2(h)
        h_2 = h.clone()

        if debug:
            return self.skip_connection(x) + h, h_gn1, h_silu1, h_ud, h_1, x_1, emb_1, emb_broad, h_plus_emb, h_gn2, h_silu2, h_2
            #return self.skip_connection(x) + h, h_silu2
        return self.skip_connection(x) + h


class ResBlockO(TimestepBlock):
    """
    Copied original implementation
    """

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels)
            self.x_upd = Upsample(channels)
        elif down:
            self.h_upd = Downsample(channels)
            self.x_upd = Downsample(channels)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            # there used to be a zero_module call here, not sure what the point is
            # omitting for now
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

if __name__ == "__main__":
    # check that the two implementations are the same
    B = 32
    C = 192
    C_emb = 256
    C_out = 64
    H = 64
    W = 64
    up = False
    down = False
    res_block = ResBlock(C, C_emb, out_channels=C_out, up=up, down=down)
    res_block_o = ResBlockO(C, C_emb, out_channels=C_out, up=up, down=down)
    
    rename_dict = {
        "gn1.weight": "in_layers.0.weight",
        "gn1.bias": "in_layers.0.bias",
        "cv3_1.weight": "in_layers.2.weight",
        "cv3_1.bias": "in_layers.2.bias",
        "l_emb.weight": "emb_layers.1.weight",
        "l_emb.bias": "emb_layers.1.bias",
        "gn2.weight": "out_layers.0.weight",
        "gn2.bias": "out_layers.0.bias",
        "cv3_2.weight": "out_layers.2.weight",
        "cv3_2.bias": "out_layers.2.bias",
        "skip_connection.weight": "skip_connection.weight",
        "skip_connection.bias": "skip_connection.bias",
    }
    new_state_dict = res_block.state_dict()
    old_state_dict = {}
    # rename the keys for updating old dict
    for new_key, old_key in rename_dict.items():
        if new_key in new_state_dict:
            old_state_dict[old_key] = new_state_dict[new_key]
    res_block_o.load_state_dict(old_state_dict)

    # random input
    # x = torch.randn(1, C, 32, 32)
    # emb = torch.randn(1, C_emb)
    # handcrafted inputs
    # x = (torch.arange(B * C * H * W).view(B, C, H, W) * 0.1).requires_grad_(True)
    # x_o = (torch.arange(B * C * H * W).view(B, C, H, W) * 0.1).requires_grad_(True)
    # emb = (torch.arange(B * C_emb).view(B, C_emb) * 0.1).requires_grad_(True)
    # emb_o = (torch.arange(B * C_emb).view(B, C_emb) * 0.1).requires_grad_(True)
    x = (torch.randn(B, C, H, W) * (H * W)**-0.5).requires_grad_(True)
    x_o = x.detach().clone().requires_grad_(True)
    emb = (torch.randn(B, C_emb) * (C_emb)**-0.5).requires_grad_(True)
    emb_o = emb.detach().clone().requires_grad_(True)
    out, h_gn1, h_silu1, h_ud, h_1, x_1, emb_1, emb_broad, h_plus_emb, h_gn2, h_silu2, h_2 = res_block(x, emb, debug=True)
    out_o = res_block_o(x_o, emb_o)
    print(f"out difference: {torch.sum(out - out_o).detach().item()}")

    dout = torch.randn_like(out) / (B * (H * W)**0.5)
    fakeloss = torch.sum(out * dout)
    fakeloss.backward()
    fakeloss_o = torch.sum(out_o * dout)
    fakeloss_o.backward()

    print(f"dx difference: {torch.sum(x.grad - x_o.grad).detach().item()}")
    print(f"demb difference: {torch.sum(emb.grad - emb_o.grad).detach().item()}")

    for name, param in res_block.named_parameters():
        print(name, param.shape)
        print(f"{name} grad shape: {param.grad.shape}")

    
    # save weights
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 12345678 # magic number
    header[1] = B
    header[2] = C
    header[3] = C_emb
    header[4] = C_out
    header[5] = H
    header[6] = W
    header[7] = 1 if up else 0
    header[8] = 1 if down else 0
    header[9] = 32 # num groups for group norm

    with open("resblock_params.bin", "wb") as f:
        f.write(header.numpy().tobytes())
        for _, param in res_block.named_parameters():
            write(param, f)
    
    # save acts and grads for debugging
    with open("resblock_states.bin", "wb") as f:
        write(x, f)
        write(emb, f)
        write(h_gn1, f)
        write(h_silu1, f)
        write(h_1, f)
        write(x_1, f)
        write(emb_1, f)
        write(h_plus_emb, f)
        write(h_gn2, f)
        write(h_silu2, f)
        write(h_2, f)
        write(out, f)
        write(dout, f)
        write(x.grad, f)
        write(emb.grad, f)
        write(emb_broad, f)
        if up or down:
            print("writing h_ud")
            write(h_ud, f)
        for _, param in res_block.named_parameters():
            write(param.grad, f)
    
    print("\nDone\n")

    print("resblock benchmarking")
    repeats = 100
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    full_start = torch.cuda.Event(enable_timing=True)
    full_end = torch.cuda.Event(enable_timing=True)
    forward_time = 0.0
    full_time = 0.0

    print("benchmarking forward pass")
    device = "cuda:1"
    res_block.to(device)
    x = x.to(device)
    emb = emb.to(device)
    torch.cuda.synchronize()
    forward_start.record()
    for i in tqdm(range(repeats)):
        out = res_block(x, emb)
    torch.cuda.synchronize()
    forward_end.record()
    torch.cuda.synchronize()
    forward_time = forward_start.elapsed_time(forward_end) / repeats
    print(f"forward time: {forward_time} ms")