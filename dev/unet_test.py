from tqdm import tqdm
import torch
import torch.nn as nn
from unet import timestep_embedding, AttentionBlock, UNetModel
from resblock import ResBlock, Downsample, Upsample
from utils import write
import torch._inductor.config as config

import sys
import os
sys.path.append(os.path.abspath('..'))
from train_unet import GaussianDiffusion, get_named_beta_schedule
from prepare_data import load_img_from_bin

torch.manual_seed(0)

# old code that was used to build up the unet model
#class UNetTestModel(nn.Module):
#    """
#    This class should eventually mirror the UNetModel in unet.py
#    
#    But we will build it up layer by layer, and use its outputs to test
#    the cuda implementation
#    """
#    def __init__(
#        self,
#        in_channels,
#        model_channels,
#    ):
#        super().__init__()
#        
#        self.in_channels = in_channels
#        self.model_channels = model_channels
#
#        time_embed_dim = model_channels * 4
#
#        # time step embeddings
#        self.time_lin1 = nn.Linear(model_channels, time_embed_dim)
#        #self.time_lin1.weight = nn.Parameter(torch.arange(model_channels * time_embed_dim).view(time_embed_dim, model_channels) * 0.1)
#        #print(f"lin1 weight: {self.time_lin1.weight}")
#        #self.time_lin1.bias = nn.Parameter(torch.arange(time_embed_dim) * 0.1)
#        #print(f"lin1 bias: {self.time_lin1.bias}")
#        self.time_silu = nn.SiLU()
#        self.time_lin2 = nn.Linear(time_embed_dim, time_embed_dim)
#
#        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
#
#        self.res1 = ResBlock(model_channels, time_embed_dim, model_channels)
#        self.res2 = ResBlock(model_channels, time_embed_dim, model_channels)
#
#        self.down1 = Downsample(model_channels)
#        self.res3 = ResBlock(model_channels, time_embed_dim, model_channels * 2)
#        self.res4 = ResBlock(model_channels * 2, time_embed_dim, model_channels * 2)
#        self.down2 = Downsample(model_channels * 2)
#
#        self.res5 = ResBlock(model_channels * 2, time_embed_dim, model_channels * 3)
#        self.att1 = AttentionBlock(model_channels * 3, HS=32)
#        self.res6 = ResBlock(model_channels * 3, time_embed_dim, model_channels * 3)
#        self.att2 = AttentionBlock(model_channels * 3, HS=32)
#        self.down3 = Downsample(model_channels * 3)
#
#        self.res7 = ResBlock(model_channels * 3, time_embed_dim, model_channels * 4)
#        C = model_channels * 4
#        self.att3 = AttentionBlock(C, HS=32)
#        self.res8 = ResBlock(C, time_embed_dim, C)
#        self.att4 = AttentionBlock(C, HS=32)
#
#        self.res9 = ResBlock(C, time_embed_dim, C)
#        self.att5 = AttentionBlock(C, HS=32)
#        self.res10 = ResBlock(C, time_embed_dim, C)
#
#        self.upres1 = ResBlock(C * 2, time_embed_dim, C)
#        self.upatt1 = AttentionBlock(C, HS=32)
#        self.upres2 = ResBlock(C * 2, time_embed_dim, C)
#        self.upatt2 = AttentionBlock(C, HS=32)
#        C_skip = model_channels * 3
#        self.upres3 = ResBlock(C + C_skip, time_embed_dim, C)
#        self.upatt3 = AttentionBlock(C, HS=32)
#
#        self.up1 = Upsample(C)
#
#        OC = model_channels * 3
#        self.upres4 = ResBlock(C + C_skip, time_embed_dim, OC)
#        C = OC
#        self.upatt4 = AttentionBlock(C)
#        self.upres5 = ResBlock(C + C_skip, time_embed_dim, OC)
#        self.upatt5 = AttentionBlock(C)
#        C_skip = model_channels * 2
#        self.upres6 = ResBlock(C + C_skip, time_embed_dim, OC)
#        self.upatt6 = AttentionBlock(C)
#        self.up2 = Upsample(C)
#
#        OC = model_channels * 2
#        self.upres7 = ResBlock(C + C_skip, time_embed_dim, OC)
#        C = OC
#        self.upres8 = ResBlock(C + C_skip, time_embed_dim, OC)
#        C_skip = model_channels
#        self.upres9 = ResBlock(C + C_skip, time_embed_dim, OC)
#        self.up3 = Upsample(C)
#
#        OC = model_channels
#        self.upres10 = ResBlock(C + C_skip, time_embed_dim, OC)
#        C = OC
#        self.upres11 = ResBlock(C + C_skip, time_embed_dim, OC)
#        self.upres12 = ResBlock(C + C_skip, time_embed_dim, OC)
#
#        self.gn = nn.GroupNorm(32, C)
#        self.silu = nn.SiLU()
#        self.out_conv = nn.Conv2d(C, 3, 3, padding=1)
#    
#    def forward(self, x, time_emb, debug=False):
#        emb = self.time_lin1(time_emb)
#        emb = self.time_silu(emb)
#        emb = self.time_lin2(emb)
#
#        h_skips = []
#        h = self.init_conv(x)
#        h_skips.append(h)
#        h = self.res1(h, emb)
#        h_skips.append(h)
#        h = self.res2(h, emb)
#        h_skips.append(h)
#
#        h = self.down1(h)
#        h_skips.append(h)
#        h = self.res3(h, emb)
#        h_skips.append(h)
#        h = self.res4(h, emb)
#        h_skips.append(h)
#
#        h = self.down2(h)
#        h_skips.append(h)
#        h = self.res5(h, emb)
#        h = self.att1(h)
#        h_skips.append(h)
#        h = self.res6(h, emb)
#        h = self.att2(h)
#        h_skips.append(h)
#
#        h = self.down3(h)
#        h_skips.append(h)
#        h = self.res7(h, emb)
#        h = self.att3(h)
#        h_skips.append(h)
#        h = self.res8(h, emb)
#        h = self.att4(h)
#        h_skips.append(h)
#
#        h = self.res9(h, emb)
#        h = self.att5(h)
#        h = self.res10(h, emb)
#
#        hs = h_skips.pop()
#        h = torch.cat([h, hs], dim=1)
#        h = self.upres1(h, emb)
#        h = self.upatt1(h)
#        hs = h_skips.pop()
#        h = torch.cat([h, hs], dim=1)
#        h = self.upres2(h, emb)
#        h = self.upatt2(h)
#        hs = h_skips.pop()
#        h = torch.cat([h, hs], dim=1)
#        h = self.upres3(h, emb)
#        h = self.upatt3(h)
#        h = self.up1(h)
#
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres4(h, emb)
#        h = self.upatt4(h)
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres5(h, emb)
#        h = self.upatt5(h)
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres6(h, emb)
#        h = self.upatt6(h)
#        h = self.up2(h)
#
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres7(h, emb)
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres8(h, emb)
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres9(h, emb)
#        h = self.up3(h)
#        
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres10(h, emb)
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres11(h, emb)
#        h = torch.cat([h, h_skips.pop()], dim=1)
#        h = self.upres12(h, emb)
#
#        h = self.gn(h)
#        h = self.silu(h)
#        h = self.out_conv(h)
#        return h

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

if __name__ == "__main__":
    B = 32
    C_in = 3
    C_model = 64
    C_out = 3
    H = 64
    W = 64
    max_period = 1000
    device = "cuda:1"

    # generate random timesteps in a way that is reproducible on cuda
    # has to be done before we initialize model weights
    #timesteps = torch.randint(0, max_period, (B, 1), device=device).float()
    all_timesteps = torch.rand(B * 10)
    all_timesteps = (all_timesteps * max_period).long().float().to(device)

    # generate random noise to create actual x_t targets
    # generate before model init to reproduce in cuda
    all_noise = torch.randn(10 * B * C_in * H * W).view(10, B, C_in, H, W).to(device)

    betas = get_named_beta_schedule("linear", 1000)
    diffusion = GaussianDiffusion(betas)

    
    #model = UNetTestModel(C_in, C_model)
    # by now our unet model should match the actual model, so use that instead
    model = UNetModel(C_in, C_model, C_out, 2, (4, 8), num_head_channels=32)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    
    #dout = torch.randn_like(out) * (H * W)**-0.5 / B
    ## dout = (torch.arange(B * C_model * 4).view(B, C_model*4) * 0.1)
    #fakeloss = (out * dout).sum()
    #fakeloss.backward()
    save_model_params_to_bin(
        model,
        "unet_test_params.bin",
        B=B,
        C_in=C_in,
        C_model=C_model,
        C_out=C_out,
        H=H,
        W=W,
        max_period=max_period,
    )

    compile_model = False
    if compile_model:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print("compiling the model...")
        model = torch.compile(model)
    # random inputs
    #x = (torch.randn(B, C_in, H, W, device=device) * (H * W)**-0.5).requires_grad_(True)
    #emb = (torch.randn(B, C_model, device=device) * (C_model)**-0.5).requires_grad_(True)

    # load training data batches by hand
    data_file = "../datasets/elephant_train.bin"
    imgs_np = load_img_from_bin(data_file)
    inputs = torch.tensor(imgs_np, device=device)
    batch_idx = 0
    x = inputs[:B].clone().requires_grad_(True)
    x.retain_grad()
    batch_idx += B

    #emb = timestep_embedding(timesteps, model.model_channels, max_period)
    # emb = (torch.arange(B * C_model).view(B, C_model) * 0.1).requires_grad_(True)
    losses = []

    #out = model(x, emb)
    timesteps = all_timesteps[:B].view(B, 1)
    out = model(x, timesteps)
    fake_target = torch.randn_like(out, device=device)
    fake_mse = ((out - fake_target) ** 2).mean()
    noise = all_noise[0]
    target = diffusion.q_sample(x, timesteps, noise)
    mse = ((out - target) ** 2).mean()
    losses.append(mse)
    mse.backward()
    #losses.append(fake_mse)
    #fake_mse.backward()

    # save acts and grads for debugging
    with open("unet_test_acts.bin", "wb") as f:
        write(x, f)
        write(timesteps, f)
        write(out, f)
        #write(dout, f)
        #write(emb.grad, f)
        write(x.grad, f)
        #write(fake_target, f)
        write(target, f)
        
        for name, param in model.named_parameters():
            write(param.grad, f)
    opt.step()
    opt.zero_grad()


    # calculate a new loss
    #mse2 = ((model(x, emb) - fake_target) ** 2).mean()
    for i in range(1, 10):
        #mse_new = ((model(x, emb) - fake_target) ** 2).mean()
        timesteps = all_timesteps[i*B:(i+1)*B].view(B, 1)
        x = inputs[i*B:(i+1)*B]
        noise = all_noise[i]
        target = diffusion.q_sample(x, timesteps, noise)
        mse_new = ((model(x, timesteps) - target) ** 2).mean()
        #mse_new = ((model(x, timesteps) - fake_target) ** 2).mean()
        losses.append(mse_new)
        mse_new.backward()
        opt.step()
        opt.zero_grad()
    
    # save the losses
    with open("unet_test_losses.bin", "wb") as f:
        for loss in losses:
            write(loss, f)

    print("benchmarking")
    repeats = 100
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    forward_time = 0.0
    backward_time = 0.0

    torch.cuda.synchronize()
    forward_start.record()
    for i in tqdm(range(repeats)):
        out = model(x, timesteps)    
        mse_new = ((out - target) ** 2).mean()
    torch.cuda.synchronize()
    forward_end.record()
    torch.cuda.synchronize()
    forward_time = forward_start.elapsed_time(forward_end)

    torch.cuda.synchronize()
    backward_start.record()
    for i in tqdm(range(repeats)):
        out = model(x, timesteps)    
        mse_new = ((out - target) ** 2).mean()
        mse_new.backward()
        opt.zero_grad()
    torch.cuda.synchronize()
    backward_end.record()
    torch.cuda.synchronize()
    backward_time = backward_start.elapsed_time(backward_end)
    backward_time = backward_time - forward_time

    print(f"forward avg time in ms: {forward_time/repeats}")
    print(f"backward avg time in ms: {backward_time/repeats}")
    
    print("\nDone\n")