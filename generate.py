import argparse
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from train_unet import UNetModel, get_named_beta_schedule, GaussianDiffusion

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model_weights(model: nn.Module, filename: str):
    if filename.endswith(".pt") or filename.endswith(".pth"):
        state_dict = torch.load(filename)
        # might have some artifacts in statedict from torch.compile
        adjusted_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(adjusted_state_dict)
    elif filename.endswith(".bin"):
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            weights_np = np.frombuffer(f.read(), dtype=np.float32).copy().copy().copy().copy()
            assert header[0] == 12345678, "Bad magic number"
            
            offset = 0
            for name, param in model.named_parameters():
                numel = param.numel()
                param.data = torch.from_numpy(weights_np[offset:offset + numel]).view_as(param)
                offset += numel

def sample_next_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model: nn.Module,
    T: int,
    betas: torch.Tensor,
    alpha_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    sample x_{t-1} given x_t and x_0
    """
    assert 2 <= t and t < T, f"time index {t} must be in range [2, {T}]"
    beta_t = betas[t - 1]
    alpha_t = alpha_cumprod[t - 1]
    alpha_t_1 = alpha_cumprod[t - 2]

    # conditional mean and variance
    epsilon = model(x_t, t)
    mu_t = (x_t - (beta_t / torch.sqrt(1 - alpha_t)) * epsilon) / torch.sqrt(
        1 - beta_t
    )
    sigma_t = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t) * beta_t)

    return mu_t + sigma_t * torch.randn_like(mu_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="sample.jpg")
    args = parser.parse_args()
    
    C_in = 3
    C_model = 64
    C_out = 3
    max_period = 1000
    model = UNetModel(C_in, C_model, C_out, 2, (4, 8), num_head_channels=32)

    load_model_weights(model, args.model_filename)

    model.eval()
    model.to(device)

    betas = get_named_beta_schedule("linear", max_period)
    gaussian_diffusion = GaussianDiffusion(betas=betas)

    x_curr = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float32)
    beta_t = torch.tensor(betas, device=device, dtype=torch.float32)
    alpha_cumprod = torch.tensor(gaussian_diffusion.alphas_cumprod, device=device)
    with torch.no_grad():
        for t in tqdm(list(range(max_period - 1, 1, -1))):
            t_tensor = torch.tensor([[t]], device=device)
            x_curr = sample_next_step(x_curr, t_tensor, model, max_period, beta_t, alpha_cumprod)
        
    img_np = x_curr[0].cpu().permute(1, 2, 0).numpy()
    # unscale
    img_np = (img_np + 1) * 127.5
    img_np = img_np.astype(np.uint8)
    # save img_np as image
    img = Image.fromarray(img_np)
    img.save(args.output_filename)
    print(f"Saved sample to {args.output_filename}")