import torch
import math
from utils import write

torch.manual_seed(0)

def timestep_embedding(timesteps, dim, max_period=10000):
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
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding, freqs

B = 32
dim = 64
max_period = 1000

timesteps = torch.randint(0, max_period, (B, 1)).float()
embeddings, freqs = timestep_embedding(timesteps, dim, max_period)

with open("time_emb.bin", "wb") as f:
    write(timesteps, f)
    write(embeddings, f)
    write(freqs, f)

print("Done")