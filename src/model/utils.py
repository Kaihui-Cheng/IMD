import torch
import math
import torch.nn.functional as F
from torch import nn
#def get_index_embedding(indices, embed_size, max_len=512):
def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_frame_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    timesteps (eg: [B], [B, T], [B, T, N])
    """
    orig_shape = timesteps.shape
    timesteps = timesteps.reshape(-1)
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb.reshape(*orig_shape, embedding_dim)

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply_1d(x, freqs):
    """
    x:     (B, S, H, D)
    freqs: (max_S, D/2)
    This code need to assert the sequence is not padded
    """
    B, S, H, D = x.shape
    assert D % 2 == 0
    assert S <= freqs.size(0)

    # 转为 complex
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(B, S, H, D // 2, 2)
    )  # (B, S, H, D/2)
    # 取对应长度的 freqs，并对 B、H 广播
    freqs_s = freqs[:S].view(1, S, 1, -1)  # (1, S, 1, D/2)
    # 应用 RoPE
    x_rot = x_complex * freqs_s
    # 转回 real
    x_out = torch.view_as_real(x_rot).reshape(B, S, H, D)
    return x_out.type_as(x)


def rope_apply_1d_with_lengths(x, freqs, lengths):
    B, S, H, D = x.shape
    x_out = x.clone()

    for i in range(B):
        L = lengths[i]
        x_out[i, :L] = rope_apply_1d(x[i : i + 1, :L], freqs)[0]
        # x_out[i, L:] 不动

    return x_out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def gpu_mem(msg=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"[GPU MEM] {msg:30s} | alloc={alloc:8.1f} MB | "
        f"reserved={reserved:8.1f} MB | peak={peak:8.1f} MB"
    )
