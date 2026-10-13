# this module is based on the Wan Project: https://github.com/Wan-Video/Wan2.2 and self-forcing: https://github.com/guandeh17/Self-Forcing/blob/main/wan/modules/causal_model.py
import os
import torch
from torch import nn
from src.model.utils import RMSNorm, rope_apply_1d
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
from src.model.attention import attention
import math

# flex_attention = torch.compile(flex_attention, dynamic=True, mode="reduce-overhead")

# flex_attention = torch.compile(
#     flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
# )


def prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)
            from torch.nn.attention.flex_attention import create_mask
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            dense_mask = create_mask(
                attention_mask,
                B=None,
                H=None,
                Q_LEN=total_length + padded_length,
                KV_LEN=total_length + padded_length,
                device=device,
            )[0, 0, :total_length, :total_length].to(torch.float32).cpu()

            debug_dir = os.path.join(os.getcwd(), "tmp_debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_stem = (
                f"causal_mask_frames{num_frames}"
                f"_seqlen{frame_seqlen}"
                f"_block{num_frame_per_block}"
                f"_local{local_attn_size}"
            )
            pt_path = os.path.join(debug_dir, f"{debug_stem}.pt")
            png_path = os.path.join(debug_dir, f"{debug_stem}.png")
            torch.save(dense_mask, pt_path)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(dense_mask.numpy(), cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_title("Temporal Causal Mask")
            ax.set_xlabel("kv_idx")
            ax.set_ylabel("q_idx")
            fig.tight_layout()
            fig.savefig(png_path, dpi=200)
            plt.close(fig)
            print(f" saved dense causal mask tensor to {pt_path}")
            print(f" saved dense causal mask image to {png_path}")

        return block_mask


def causal_rope_apply_1d(x, freqs, start_frame=0):
    """
    x:     (B, N, D)
    freqs: (max_N, D/2)
    """
    B, N, D = x.shape
    assert D % 2 == 0
    assert start_frame + N <= freqs.size(0)

    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(B, N, D // 2, 2)
    )
    freqs = freqs[start_frame:start_frame + N].unsqueeze(0)
    x_out = torch.view_as_real(x_complex * freqs).reshape(B, N, D)

    return x_out.type_as(x)


class CausalDyneTrionSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = (
            21 * 1560 if local_attn_size == -1 else local_attn_size * 1560
        )

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads * C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask): chunk attention mask
            current_start_frame (int): indicator of current frame id (frame)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert seq_lens.shape[0] == b

        if cache_start is None:
            cache_start = current_start
        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)

        if kv_cache is None:
            if False: #is_tf:
                pass
            else:
                roped_query = rope_apply_1d(q, freqs=freqs).type_as(v)
                roped_key = rope_apply_1d(k, freqs=freqs).type_as(v)
                if block_mask is None: #NOTE
                    x = attention(roped_query, roped_key, v)
                else:
                    # we need to pad seq to 128x
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]

                    padded_roped_query = torch.cat([
                        roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device, dtype=v.dtype)]
                        , dim = 1
                    )

                    padded_roped_key = torch.cat([
                        roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                        device=k.device, dtype=v.dtype)]
                        , dim = 1
                    )

                    padded_v = torch.cat([
                        v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device, dtype=v.dtype)]
                        , dim = 1
                    )

                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask,
                    )[:, :, :-padded_length].transpose(2, 1)

        else: # for inference
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply_1d(
                q, freqs=freqs, start_frame=current_start_frame
            ).type_as(v)
            roped_key = causal_rope_apply_1d(
                k, freqs=freqs, start_frame=current_start_frame
            ).type_as(v)
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if (
                self.local_attn_size != -1 # local attention or not
                and (current_end > kv_cache["global_end_index"].item()) # if we generate new tokens
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size) # if the cache overflows
            ):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens: sink_tokens+num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens+num_evicted_tokens:sink_tokens+num_evicted_tokens+num_rolled_tokens]
                kv_cache["k"][:, sink_tokens: sink_tokens+num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens+num_evicted_tokens:sink_tokens+num_evicted_tokens+num_rolled_tokens]
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_idx"] + \
                    current_end - kv_cache-["global_end_idx"] - \
                    num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache['k'][:, local_start_index: local_end_index] = roped_key
                kv_cache['v'][:, local_start_index: local_end_index] = v
            else:
                local_end_index = kv_cache["local_end_idx"] + \
                    current_end - kv_cache-["global_end_idx"]
                local_start_index = local_end_index - num_new_tokens
                kv_cache['k'][:, local_start_index: local_end_index] = roped_key
                kv_cache['v'][:, local_start_index: local_end_index] = v
            x = attention(
                roped_query,
                kv_cache['k'][:, max(0, local_end_index-self.max_attention_size): local_end_index],
                kv_cache['v'][:, max(0, local_end_index-self.max_attention_size): local_end_index],
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
