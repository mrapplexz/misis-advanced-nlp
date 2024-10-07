import torch


def build_square_attention_mask(attention_mask: torch.Tensor, dtype, device, num_heads: int) -> torch.Tensor:
    # batch independent
    # 1: q+ k+
    # 0: q- k-
    # 1 0 1 0 0
    #
    # 1 0 1 0 0
    # 0 1 0 0 0
    # 1 0 1 0 0
    # 0 0 0 1 0
    # 0 0 0 0 1

    # 1 1 1 0 0

    # 1 1 1 0 0
    # 1 1 1 0 0
    # 1 1 1 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    attention_mask = attention_mask.to(dtype)
    diag_matrix = torch.diag_embed(torch.ones_like(attention_mask, dtype=dtype, device=device))
    for batch_i in range(attention_mask.shape[0]):
        att_mask_value = attention_mask[batch_i]
        diag_matrix[batch_i, att_mask_value.bool()] = att_mask_value
    diag_matrix[diag_matrix == 0] = torch.finfo(dtype).min
    diag_matrix = diag_matrix.repeat_interleave(num_heads, 0)
    return diag_matrix
