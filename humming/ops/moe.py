"""Routing helpers for Humming's MoE GEMM, in plain torch (no extension, CUDA-graph capturable).

``moe_align_block_size`` mirrors vLLM's op of the same name (same outputs, same buffer sizes) so
the indexed MoE GEMM can be driven without vLLM's compiled ``_C`` ops, e.g. on ROCm. Every op is
device-side with static shapes: no host synchronisation.
"""

import torch


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the routed rows ``topk_ids.flatten()`` by expert, each expert padded to ``block_size``.

    Returns ``(sorted_ids, expert_ids, num_tokens_padded)``:

    * ``sorted_ids`` int32 ``[max_num_tokens_padded]``: row ids ``token * top_k + slot`` grouped by
      expert; padding (and the unused tail) is ``topk_ids.numel()``.
    * ``expert_ids`` int32 ``[max_num_tokens_padded // block_size]``: the (local, if
      ``expert_map`` is given) expert of every block; -1 for unused blocks and non-local experts.
    * ``num_tokens_padded`` int32 ``[1]``: the used length of ``sorted_ids``.

    With ``expert_map`` (global id -> local id or -1), rows routed to non-local experts are dropped
    when ``ignore_invalid_experts`` is set, otherwise they keep blocks whose expert id is -1.
    """
    device = topk_ids.device
    numel = topk_ids.numel()
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    max_num_tokens_padded = max_num_blocks * block_size

    experts = topk_ids.reshape(-1).to(torch.int64)
    valid = (experts >= 0) & (experts < num_experts)
    if expert_map is not None and ignore_invalid_experts:
        valid &= expert_map[experts.clamp(0, num_experts - 1)] >= 0
    # Invalid rows go to a dummy bucket `num_experts` that gets no blocks.
    bucket = torch.where(valid, experts, num_experts)

    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    counts.scatter_add_(0, bucket, torch.ones_like(bucket))
    counts = counts[:num_experts]
    blocks = (counts + block_size - 1) // block_size
    block_end = blocks.cumsum(0)
    padded_start = (block_end - blocks) * block_size
    row_start = counts.cumsum(0) - counts

    order = torch.sort(bucket, stable=True).indices
    sorted_bucket = bucket[order]
    is_valid = sorted_bucket < num_experts
    safe_bucket = sorted_bucket.clamp(max=num_experts - 1)
    rank = torch.arange(numel, device=device) - row_start[safe_bucket]
    dest = torch.where(is_valid, padded_start[safe_bucket] + rank, max_num_tokens_padded)

    sorted_ids = torch.full((max_num_tokens_padded + 1,), numel, dtype=torch.int32, device=device)
    sorted_ids.scatter_(0, dest, order.to(torch.int32))
    sorted_ids = sorted_ids[:max_num_tokens_padded]

    block_index = torch.arange(max_num_blocks, device=device)
    block_expert = torch.searchsorted(block_end, block_index, right=True)
    used = block_expert < num_experts
    block_expert = block_expert.clamp(max=num_experts - 1)
    if expert_map is not None:
        block_expert = expert_map[block_expert].to(torch.int64)
    expert_ids = torch.where(used, block_expert, -1).to(torch.int32)
    num_tokens_padded = (block_end[-1:] * block_size).to(torch.int32)
    return sorted_ids, expert_ids, num_tokens_padded
