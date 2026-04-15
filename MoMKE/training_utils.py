import torch


def compute_monitor_value(dataset, metric_mode, f1, acc, corr, mae):
    mode = metric_mode
    if mode == "auto":
        mode = "f1_acc" if dataset in ["CMUMOSI", "CMUMOSEI"] else "acc"

    if mode == "f1":
        return float(f1)
    if mode == "acc":
        return float(acc)
    if mode == "f1_acc":
        return float((f1 + acc) / 2.0)
    if mode == "corr":
        return float(corr)
    if mode == "neg_mae":
        return float(-mae)

    raise ValueError(f"Unsupported monitor metric: {metric_mode}")


def resolve_cpu_thread_settings(cpu_threads, cpu_interop_threads):
    resolved_cpu_threads = 1 if not cpu_threads or cpu_threads <= 0 else int(cpu_threads)
    resolved_cpu_interop_threads = 1 if not cpu_interop_threads or cpu_interop_threads <= 0 else int(cpu_interop_threads)
    return resolved_cpu_threads, resolved_cpu_interop_threads


def apply_stage2_modality_dropout(
    base_mask,
    umask,
    train,
    first_stage,
    enable_dropout,
    audio_drop_prob,
    text_drop_prob,
    visual_drop_prob,
    generator=None,
):
    mask = base_mask.clone()
    valid_positions = umask.transpose(0, 1).unsqueeze(-1) > 0
    mask = mask * valid_positions.float()

    if (not train) or first_stage or (not enable_dropout):
        return mask

    drop_probs = torch.tensor(
        [audio_drop_prob, text_drop_prob, visual_drop_prob],
        device=mask.device,
        dtype=mask.dtype,
    ).view(1, 1, 3)

    if torch.max(drop_probs) <= 0:
        return mask

    random_tensor = torch.rand(mask.shape, device=mask.device)
    if generator is not None and mask.device.type == "cpu":
        random_tensor = torch.rand(mask.shape, generator=generator, device=mask.device)

    active = (mask > 0) & valid_positions
    keep = active & (random_tensor >= drop_probs)

    need_fix = valid_positions.squeeze(-1) & (active.sum(dim=-1) > 0) & (keep.sum(dim=-1) == 0)
    if need_fix.any():
        fix_indices = need_fix.nonzero(as_tuple=False)
        for seq_idx, batch_idx in fix_indices.tolist():
            available = active[seq_idx, batch_idx].nonzero(as_tuple=False).view(-1)
            if available.numel() == 0:
                continue
            if generator is not None and mask.device.type == "cpu":
                choice = torch.randint(available.numel(), (1,), generator=generator).item()
            else:
                choice = torch.randint(available.numel(), (1,), device=available.device).item()
            keep[seq_idx, batch_idx, available[choice]] = True

    return keep.float() * valid_positions.float()
