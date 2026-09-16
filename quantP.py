#!/usr/bin/env python3
"""GSM8K W4 benchmark: BF16, RTN-W4, SpinQuant-W4 and BOSESIR.

The benchmark is weight-only (W4A16). ``--rotation-path`` should point to the
``R.bin`` written by SpinQuant's ``optimize_rotation.py``. Without it, the
SpinQuant row uses deterministic Hadamard rotations (a QuaRot-style baseline),
which is useful for a self-contained smoke test but is not learned SpinQuant.

BOSESIR is the structured FFN-group condensation method from
weight_compression_bench_fixed.py; it is a pruning comparison, not a W4 row.
All rows use the same first 500 GSM8K test examples by default.
"""
import argparse
import gc
import json
import math
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n", type=int, default=500,
                   help="GSM8K test examples; default is the requested 500")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--groupsize", type=int, default=-1,
                   help="-1 means per-output-channel, as in SpinQuant")
    p.add_argument("--asym", action="store_true",
                   help="use asymmetric RTN W4 with stored zero-points")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--calib", type=int, default=16,
                   help="GSM8K train examples used by BOSESIR evidence")
    p.add_argument("--boseir-group-size", type=int, default=128)
    p.add_argument("--boseir-retain-frac", type=float, default=0.5)
    p.add_argument("--rotation-path", default=None,
                   help="SpinQuant R.bin; omit for Hadamard rotations")
    p.add_argument("--conditions", default="bf16,rtn_w4,spinquant_w4,bosesir")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--out", default="results/quant_bench.json")
    return p.parse_args()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def cfg(config, name):
    value = getattr(config, name, None)
    if value is not None:
        return value
    text_config = getattr(config, "text_config", None)
    return getattr(text_config, name, None) if text_config is not None else None


def layers_of(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("This script expects a model.model.layers decoder architecture.")
    return list(model.model.layers)


def get_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def projection_slots(model):
    """Return logical projection names and their owning object/attribute."""
    slots = []
    for i, layer in enumerate(layers_of(model)):
        for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                     "mlp.down_proj"):
            parent_path, attr = path.rsplit(".", 1)
            parent = get_path(layer, parent_path)
            child = getattr(parent, attr)
            if isinstance(child, _OnlineHadLinear):
                owner, owner_attr, linear = child, "linear", child.linear
            elif isinstance(child, nn.Linear):
                owner, owner_attr, linear = parent, attr, child
            else:
                raise RuntimeError(f"{path} in layer {i} is not a Linear module.")
            slots.append((f"model.layers.{i}.{path}", owner, owner_attr, linear))
    found = {name.rsplit(".", 1)[-1] for name, *_ in slots}
    if found != set(PROJECTIONS):
        raise RuntimeError(f"Projection coverage is incomplete: found {sorted(found)}")
    return slots


# ---------------------------------------------------------------------------
# Hadamard rotations
def hadamard(n):
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def fwht(x):
    shape = x.shape
    n = shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.cat(((a + b).unsqueeze(-2), (a - b).unsqueeze(-2)), -2).reshape(*shape)
        h *= 2
    return x / math.sqrt(n)


def orthogonal(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return (q * signs.unsqueeze(0)).float()


def hadamard_or_orthogonal(n, seed):
    if n & (n - 1) == 0:
        return hadamard(n)
    # Keep the QR small: n = k * m, where m is the largest power of two.
    m = n & -n
    k = n // m
    hk = hadamard(k) if k & (k - 1) == 0 else orthogonal(k, seed)
    # torch.kron may use view internally; make both operands contiguous for
    # compatibility with PyTorch versions that reject non-contiguous views.
    return torch.kron(hk.contiguous(), hadamard(m).contiguous())


def down_hadamard(n, seed):
    m = n & -n
    k = n // m
    hk = hadamard(k) if k & (k - 1) == 0 else orthogonal(k, seed + 17)
    return torch.kron(hk.contiguous(), hadamard(m).contiguous()), hk, m


def online_hadamard(x, hk, m):
    shape = x.shape
    k = hk.shape[0]
    y = fwht(x.reshape(*shape[:-1], k, m))
    return torch.matmul(hk, y).reshape(*shape)


def block_output_rotation(weight, r):
    """For y=xW^T, return W' with y'=y*blockdiag(r)."""
    blocks, head_dim = weight.shape[0] // r.shape[0], r.shape[0]
    x = weight.reshape(blocks, head_dim, weight.shape[1])
    return torch.matmul(r.t(), x).reshape_as(weight)


def block_input_rotation(weight, r):
    """Return W' = W*blockdiag(r) without materializing the block matrix."""
    blocks, head_dim = weight.shape[1] // r.shape[0], r.shape[0]
    x = weight.reshape(weight.shape[0], blocks, head_dim)
    return torch.matmul(x, r).reshape_as(weight)


def load_rotations(path, hidden_size, head_dim, n_layers, seed, device):
    """Load SpinQuant's R1/R2 and convert R1 to this script's row-vector form."""
    if path is None:
        r1 = hadamard_or_orthogonal(hidden_size, seed)
        r2 = [hadamard_or_orthogonal(head_dim, seed + i + 1) for i in range(n_layers)]
        return r1.t().to(device), [x.to(device) for x in r2], "hadamard"
    if os.path.isdir(path):
        path = os.path.join(path, "R.bin")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if "R1" not in state:
        raise RuntimeError(f"{path} does not contain SpinQuant key 'R1'.")
    r1 = state["R1"].float()
    if tuple(r1.shape) != (hidden_size, hidden_size):
        raise RuntimeError(f"R1 has shape {tuple(r1.shape)}, expected {(hidden_size, hidden_size)}")
    r2 = []
    for i in range(n_layers):
        key = f"model.layers.{i}.self_attn.R2"
        value = state.get(key)
        if value is None:
            print(f"warning: missing {key} in {path}; using Hadamard R2", flush=True)
            value = hadamard_or_orthogonal(head_dim, seed + i + 1)
        if tuple(value.shape) != (head_dim, head_dim):
            raise RuntimeError(f"{key} has shape {tuple(value.shape)}, expected {(head_dim, head_dim)}")
        r2.append(value.float())
    # SpinQuant applies x @ R1 at runtime; the baked model below uses Q=R1.T.
    return r1.t().to(device), [x.to(device) for x in r2], path


class _Rotate(nn.Module):
    def __init__(self, inner, rotation, pre=False):
        super().__init__()
        self.inner = inner
        self.pre = pre
        self.register_buffer("rotation", rotation.float())

    def forward(self, x):
        if self.pre:
            return self.inner((x.float() @ self.rotation).to(x.dtype))
        y = self.inner(x)
        return (y.float() @ self.rotation).to(y.dtype)

    @property
    def weight(self):
        return self.inner.weight


class _OnlineHadLinear(nn.Module):
    def __init__(self, linear, hk, m):
        super().__init__()
        self.linear = linear
        self.register_buffer("had_k", hk.float())
        self.m = m

    def forward(self, x):
        return self.linear(online_hadamard(x.float(), self.had_k, self.m).to(x.dtype))


def fuse_norms(model):
    """Fuse RMSNorm/LayerNorm affine parameters exactly into following linears."""
    for layer in layers_of(model):
        for norm_name, paths in (
            ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ):
            norm = getattr(layer, norm_name, None)
            if norm is None or not hasattr(norm, "weight"):
                raise RuntimeError(f"Missing {norm_name}; cannot apply SpinQuant safely.")
            gamma = norm.weight.detach().float()
            beta = getattr(norm, "bias", None)
            beta = beta.detach().float() if beta is not None else None
            for path in paths:
                linear = get_path(layer, path)
                if not isinstance(linear, nn.Linear):
                    raise RuntimeError(f"{path} is not a plain Linear before rotation.")
                old_w = linear.weight.detach().float()
                linear.weight.data.copy_((old_w * gamma).to(linear.weight.dtype))
                if beta is not None:
                    if linear.bias is None:
                        linear.bias = nn.Parameter(torch.zeros(
                            linear.out_features, device=linear.weight.device,
                            dtype=linear.weight.dtype))
                    linear.bias.data.add_((old_w @ beta).to(linear.bias.dtype))
            norm.weight.data.fill_(1)
            if getattr(norm, "bias", None) is not None:
                norm.bias.data.zero_()


def apply_spinquant(model, seed, rotation_path=None):
    """Bake SpinQuant R1/R2 and R4 into a normal HF model."""
    fuse_norms(model)
    mm = model.model
    hidden = cfg(model.config, "hidden_size")
    intermediate = cfg(model.config, "intermediate_size")
    heads = cfg(model.config, "num_attention_heads")
    kv_heads = cfg(model.config, "num_key_value_heads") or heads
    head_dim = cfg(model.config, "head_dim") or (hidden // heads)
    device = next(model.parameters()).device
    q, r2s, rotation_name = load_rotations(
        rotation_path, hidden, head_dim, len(mm.layers), seed, device)
    down_f, hk, m = down_hadamard(intermediate, seed)
    down_f, hk = down_f.to(device), hk.to(device)

    with torch.no_grad():
        mm.embed_tokens = _Rotate(mm.embed_tokens, q.t())
        mm.norm = _Rotate(mm.norm, q, pre=True)
        for i, layer in enumerate(mm.layers):
            for path in ("self_attn.q_proj", "self_attn.k_proj",
                         "mlp.gate_proj", "mlp.up_proj"):
                linear = get_path(layer, path)
                linear.weight.data.copy_((linear.weight.float() @ q.t()).to(linear.weight.dtype))

            v = layer.self_attn.v_proj
            o = layer.self_attn.o_proj
            r2 = r2s[i]
            v.weight.data.copy_(block_output_rotation(
                v.weight.float() @ q.t(), r2).to(v.weight.dtype))
            if v.bias is not None:
                v.bias.data.copy_((v.bias.float().reshape(kv_heads, head_dim) @ r2)
                                  .reshape(-1).to(v.bias.dtype))

            o_weight = q @ o.weight.float()
            o.weight.data.copy_(block_input_rotation(o_weight, r2).to(o.weight.dtype))
            if o.bias is not None:
                o.bias.data.copy_((q @ o.bias.float()).to(o.bias.dtype))

            down = layer.mlp.down_proj
            down.weight.data.copy_((q @ down.weight.float() @ down_f.t()).to(down.weight.dtype))
            if down.bias is not None:
                down.bias.data.copy_((q @ down.bias.float()).to(down.bias.dtype))
            layer.mlp.down_proj = _OnlineHadLinear(down, hk, m)
    return rotation_name


# ---------------------------------------------------------------------------
# Real packed W4 storage
def pack_bits(codes, bits):
    codes = codes.to(torch.int64).flatten()
    n = codes.numel()
    packed = torch.zeros((n * bits + 7) // 8, dtype=torch.int64, device=codes.device)
    start = torch.arange(n, device=codes.device, dtype=torch.int64) * bits
    for bit in range(bits):
        positions = start + bit
        ones = ((codes >> bit) & 1).bool()
        if ones.any():
            packed.index_put_((positions[ones] // 8,),
                              torch.ones_like(positions[ones]) << (positions[ones] % 8),
                              accumulate=True)
    return packed.to(torch.uint8)


def unpack_bits(packed, n, bits):
    out = torch.zeros(n, dtype=torch.int64, device=packed.device)
    start = torch.arange(n, device=packed.device, dtype=torch.int64) * bits
    for bit in range(bits):
        positions = start + bit
        out |= (((packed[positions // 8].long() >> (positions % 8)) & 1) << bit)
    return out


def quantize_weight(weight, bits, groupsize, symmetric):
    out_features, in_features = weight.shape
    group = in_features if groupsize == -1 else groupsize
    if group <= 0 or in_features % group:
        raise ValueError(f"groupsize={groupsize} does not divide in_features={in_features}")
    x = weight.detach().float().reshape(out_features, in_features // group, group)
    qmax = 2 ** bits - 1
    if symmetric:
        maxq = 2 ** (bits - 1) - 1
        offset = 2 ** (bits - 1)
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / maxq
        codes = (torch.round(x / scale).clamp(-offset, maxq) + offset).long()
        zero = None
    else:
        lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        constant = (hi - lo) < 1e-8
        lo = torch.where(constant, lo - 0.5, lo)
        hi = torch.where(constant, hi + 0.5, hi)
        scale = (hi - lo) / qmax
        zero = torch.round(-lo / scale)
        codes = torch.round(x / scale + zero).clamp(0, qmax).long()
        offset = 0
    return codes.flatten(), scale.squeeze(-1).to(torch.float16), (
        None if zero is None else zero.squeeze(-1).to(torch.float16)), offset


class PackedLinear(nn.Module):
    """Packed-at-rest linear with a lazy one-time dequantization cache."""
    def __init__(self, linear, bits, groupsize, symmetric):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.group = self.in_features if groupsize == -1 else groupsize
        self.symmetric = symmetric
        self.compute_dtype = linear.weight.dtype
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None
        if bits == 0:
            self.packed = self.scale = self.zero = None
            self.weight_fp = None
        elif bits >= 16:
            self.register_buffer("weight_fp", linear.weight.detach().clone())
            self.packed = self.scale = self.zero = None
        else:
            codes, scale, zero, offset = quantize_weight(
                linear.weight, bits, groupsize, symmetric)
            self.register_buffer("packed", pack_bits(codes, bits))
            self.register_buffer("scale", scale)
            if zero is None:
                self.zero = None
            else:
                self.register_buffer("zero", zero)
            self.weight_fp = None
            self.q_offset = offset
        # Materialized weights are deliberately not registered: they are a
        # runtime cache, not part of the packed model's stored footprint.
        self._materialized = None

    def stored_bytes(self):
        if self.bits == 0:
            total = 0
        elif self.bits >= 16:
            total = self.weight_fp.numel() * self.weight_fp.element_size()
        else:
            total = self.packed.numel() + self.scale.numel() * 2
            total += 0 if self.zero is None else self.zero.numel() * 2
        return total + (0 if self.bias is None else self.bias.numel() * self.bias.element_size())

    def _materialize(self):
        if self._materialized is not None:
            return self._materialized
        if self.bits >= 16:
            weight = self.weight_fp
        else:
            q = unpack_bits(self.packed, self.out_features * self.in_features, self.bits)
            q = q.reshape(self.out_features, self.in_features // self.group, self.group).float()
            if self.symmetric:
                q = (q - self.q_offset) * self.scale[..., None].float()
            else:
                q = (q - self.zero[..., None].float()) * self.scale[..., None].float()
            weight = q.reshape(self.out_features, self.in_features).to(self.compute_dtype)
        self._materialized = weight
        return weight

    def forward(self, x):
        if self.bits == 0:
            shape = (*x.shape[:-1], self.out_features)
            if self.bias is None:
                return torch.zeros(shape, device=x.device, dtype=x.dtype)
            return self.bias.expand(*shape).to(x.dtype)
        return F.linear(x, self._materialize(), self.bias)


def replace_projections(model, bits, groupsize, symmetric, assignment=None):
    for name, owner, attr, linear in projection_slots(model):
        b = bits if assignment is None else assignment.get(name, 16)
        setattr(owner, attr, PackedLinear(linear, b, groupsize, symmetric))


def model_storage_bytes(model):
    total, seen = 0, set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if id(tensor) not in seen:
            total += tensor.numel() * tensor.element_size()
            seen.add(id(tensor))
    return total


def self_test():
    generator = torch.Generator().manual_seed(0)
    for bits in range(1, 9):
        n = 513
        codes = torch.randint(0, 2 ** bits, (n,), generator=generator)
        assert torch.equal(unpack_bits(pack_bits(codes, bits), n, bits), codes)
    x = torch.randn(2, 3, 8, generator=generator)
    assert torch.allclose(fwht(x), x @ hadamard(8).t(), atol=1e-5)
    x = torch.randn(2, 3, 12, generator=generator)
    hk, m = down_hadamard(12, 0)[1:]
    explicit = torch.kron(hk.contiguous(), hadamard(m).contiguous())
    assert torch.allclose(online_hadamard(x, hk, m), x @ explicit.t(), atol=1e-5)


# ---------------------------------------------------------------------------
# BOSESIR structured FFN condensation
def calibration_loss(model, tokenizer, text, max_len=512):
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(next(model.parameters()).device)
        mask = enc["attention_mask"].to(ids.device)
        labels = ids.masked_fill(mask == 0, -100)
        return model(input_ids=ids, attention_mask=mask, labels=labels,
                     use_cache=False).loss
    finally:
        tokenizer.padding_side = old_side


def boseir_evidence(model, tokenizer, texts, group_size):
    layers = layers_of(model)
    refs, gates, utility, handles = [], {}, {}, []
    old_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for i, layer in enumerate(layers):
            gate_proj, up_proj, down_proj = layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
            if not all(isinstance(x, nn.Linear) for x in (gate_proj, up_proj, down_proj)):
                raise RuntimeError("BOSESIR requires ordinary FFN Linear projections.")
            if gate_proj.out_features != up_proj.out_features or gate_proj.out_features != down_proj.in_features:
                raise RuntimeError(f"Layer {i} has inconsistent FFN dimensions.")
            if gate_proj.out_features % group_size:
                raise RuntimeError(f"FFN size is not divisible by --boseir-group-size={group_size}.")
            n_groups = gate_proj.out_features // group_size
            refs.append((layer, group_size, n_groups))
            utility[i] = torch.zeros(n_groups, dtype=torch.float64)

            def hook(_module, inputs, index=i):
                gate = gates[index]
                expanded = gate.repeat_interleave(group_size)
                return (inputs[0].float() * expanded).to(inputs[0].dtype), *inputs[1:]
            handles.append(down_proj.register_forward_pre_hook(hook))

        model.eval()
        used = 0
        for text in texts:
            device = next(model.parameters()).device
            gates = {i: torch.ones(n, device=device, dtype=torch.float32, requires_grad=True)
                     for i, (_layer, _g, n) in enumerate(refs)}
            model.zero_grad(set_to_none=True)
            loss = calibration_loss(model, tokenizer, text)
            loss.backward()
            signals = []
            for i in utility:
                grad = gates[i].grad
                if grad is None:
                    raise RuntimeError(f"BOSESIR gate gradient missing at layer {i}.")
                score = (-gates[i].detach() * grad.detach()).clamp_min(0).double().cpu()
                utility[i] += score
                signals.append(bool(torch.any(score != 0)))
            if not any(signals):
                raise RuntimeError("BOSESIR produced all-zero gate evidence.")
            used += 1
    finally:
        for handle in handles:
            handle.remove()
        for p, state in zip(model.parameters(), old_requires_grad):
            p.requires_grad_(state)
        model.zero_grad(set_to_none=True)
    if not used:
        raise RuntimeError("No BOSESIR calibration example was usable.")
    return refs, {i: value / used for i, value in utility.items()}


def boseir_allocate(utility, retain_frac):
    capacities = {i: int(v.numel()) for i, v in utility.items()}
    total = sum(capacities.values())
    target = max(len(capacities), min(total, int(round(total * retain_frac))))
    layer_score = {i: float(v.mean()) for i, v in utility.items()}
    values = torch.tensor(list(layer_score.values()), dtype=torch.float64)
    energy = {i: -(s - float(values.mean())) / math.sqrt(float(values.var(correction=0)) + 1e-8)
              for i, s in layer_score.items()}

    def occupation(mu):
        result = {}
        for i, e in energy.items():
            z = max(min(e - mu, 50.0), -50.0)
            result[i] = min(float(capacities[i]), 1.0 / (math.exp(z) - 1.0 + 1e-8))
            result[i] = max(1.0, result[i])
        return result

    lo, hi = min(energy.values()) - 50.0, min(energy.values()) - 1e-8
    for _ in range(80):
        mid = (lo + hi) / 2
        if sum(occupation(mid).values()) < target:
            lo = mid
        else:
            hi = mid
    continuous = occupation((lo + hi) / 2)
    quotas = {i: min(capacities[i], max(1, int(math.floor(v))))
              for i, v in continuous.items()}
    remaining = target - sum(quotas.values())
    order = sorted(quotas, key=lambda i: continuous[i] - math.floor(continuous[i]), reverse=True)
    while remaining > 0:
        changed = False
        for i in order:
            if quotas[i] < capacities[i] and remaining:
                quotas[i] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    while remaining < 0:
        changed = False
        for i in reversed(order):
            if quotas[i] > 1 and remaining < 0:
                quotas[i] -= 1
                remaining += 1
                changed = True
        if not changed:
            break
    if sum(quotas.values()) != target:
        raise RuntimeError("BOSESIR quota rounding did not preserve the requested budget.")
    return quotas, energy, target


def slice_ffn(model, refs, utility, quotas):
    selected = {}
    for i, (layer, group_size, _n_groups) in enumerate(refs):
        selected[i] = torch.topk(utility[i], quotas[i], largest=True).indices.sort().values
        channels = (selected[i][:, None] * group_size +
                    torch.arange(group_size)[None, :]).reshape(-1)
        channels = channels.to(layer.mlp.gate_proj.weight.device)
        gate, up, down = layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
        new_gate = nn.Linear(gate.in_features, channels.numel(), gate.bias is not None,
                             device=gate.weight.device, dtype=gate.weight.dtype)
        new_up = nn.Linear(up.in_features, channels.numel(), up.bias is not None,
                           device=up.weight.device, dtype=up.weight.dtype)
        new_down = nn.Linear(channels.numel(), down.out_features, down.bias is not None,
                             device=down.weight.device, dtype=down.weight.dtype)
        with torch.no_grad():
            new_gate.weight.copy_(gate.weight.index_select(0, channels))
            new_up.weight.copy_(up.weight.index_select(0, channels))
            new_down.weight.copy_(down.weight.index_select(1, channels.to(down.weight.device)))
            if gate.bias is not None:
                new_gate.bias.copy_(gate.bias.index_select(0, channels))
            if up.bias is not None:
                new_up.bias.copy_(up.bias.index_select(0, channels))
            if down.bias is not None:
                new_down.bias.copy_(down.bias)
        layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj = new_gate, new_up, new_down
    return selected


def run_boseir(model, tokenizer, texts, group_size, retain_frac):
    refs, utility = boseir_evidence(model, tokenizer, texts, group_size)
    quotas, energy, target = boseir_allocate(utility, retain_frac)
    selected = slice_ffn(model, refs, utility, quotas)
    total = sum(n for _layer, _g, n in refs)
    return {"method": "BOSESIR", "group_size": group_size,
            "retained_groups": sum(quotas.values()), "total_groups": total,
            "retained_fraction": sum(quotas.values()) / total,
            "target_groups": target, "energy_by_layer": energy,
            "groups_by_layer": {str(i): int(v.numel()) for i, v in selected.items()}}


# ---------------------------------------------------------------------------
# GSM8K and measurements
def answer_number(text):
    text = text.replace(",", "").replace("$", "")
    if "####" in text:
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def gsm8k_data(n, shots, calib_n):
    ds = load_dataset("openai/gsm8k", "main")
    clean = lambda s: re.sub(r"<<[^>]*>>", "", s).strip()
    shot_text = "".join(
        f"Question: {x['question']}\nAnswer: {clean(x['answer'])}\n\n"
        for x in ds["train"].select(range(min(shots, len(ds["train"]))))
    )
    test = [(x["question"], answer_number(x["answer"]))
            for x in ds["test"].select(range(min(n, len(ds["test"]))))]
    calib = [f"Question: {x['question']}\nAnswer: {clean(x['answer'])}"
             for x in ds["train"].select(range(min(calib_n, len(ds["train"]))))]
    return shot_text, test, calib


def gsm8k_eval(model, tokenizer, data, shot_text, args, tag):
    old_pad, old_trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    correct = 0
    try:
        for start in range(0, len(data), args.batch_size):
            batch = data[start:start + args.batch_size]
            prompts = [shot_text + f"Question: {q}\nAnswer:" for q, _ in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_input_tokens)
            device = next(model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                output = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                        do_sample=False, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.batch_decode(
                output[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for text, (_, gold) in zip(generated, batch):
                pred = answer_number(text)
                correct += int(pred is not None and gold is not None and abs(pred - gold) < 1e-4)
            done = min(start + args.batch_size, len(data))
            if done == len(data) or done % 100 == 0:
                print(f"  [{tag}] {done}/{len(data)}  accuracy={correct / done:.3f}", flush=True)
    finally:
        tokenizer.padding_side = old_pad
        tokenizer.truncation_side = old_trunc
    return correct / max(len(data), 1)


def last_logits(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    with torch.inference_mode():
        return model(input_ids=enc["input_ids"].to(device),
                     attention_mask=enc["attention_mask"].to(device),
                     use_cache=False).logits[0, -1].float().cpu()


def load_model(args):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": "cuda"},
        low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code)
    model.eval()
    return model, tokenizer


def main():
    args = args_parser()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the requested VRAM measurements.")
    if args.bits not in range(1, 9):
        raise SystemExit("--bits must be in 1..8 for packed inference.")
    if args.groupsize != -1 and args.groupsize <= 0:
        raise SystemExit("--groupsize must be -1 or a positive divisor.")
    if not 0 < args.boseir_retain_frac <= 1:
        raise SystemExit("--boseir-retain-frac must be in (0, 1].")
    if args.calib < 1:
        raise SystemExit("--calib must be at least 1.")
    torch.manual_seed(args.seed)
    self_test()

    aliases = {"boseir": "bosesir", "bosesir": "bosesir", "bf16": "bf16",
               "rtn_w4": "rtn_w4", "spinquant_w4": "spinquant_w4"}
    conditions = []
    for name in (x.strip().lower() for x in args.conditions.split(",")):
        if not name:
            continue
        if name not in aliases:
            raise SystemExit(f"Unknown condition: {name}")
        name = aliases[name]
        if name not in conditions:
            conditions.append(name)
    if "bf16" not in conditions:
        conditions.insert(0, "bf16")

    shot_text, test, calib = gsm8k_data(args.n, shots=5, calib_n=args.calib)
    print(f"model={args.model}  GSM8K test={len(test)}  W{args.bits}  "
          f"groupsize={args.groupsize}  dtype={args.dtype}")
    print("conditions:", ",".join(conditions))

    results = {}
    for condition in conditions:
        start = time.time()
        model, tokenizer = load_model(args)
        rotation_cosine = None
        extra = {}
        if condition == "rtn_w4":
            replace_projections(model, args.bits, args.groupsize, not args.asym)
        elif condition == "spinquant_w4":
            reference = last_logits(model, tokenizer, calib[0])
            rotation = apply_spinquant(model, args.seed, args.rotation_path)
            rotated = last_logits(model, tokenizer, calib[0])
            rotation_cosine = float(F.cosine_similarity(reference, rotated, dim=0))
            if rotation_cosine < 0.999:
                raise RuntimeError(f"rotation is not numerically exact (cosine={rotation_cosine:.6f})")
            replace_projections(model, args.bits, args.groupsize, not args.asym)
            extra["rotation"] = rotation
        elif condition == "bosesir":
            extra["bosesir"] = run_boseir(
                model, tokenizer, calib, args.boseir_group_size, args.boseir_retain_frac)

        cleanup()
        storage = model_storage_bytes(model)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 2 ** 30
        torch.cuda.reset_peak_memory_stats()
        accuracy = gsm8k_eval(model, tokenizer, test, shot_text, args, condition)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        results[condition] = {
            "gsm8k_accuracy": accuracy,
            "stored_model_bytes": storage,
            "stored_model_gib": storage / 2 ** 30,
            "allocated_vram_gib": allocated,
            "peak_vram_gib": peak,
            "rotation_cosine": rotation_cosine,
            "time_s": round(time.time() - start, 1),
            **extra,
        }
        print(f"[{condition}] accuracy={accuracy:.3f}  stored={storage / 2 ** 30:.3f} GiB  "
              f"allocated={allocated:.3f} GiB  peak={peak:.3f} GiB\n")
        del model, tokenizer
        cleanup()

    base_acc = results["bf16"]["gsm8k_accuracy"]
    base_bytes = results["bf16"]["stored_model_bytes"]
    print("=" * 100)
    print(f"GSM8K W{args.bits} weight benchmark ({len(test)} examples, 5-shot greedy)")
    print(f"{'condition':<16}{'acc':>8}{'delta':>9}{'stored GiB':>13}"
          f"{'compression':>13}{'alloc GiB':>12}{'peak GiB':>11}")
    for condition, row in results.items():
        delta = 100 * (row["gsm8k_accuracy"] - base_acc)
        ratio = base_bytes / row["stored_model_bytes"]
        print(f"{condition:<16}{row['gsm8k_accuracy']:>8.3f}{delta:>+8.2f}pt"
              f"{row['stored_model_gib']:>13.3f}{ratio:>12.2f}x"
              f"{row['allocated_vram_gib']:>12.3f}{row['peak_vram_gib']:>11.3f}")

    for row in results.values():
        row["delta_accuracy_points_vs_bf16"] = 100 * (row["gsm8k_accuracy"] - base_acc)
        row["compression_vs_bf16"] = base_bytes / row["stored_model_bytes"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"args": vars(args), "results": results}, handle, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
