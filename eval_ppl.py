"""KV 预算与困惑度评测脚本（kvpress + Qwen2.5-1.5B-Instruct）

约定：
  1. 前缀 P（prefix_len tokens）prefill，期间 press 将 KV cache 压缩到预算 B
  2. 评测段 E（eval_len tokens）按 chunk 流式前向：cache = 压缩前缀 + 已有评测 token，
     评测 token 的 KV 保留在 cache 中（不裁剪），保证与 chunk 大小无关
  3. 显式 position_ids 保持真实绝对位置；PPL = exp(sum NLL / n_tokens)
  4. 精度损失 = PPL(method, B) - PPL(full)
"""
import argparse
import csv
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/Qwen2.5-1.5B-Instruct"


def load_model(dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=dtype, attn_implementation="eager"
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tok


def make_press(name, ratio):
    from kvpress import (
        ObservedAttentionPress,
        PyramidKVPress,
        RandomPress,
        SnapKVPress,
        StreamingLLMPress,
    )

    if name == "random":
        return RandomPress(compression_ratio=ratio)
    if name == "streamingllm":
        return StreamingLLMPress(compression_ratio=ratio, n_sink=4)
    if name == "snapkv":
        return SnapKVPress(compression_ratio=ratio, window_size=64, kernel_size=5)
    if name == "h2o":
        return ObservedAttentionPress(compression_ratio=ratio)
    if name == "pyramidkv":
        return PyramidKVPress(compression_ratio=ratio, window_size=64, kernel_size=5, beta=20)
    raise ValueError(name)


@torch.no_grad()
def compress_prefix(model, prefix_ids, press):
    """prefill 前缀并压缩；返回 cache、每层保留长度、末尾 token 的 hidden state"""
    if press is None:
        out = model.model(input_ids=prefix_ids, use_cache=True)
    else:
        with press(model):
            out = model.model(input_ids=prefix_ids, use_cache=True)
    cache = out.past_key_values
    layer_lens = [int(cache.layers[i].keys.shape[2]) for i in range(len(cache.layers))]
    last_hidden = out.last_hidden_state[:, -1]  # (1, H)，用于预测第一个评测 token
    return cache, layer_lens, last_hidden


@torch.no_grad()
def eval_nll(model, cache, layer_lens, last_hidden, prefix_len, feed_ids, target_ids, chunk=512):
    """对 target_ids 计算 NLL。

    第一个评测 token 由前缀末尾 hidden state 预测（不经 cache）；
    其余按 chunk 流式前向：feed_ids[i] 的绝对位置为 prefix_len+i，
    评测 token 的 KV 保留在 cache 中（标准因果 mask 即正确）。
    逐层预算不一致（PyramidKV）时逐 token 前向（q_len=1 时全可见 mask 天然正确）。
    """
    per_layer = len(set(layer_lens)) > 1
    step = 1 if per_layer else chunk
    n = feed_ids.shape[1]

    logit0 = model.lm_head(last_hidden.unsqueeze(1))[0, 0].float()
    total_nll = F.cross_entropy(logit0, target_ids[0, 0].unsqueeze(0), reduction="sum").item()
    total_tok = 1

    for s in range(0, n, step):
        ids = feed_ids[:, s : s + step]
        c = ids.shape[1]
        tgt = target_ids[:, s + 1 : s + 1 + c]
        cur = int(cache.get_seq_length())
        position_ids = torch.arange(prefix_len + s, prefix_len + s + c, device=ids.device).unsqueeze(0)
        cache_position = torch.arange(cur, cur + c, device=ids.device)

        out = model(
            input_ids=ids,
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = out.logits[0].float()
        total_nll += F.cross_entropy(logits, tgt[0], reduction="sum").item()
        total_tok += c
        cache = out.past_key_values

    return total_nll, total_tok


@torch.no_grad()
def reference_nll(model, ids, n_target):
    """单次完整前向的参考 NLL（最后 n_target 个 token），用于 sanity check"""
    out = model(ids)
    logits = out.logits[0, -n_target - 1 : -1].float()
    tgt = ids[0, -n_target:]
    return F.cross_entropy(logits, tgt, reduction="sum").item(), n_target


def sanity_check(model, tokens):
    prefix_len, eval_len = 2048, 512
    prefix = tokens[:prefix_len].unsqueeze(0).cuda()
    feed = tokens[prefix_len : prefix_len + eval_len - 1].unsqueeze(0).cuda()
    target = tokens[prefix_len : prefix_len + eval_len].unsqueeze(0).cuda()

    cache, layer_lens, last_hidden = compress_prefix(model, prefix, None)
    nll_cont, n_cont = eval_nll(model, cache, layer_lens, last_hidden, prefix_len, feed, target)
    nll_ref, n_ref = reference_nll(model, tokens[: prefix_len + eval_len].unsqueeze(0).cuda(), eval_len)

    assert n_cont == n_ref == eval_len
    diff = abs(nll_cont - nll_ref) / eval_len
    print(f"[sanity] continuation NLL/token = {nll_cont/eval_len:.6f}")
    print(f"[sanity] reference    NLL/token = {nll_ref/eval_len:.6f}")
    print(f"[sanity] per-token diff = {diff:.6f}")
    ok = diff < 1e-3
    print(f"[sanity] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "run"], default="run")
    ap.add_argument("--prefix_len", type=int, default=6144)
    ap.add_argument("--eval_len", type=int, default=3072)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["full", "random", "streamingllm", "snapkv", "h2o", "pyramidkv"],
    )
    ap.add_argument("--out", default="results/ppl_results.csv")
    args = ap.parse_args()

    model, tok = load_model()
    text = open("data/long_text.txt", encoding="utf-8", errors="ignore").read()
    tokens = tok(text, return_tensors="pt").input_ids[0]
    print(f"total tokens available: {tokens.shape[0]}")
    assert tokens.shape[0] >= args.prefix_len + args.eval_len

    if args.mode == "sanity":
        model, tok = load_model(dtype=torch.float32)
        sanity_check(model, tokens)
        return

    need = args.prefix_len + args.eval_len
    tokens = tokens[:need]
    prefix = tokens[: args.prefix_len].unsqueeze(0).cuda()
    feed = tokens[args.prefix_len : need - 1].unsqueeze(0).cuda()
    target = tokens[args.prefix_len : need].unsqueeze(0).cuda()

    rows = []
    for method in args.methods:
        budgets = [args.prefix_len] if method == "full" else args.budgets
        for B in budgets:
            torch.manual_seed(42)
            ratio = 0.0 if method == "full" else 1.0 - B / args.prefix_len
            press = None if method == "full" else make_press(method, ratio)
            t0 = time.time()
            cache, layer_lens, last_hidden = compress_prefix(model, prefix, press)
            nll, ntok = eval_nll(
                model, cache, layer_lens, last_hidden, args.prefix_len, feed, target, args.chunk
            )
            dt = time.time() - t0
            ppl = math.exp(nll / ntok)
            b_mean = sum(layer_lens) / len(layer_lens)
            rows.append(
                dict(method=method, budget=B, budget_measured=round(b_mean, 1),
                     n_tokens=ntok, ppl=round(ppl, 4), seconds=round(dt, 1))
            )
            print(f"{method:14s} B={B:5d} (measured {b_mean:7.1f})  PPL={ppl:9.4f}  [{dt:.1f}s]")
            del cache
            torch.cuda.empty_cache()

    ppl_full = next(r["ppl"] for r in rows if r["method"] == "full")
    for r in rows:
        r["delta_ppl"] = round(r["ppl"] - ppl_full, 4)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
