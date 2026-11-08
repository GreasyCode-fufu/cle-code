#!/usr/bin/env python3
"""R6 复杂度/延迟基准: Params(部署口径) / MACs(thop) / CPU 单线程与 4 线程延迟 / GPU bs=1 延迟 / GPU bs64 AMP 吞吐。
轻量网络的优势体现在 CPU/边缘端与批吞吐; 4090 上 bs=1 受 kernel 启动开销限制, 各模型均 ~1.15ms(见 P0)。
测量方式: 各模型交替轮测 ROUNDS 轮取中位数, 抵消 CPU 频率/后台负载漂移; 须在机器空闲时运行。
用法: python experiments/r6_bench_latency.py [model ...] [--out experiments/r6_bench_latency.json]
"""
import argparse
import copy
import json
import os
import statistics
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from cle_fair_common import build_model  # noqa: E402

TRAIN_ONLY = ("anchor_net", "r6edge")       # 训练期专用模块(推理不执行), 不计入部署参数


def deploy_params(m):
    return sum(p.numel() for n, p in m.named_parameters() if n.split(".")[0] not in TRAIN_ONLY)


def macs_g(m):
    from thop import profile
    mm = copy.deepcopy(m).cpu().eval()
    macs, _ = profile(mm, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    return macs / 1e9


@torch.inference_mode()
def cpu_once(m, x, iters):
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        m(x)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


@torch.inference_mode()
def gpu_b1(m, iters=300):
    x = torch.randn(1, 3, 224, 224, device="cuda")
    for _ in range(30):
        m(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        m(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


@torch.inference_mode()
def gpu_tput(m, bs=64, iters=40):
    x = torch.randn(bs, 3, 224, 224, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
    return bs * iters / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["mobilenet_v3_large", "mbv3_r6_star2"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "r6_bench_latency.json"))
    args = ap.parse_args()
    nets = {n: build_model(n, 3).eval() for n in args.models}
    res = {n: {"params_M": round(deploy_params(m) / 1e6, 3), "macs_G": round(macs_g(m), 4)} for n, m in nets.items()}
    x = torch.randn(1, 3, 224, 224)
    for th in (1, 4):
        torch.set_num_threads(th)
        for m in nets.values():
            cpu_once(m, x, 5)                   # 预热
        samples = {n: [] for n in nets}
        for _ in range(args.rounds):            # 交替轮测
            for n, m in nets.items():
                samples[n].append(cpu_once(m, x, 20))
        for n in nets:
            res[n][f"cpu_t{th}_ms"] = round(statistics.median(samples[n]), 2)
    for n, m in nets.items():
        g = m.cuda()
        res[n]["gpu_b1_ms"] = round(gpu_b1(g), 3)
        res[n]["gpu_b64_amp_img_s"] = round(gpu_tput(g))
        nets[n] = g.cpu()
        torch.cuda.empty_cache()
    for n, r in res.items():
        print(f"{n:22s} " + "  ".join(f"{k}={v}" for k, v in r.items()))
    json.dump(res, open(args.out, "w"), indent=1)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
