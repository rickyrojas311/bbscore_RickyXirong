#!/usr/bin/env python3
"""
BBScore Validation Script

Verifies that your machine can run BBScore benchmarks end-to-end.
Run this before starting your project.

Usage:
    python validate.py              # Run all tiers (1-3)
    python validate.py --tier 1     # Environment only
    python validate.py --tier 2     # Environment + model inference
    python validate.py --tier 3     # All tiers (default)
"""

import argparse
import sys
import os
import time
import traceback
import warnings

# Enable ANSI escape codes on Windows
if os.name == 'nt':
    os.system('color')

warnings.filterwarnings('ignore')


# -- Formatting helpers --------------------------------------------------

class _C:
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    B = '\033[1m'
    E = '\033[0m'


def _pass(msg):
    print(f"  {_C.G}PASS{_C.E}  {msg}")


def _fail(msg, detail=""):
    print(f"  {_C.R}FAIL{_C.E}  {msg}")
    if detail:
        for line in str(detail).strip().split('\n'):
            print(f"        {line}")


def _warn(msg):
    print(f"  {_C.Y}WARN{_C.E}  {msg}")


def _section(title):
    print(f"\n{_C.B}  {title}{_C.E}")


def _header(tier, title):
    print(f"\n{_C.B}{'=' * 60}{_C.E}")
    print(f"{_C.B}  Tier {tier}: {title}{_C.E}")
    print(f"{_C.B}{'=' * 60}{_C.E}")


# ========================================================================
# Tier 1 — Environment & Dependencies
# ========================================================================
def run_tier1():
    _header(1, "Environment & Dependencies")
    passed = True

    # Python version
    _section("Python")
    v = sys.version_info
    if v >= (3, 9):
        _pass(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor} — requires 3.9+")
        passed = False

    # Core dependencies
    _section("Dependencies")
    core_modules = [
        "torch", "numpy", "scipy", "sklearn", "h5py",
        "transformers", "timm", "PIL", "cv2", "tqdm",
        "boto3", "rsatoolbox",
    ]
    for mod in core_modules:
        try:
            __import__(mod)
            _pass(mod)
        except ImportError:
            _fail(mod, "Not installed. Run: pip install -r requirements.txt")
            passed = False

    # BBScore registries
    _section("BBScore Registries")
    try:
        from benchmarks import BENCHMARK_REGISTRY
        from models import MODEL_REGISTRY
        from metrics import METRICS
        _pass(
            f"{len(MODEL_REGISTRY)} models, "
            f"{len(BENCHMARK_REGISTRY)} benchmarks, "
            f"{len(METRICS)} metrics"
        )
    except Exception as e:
        _fail(f"Failed to load registries: {e}")
        passed = False

    # Hardware
    _section("Hardware")
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
        _pass(f"CPU: {cores} cores / {threads} threads")

        ram_gb = psutil.virtual_memory().total / 1e9
        if ram_gb >= 16:
            _pass(f"RAM: {ram_gb:.0f} GB")
        elif ram_gb >= 8:
            _warn(f"RAM: {ram_gb:.0f} GB (16+ GB recommended)")
        else:
            _fail(f"RAM: {ram_gb:.0f} GB (8 GB minimum)")
            passed = False
    except ImportError:
        _warn("psutil not installed — cannot check RAM/CPU")

    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        _pass(f"GPU: {name} ({vram:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _pass("GPU: Apple MPS (Metal Performance Shaders)")
    else:
        _warn("No GPU detected — CPU only (functional but slower)")

    # Disk
    try:
        import psutil
        data_dir = os.environ.get("SCIKIT_LEARN_DATA", "")
        check_dir = (
            data_dir if data_dir and os.path.exists(data_dir)
            else os.getcwd()
        )
        disk_free = psutil.disk_usage(check_dir).free / 1e9
        if disk_free >= 50:
            _pass(f"Disk: {disk_free:.0f} GB free")
        else:
            _warn(f"Disk: {disk_free:.0f} GB free (50+ GB recommended)")
    except Exception:
        _warn("Could not check disk space")

    # Configuration
    _section("Configuration")
    data_dir = os.environ.get("SCIKIT_LEARN_DATA", "")
    if data_dir and os.path.isdir(data_dir):
        _pass(f"SCIKIT_LEARN_DATA = {data_dir}")
    elif data_dir:
        _warn(
            f"SCIKIT_LEARN_DATA = {data_dir} "
            "(directory does not exist yet)"
        )
    else:
        _warn(
            "SCIKIT_LEARN_DATA not set. "
            "Set it to a directory with 50+ GB free space."
        )

    return passed


# ========================================================================
# Tier 2 — Model Loading & Inference
# ========================================================================
def run_tier2():
    _header(2, "Model Loading & Inference")
    passed = True

    import torch
    # Disable torch.compile for validation — avoids inductor/segfault issues
    torch._dynamo.config.suppress_errors = True
    _orig_compile = torch.compile
    torch.compile = lambda model, *args, **kwargs: model
    from models import get_model_class_and_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Vision model: ResNet-18 ---
    _section("Vision Model (resnet18)")
    try:
        t0 = time.time()
        cls, mid = get_model_class_and_id("resnet18")
        inst = cls()
        model = inst.get_model(mid)
        model.eval().to(device)

        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.inference_mode():
            _ = model(dummy)

        elapsed = time.time() - t0
        _pass(f"Forward pass on {device} ({elapsed:.1f}s)")
        del model, dummy
    except Exception as e:
        _fail(f"resnet18: {e}")
        passed = False

    # --- Vision model: ResNet-50 (torchvision) ---
    _section("Vision Model (resnet50)")
    try:
        import torchvision.models as tv_models
        t0 = time.time()
        model = tv_models.resnet50(weights="DEFAULT")
        model.eval().to(device)

        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.inference_mode():
            _ = model(dummy)

        elapsed = time.time() - t0
        _pass(f"Forward pass on {device} ({elapsed:.1f}s)")
        del model, dummy
    except Exception as e:
        _fail(f"resnet50: {e}")
        passed = False

    # --- Language model: GPT-2 Small ---
    _section("Language Model (gpt2_small)")
    try:
        t0 = time.time()
        cls, mid = get_model_class_and_id("gpt2_small")
        inst = cls()
        model = inst.get_model(mid)
        model.eval().to(device)

        text = "The quick brown fox jumps over the lazy dog"
        input_ids = inst.preprocess_fn(text)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        # Register hook on last transformer block
        features = []

        def hook_fn(module, inp, out):
            features.append(out)

        model.transformer.h[-1].register_forward_hook(hook_fn)

        with torch.inference_mode():
            _ = model(input_ids)

        feat = features[0]
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        processed = inst.postprocess_fn(feat)
        if isinstance(processed, torch.Tensor):
            processed = processed.cpu().numpy()

        dim = processed.squeeze().shape[-1]
        elapsed = time.time() - t0
        _pass(
            f"Forward pass + feature extraction on {device} "
            f"(dim={dim}, {elapsed:.1f}s)"
        )
        del model, features
    except Exception as e:
        _fail(f"gpt2_small: {e}")
        traceback.print_exc()
        passed = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore torch.compile
    torch.compile = _orig_compile

    return passed


# ========================================================================
# Tier 3 — End-to-End Benchmarks
# ========================================================================
def run_tier3():
    _header(3, "End-to-End Benchmarks")
    print(
        "  First run downloads datasets from the cloud.\n"
        "  This may take several minutes depending on your connection.\n"
    )
    passed = True

    import torch

    # --- Vision: ResNet-18 on NSDV1 with RSA ---
    _section("Vision Benchmark (resnet18 + NSDV1 + RSA)")
    try:
        t0 = time.time()
        from benchmarks.NSD.NSDSharedBenchmark import NSDV1Shared

        bench = NSDV1Shared(
            model_identifier="resnet18",
            layer_name="_orig_mod.resnet.encoder.stages.3",
            batch_size=1,
        )
        bench.add_metric("rsa")
        results = bench.run()

        # Extract RSA score from single-layer result
        layer_key = list(results.keys())[0]
        rsa_scores = results[layer_key]['metrics']['rsa']
        spearman = rsa_scores['rsa_scores']['spearman']

        elapsed = time.time() - t0
        _pass(
            f"ResNet-18 x NSDV1 RSA: "
            f"spearman={spearman:.4f} ({elapsed:.0f}s)"
        )
    except Exception as e:
        _fail(f"Vision benchmark: {e}")
        traceback.print_exc()
        passed = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Language: GPT-2 Small on LeBel2023 with Ridge ---
    _section("Language Benchmark (gpt2_small + LeBel2023 + Ridge)")
    try:
        t0 = time.time()
        from benchmarks.LeBel2023.LeBel2023Benchmark import LeBel2023Benchmark

        bench = LeBel2023Benchmark(
            model_identifier="gpt2_small",
            layer_name="transformer.h.11",
            batch_size=1,
        )
        bench.add_metric("ridge")
        results = bench.run()

        # Extract ridge score from single-layer result
        layer_key = list(results.keys())[0]
        ridge_results = results[layer_key]['metrics']['ridge']
        pearson = ridge_results['median_ceiled_pearson']
        if hasattr(pearson, '__len__'):
            pearson = float(pearson.mean())
        else:
            pearson = float(pearson)

        elapsed = time.time() - t0
        _pass(
            f"GPT-2 Small x LeBel2023 Ridge: "
            f"median_ceiled_pearson={pearson:.4f} ({elapsed:.0f}s)"
        )
    except Exception as e:
        _fail(f"Language benchmark: {e}")
        traceback.print_exc()
        passed = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed


# ========================================================================
# Main
# ========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BBScore Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tiers are cumulative: --tier 2 runs tiers 1 and 2.\n"
            "All tiers must pass before starting your project."
        ),
    )
    parser.add_argument(
        "--tier", type=int, default=3, choices=[1, 2, 3],
        help="Run tiers 1 through N (default: 3)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    args = parser.parse_args()

    if args.no_color:
        _C.G = _C.R = _C.Y = _C.B = _C.E = ''

    print(f"\n{_C.B}BBScore Validation{_C.E}")
    print(f"Running tiers 1 through {args.tier}\n")

    results = {}
    timings = {}

    # Tier 1
    t0 = time.time()
    results[1] = run_tier1()
    timings[1] = time.time() - t0

    # Tier 2
    if args.tier >= 2:
        if results[1]:
            t0 = time.time()
            results[2] = run_tier2()
            timings[2] = time.time() - t0
        else:
            print(f"\n  Skipping Tier 2 — Tier 1 failed")
            results[2] = False
            timings[2] = 0

    # Tier 3
    if args.tier >= 3:
        if results.get(2, False):
            t0 = time.time()
            results[3] = run_tier3()
            timings[3] = time.time() - t0
        else:
            print(f"\n  Skipping Tier 3 — earlier tier failed")
            results[3] = False
            timings[3] = 0

    # Summary
    labels = {
        1: "Environment & Dependencies",
        2: "Model Loading & Inference",
        3: "End-to-End Benchmarks",
    }

    print(f"\n{_C.B}{'=' * 60}{_C.E}")
    print(f"{_C.B}  Validation Summary{_C.E}")
    print(f"{_C.B}{'=' * 60}{_C.E}")

    for tier in range(1, args.tier + 1):
        status = _C.G + "PASS" + _C.E if results.get(tier) \
            else _C.R + "FAIL" + _C.E
        t = timings.get(tier, 0)
        print(f"  {status}  Tier {tier}: {labels[tier]}  ({t:.0f}s)")

    all_pass = all(
        results.get(t, False) for t in range(1, args.tier + 1)
    )
    print()
    if all_pass:
        print(
            f"  {_C.G}{_C.B}Your machine is ready "
            f"for BBScore experiments.{_C.E}"
        )
    else:
        print(
            f"  {_C.R}{_C.B}Some checks failed. "
            f"Fix the issues above before starting.{_C.E}"
        )
    print(f"{_C.B}{'=' * 60}{_C.E}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
