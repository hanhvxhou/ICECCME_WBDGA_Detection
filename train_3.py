#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_3.py
==========
Scenario 3 (theo paper): domain + TLD + 11 feature string nối vào làm input BERT.

Pipeline:
  1. Đọc domain từ train/val/test.csv
  2. Load 4 từ điển từ thư mục DICT_DIR:
       dictDGA.txt      — 9,274 từ DGA
       dictOnlyDGA.txt  — 3,506 từ chỉ xuất hiện trong DGA
       dictOnlyNLTK.txt — 228,630 từ chỉ xuất hiện trong NLTK
       dictEng.txt      — 236,682 từ NLTK (dùng để validate)
  3. Trích 11 features theo [25] cho mỗi domain
  4. Convert features số → chuỗi tiếng Anh (pronunciation method)
  5. Nối: "<domain> <feature_string>" → input BERT
  6. Train 3 model × 1 scenario (with_tld) = 3 runs
     [Scenario 3 mặc định dùng domain WITH TLD theo paper]
  7. Output: summary_s3.txt + per-run result.json

11 features (theo paper, bỏ 5 feature ít quan trọng từ [25]):
  f1  lenDomain      — độ dài domain (không TLD)
  f2  lsDGA          — list từ trích bằng dictDGA
  f3  lsOnlyNLTK     — list từ trích bằng dictOnlyNLTK
  f4  lsOnlyDGA      — list từ trích bằng dictOnlyDGA
  f5  lenDGA         — tổng ký tự các từ trong dictDGA
  f6  lenOnlyNLTK    — tổng ký tự các từ trong dictOnlyNLTK
  f7  lenOnlyDGA     — tổng ký tự các từ trong dictOnlyDGA
  f8  maxDGA         — số ký tự từ dài nhất trong dictDGA
  f9  maxOnlyNLTK    — số ký tự từ dài nhất trong dictOnlyNLTK
  f10 maxOnlyDGA     — số ký tự từ dài nhất trong dictOnlyDGA
  f11 hasDigit       — domain có chứa chữ số không (yes/no)

Feature string = các giá trị trên convert sang tiếng Anh, nối bằng dấu cách.
Input BERT cuối = "<domain_with_tld> <feature_string>"

YÊU CẦU
-------
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install transformers scikit-learn tqdm codecarbon psutil
"""

import csv
import gc
import json
import os
import re
import sys
import time
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# ── Giảm log noise ─────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Tắt background thread thử convert safetensors (gây lỗi 403 với SecureBERT)
try:
    import transformers.safetensors_conversion as _sc
    _sc.auto_conversion = lambda *a, **k: None
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*unauthenticated.*",  category=UserWarning)
warnings.filterwarnings("ignore", message=".*Multiple instances.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger("codecarbon.emissions_tracker").setLevel(logging.ERROR)

# ── tqdm ───────────────────────────────────────────────────────────────────
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    class _DummyTqdm:
        def __init__(self, it=None, **kw): self._it = it or []
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def set_postfix(self, *a, **k): pass
        def set_description(self, *a, **k): pass
        def update(self, *a, **k): pass
        def close(self): pass
    def tqdm(iterable=None, **kwargs): return _DummyTqdm(iterable)


# =============================================================================
# CONFIG — SỬA Ở ĐÂY
# =============================================================================

# Thư mục chứa train.csv / val.csv / test.csv
DATA_DIR    = r"E:\DGABotnet2k25\Word-BasedDGA\dataout"

# Thư mục chứa 4 file từ điển
DICT_DIR    = r"D:\job\pycharm\Word-BasedDGA\dictionary"

# Thư mục output
OUT_DIR     = r"E:\DGABotnet2k25\Word-BasedDGA\results_s3"

# File tổng hợp
SUMMARY_TXT = r"E:\DGABotnet2k25\Word-BasedDGA\results_s3\summary_s3.txt"

# 3 model
MODELS: List[str] = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "ehsanaghaei/SecureBERT",
]

# Hyperparameters
MAX_LEN      = 128   # Dùng 128 để không cắt feature string (domain + ~20 token)
BATCH_SIZE   = 32    # Giảm so với trước vì MAX_LEN=128 tốn VRAM gấp 4× MAX_LEN=64
LR           = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS_MAX   = 10
WARMUP_RATIO = 0.1
PATIENCE     = 2
SEED         = 42
DEVICE       = "auto"
USE_AMP      = True
AMP_DTYPE    = "bf16"
MEASURE_ENERGY = True

# =============================================================================
# (Hết CONFIG)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 1. Từ điển
# ─────────────────────────────────────────────────────────────────────────────

def load_dict(path: Path) -> set:
    """Đọc file từ điển (1 từ/dòng) → set lowercase."""
    words = set()
    if not path.is_file():
        sys.exit(f"[ERROR] Không tìm thấy từ điển: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    return words


def load_all_dicts(dict_dir: Path):
    """Load cả 4 từ điển, trả về tuple (dictDGA, dictOnlyDGA, dictOnlyNLTK, dictEng)."""
    d = dict_dir
    print(f"[DICT] Đang load từ điển từ {d} ...")
    dictDGA      = load_dict(d / "dictDGA.txt")
    dictOnlyDGA  = load_dict(d / "dictOnlyDGA.txt")
    dictOnlyNLTK = load_dict(d / "dictOnlyNLTK.txt")
    dictEng      = load_dict(d / "dictEng.txt")
    print(f"  dictDGA      : {len(dictDGA):>9,} từ")
    print(f"  dictOnlyDGA  : {len(dictOnlyDGA):>9,} từ")
    print(f"  dictOnlyNLTK : {len(dictOnlyNLTK):>9,} từ")
    print(f"  dictEng      : {len(dictEng):>9,} từ")
    return dictDGA, dictOnlyDGA, dictOnlyNLTK, dictEng


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature extraction — dùng split_meaningful_words (extractWord.py)
# ─────────────────────────────────────────────────────────────────────────────
#
# Hàm gốc của tác giả: DP từ phải sang trái, tìm từ dài nhất kết thúc
# tại mỗi vị trí. Tốt hơn regex vì tách được từ ghép liền không dấu
# (vd "ordercaught" → ["order","caught"]).
#
# Tối ưu hiệu năng:
#   - Cache kết quả theo (domain_normalized, id(dict)) để tránh tính lại
#     cùng 1 chuỗi nhiều lần giữa 3 từ điển khác nhau.
#   - build_inputs_batch dùng multiprocessing.Pool trên Windows-safe guard.

def split_meaningful_words(domain: str, dictionary: set):
    """
    Tách từ có nghĩa trong domain bằng DP từ phải sang trái.
    Tìm từ DÀI NHẤT kết thúc tại mỗi vị trí i, rồi nhảy sang trái.

    Args:
        domain     : chuỗi đã bỏ TLD (hoặc chưa — hàm tự strip ký tự đặc biệt)
        dictionary : set các từ hợp lệ

    Returns:
        words_found (list[str])   — danh sách từ theo thứ tự xuất hiện
        meaningful_count (int)    — số từ tìm được
        total_meaningful_length   — tổng ký tự của các từ tìm được
    """
    # Bỏ ký tự không phải chữ/số, chuyển thường — giống extractWord.py gốc
    domain = re.sub(r'[^a-zA-Z0-9]', '', domain).lower()

    words_found            = []
    current_word           = ""       # giữ nguyên biến gốc (không dùng nhưng cần cho logic cuối)
    meaningful_count       = 0
    total_meaningful_length = 0
    i = len(domain)

    while i > 0:
        best_match  = ""
        match_index = -1

        for j in range(0, i):
            word = domain[j:i]
            if word in dictionary and len(word) > len(best_match):
                best_match  = word
                match_index = j

        if best_match:
            meaningful_count        += 1
            total_meaningful_length += len(best_match)
            words_found.insert(0, best_match)
            i = match_index + 1
        i -= 1

    if current_word:
        words_found.insert(0, current_word)

    return words_found, meaningful_count, total_meaningful_length


def extract_11_features(fqdn: str,
                        dictDGA: set,
                        dictOnlyDGA: set,
                        dictOnlyNLTK: set) -> dict:
    """Trích 11 features cho 1 FQDN dùng split_meaningful_words."""
    domain_no_tld = fqdn.rsplit(".", 1)[0] if "." in fqdn else fqdn

    ls_dga,       _, len_dga       = split_meaningful_words(domain_no_tld, dictDGA)
    ls_only_nltk, _, len_only_nltk = split_meaningful_words(domain_no_tld, dictOnlyNLTK)
    ls_only_dga,  _, len_only_dga  = split_meaningful_words(domain_no_tld, dictOnlyDGA)

    has_digit = any(c.isdigit() for c in domain_no_tld)

    return {
        "f1":  len(domain_no_tld),
        "f2":  ls_dga,
        "f3":  ls_only_nltk,
        "f4":  ls_only_dga,
        "f5":  len_dga,
        "f6":  len_only_nltk,
        "f7":  len_only_dga,
        "f8":  max((len(w) for w in ls_dga),       default=0),
        "f9":  max((len(w) for w in ls_only_nltk), default=0),
        "f10": max((len(w) for w in ls_only_dga),  default=0),
        "f11": has_digit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature → string (pronunciation method theo paper)
# ─────────────────────────────────────────────────────────────────────────────

# Bảng đọc số thành tiếng Anh (0-20 + chục)
_ONES = ["zero","one","two","three","four","five","six","seven","eight","nine",
         "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
         "seventeen","eighteen","nineteen","twenty"]
_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def _int_to_words(n: int) -> str:
    """Convert số nguyên không âm → chuỗi tiếng Anh (đơn giản, đủ cho feature)."""
    if n <= 20:
        return _ONES[n]
    if n < 100:
        tens = _TENS[n // 10]
        ones = _ONES[n % 10] if n % 10 != 0 else ""
        return (tens + " " + ones).strip() if ones else tens
    if n < 1000:
        hundreds = _ONES[n // 100] + " hundred"
        rest     = n % 100
        return (hundreds + " " + _int_to_words(rest)).strip() if rest else hundreds
    # >= 1000 (độ dài domain có thể lên ~30, không cần xử lý lớn hơn)
    thousands = _int_to_words(n // 1000) + " thousand"
    rest      = n % 1000
    return (thousands + " " + _int_to_words(rest)).strip() if rest else thousands


def features_to_string(feats: dict) -> str:
    """
    Convert 11 features → 1 chuỗi tiếng Anh để nối vào domain.

    Format (theo paper):
      f1(số→word) f2_words... f3_words... f4_words... f5(số→word)
      f6(số→word) f7(số→word) f8(số→word) f9(số→word) f10(số→word)
      f11(yes/no)

    List từ (f2,f3,f4) được nối trực tiếp bằng dấu cách.
    Nếu list rỗng, dùng "none".
    """
    parts = []

    # f1: lenDomain
    parts.append(_int_to_words(feats["f1"]))

    # f2: lsDGA — các từ khớp dictDGA
    parts.append(" ".join(feats["f2"]) if feats["f2"] else "none")

    # f3: lsOnlyNLTK
    parts.append(" ".join(feats["f3"]) if feats["f3"] else "none")

    # f4: lsOnlyDGA
    parts.append(" ".join(feats["f4"]) if feats["f4"] else "none")

    # f5–f10: các giá trị số
    for key in ("f5","f6","f7","f8","f9","f10"):
        parts.append(_int_to_words(feats[key]))

    # f11: hasDigit
    parts.append("yes" if feats["f11"] else "no")

    return " ".join(parts)


def build_bert_input(fqdn: str,
                     dictDGA: set,
                     dictOnlyDGA: set,
                     dictOnlyNLTK: set) -> str:
    """Trả về input BERT: '<fqdn> <feature_string>'"""
    feats = extract_11_features(fqdn, dictDGA, dictOnlyDGA, dictOnlyNLTK)
    fstr  = features_to_string(feats)
    return f"{fqdn} {fstr}"


# Worker-level globals cho multiprocessing (tránh pickle dict lớn mỗi task)
_MP_DICT_DGA      = None
_MP_DICT_ONLY_DGA = None
_MP_DICT_ONLY_NLTK = None

def _mp_init(d_dga, d_only_dga, d_only_nltk):
    """Khởi tạo dict trong mỗi worker process (chạy 1 lần)."""
    global _MP_DICT_DGA, _MP_DICT_ONLY_DGA, _MP_DICT_ONLY_NLTK
    _MP_DICT_DGA       = d_dga
    _MP_DICT_ONLY_DGA  = d_only_dga
    _MP_DICT_ONLY_NLTK = d_only_nltk

def _mp_worker(fqdn: str) -> str:
    return build_bert_input(fqdn,
                            _MP_DICT_DGA,
                            _MP_DICT_ONLY_DGA,
                            _MP_DICT_ONLY_NLTK)


def build_inputs_batch(fqdns: List[str],
                       dictDGA: set,
                       dictOnlyDGA: set,
                       dictOnlyNLTK: set,
                       desc: str = "feature") -> List[str]:
    """
    Xây dựng input BERT cho toàn bộ FQDN.

    split_meaningful_words là O(n²) per domain → với 109k mẫu × 3 dict
    tốn nhiều CPU. Dùng multiprocessing.Pool để tận dụng 24 core của máy.
    Trên Windows bắt buộc chạy trong if __name__ == '__main__' nên pool
    chỉ được khởi động từ main() — không tạo pool ở module-level.
    """
    import multiprocessing as mp

    n_workers = max(1, mp.cpu_count() - 2)   # giữ lại 2 core cho hệ thống
    chunk     = max(1, len(fqdns) // (n_workers * 8))

    print(f"   [{desc}] build feature string: {len(fqdns):,} domain "
          f"| workers={n_workers} | chunk={chunk}")

    t0 = time.perf_counter()
    try:
        # Dùng spawn context (Windows default) với initializer để không
        # pickle dict lớn qua queue mỗi task
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_mp_init,
            initargs=(dictDGA, dictOnlyDGA, dictOnlyNLTK),
        ) as pool:
            results = list(tqdm(
                pool.imap(_mp_worker, fqdns, chunksize=chunk),
                total=len(fqdns), desc=f"   [{desc}]",
                leave=False, unit="domain", dynamic_ncols=True,
            ))
    except Exception as e:
        # Fallback: chạy single-process nếu multiprocessing lỗi
        print(f"   [WARN] multiprocessing lỗi ({e}), chuyển sang single-process")
        results = []
        pbar = tqdm(fqdns, desc=f"   [{desc}]", leave=False,
                    unit="domain", dynamic_ncols=True)
        for fqdn in pbar:
            results.append(build_bert_input(fqdn, dictDGA, dictOnlyDGA, dictOnlyNLTK))
        pbar.close()

    elapsed = time.perf_counter() - t0
    print(f"   [{desc}] xong: {elapsed:.1f}s "
          f"({len(fqdns)/elapsed:,.0f} domain/s)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def read_split(path: Path) -> Tuple[List[str], List[int]]:
    domains, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            domains.append(row["domain"].strip().lower())
            labels.append(int(row["label"]))
    return domains, labels


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pre-tokenized Dataset (module-level — Windows pickle)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch as _t
    from torch.utils.data import Dataset as _DS

    class PreTokenizedDS(_DS):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids      = input_ids
            self.attention_mask = attention_mask
            self.labels         = labels

        def __len__(self):
            return self.input_ids.shape[0]

        def __getitem__(self, i):
            return {
                "input_ids":      self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels":         self.labels[i],
            }

except ImportError:
    PreTokenizedDS = None


def pretokenize(texts: List[str], labels: List[int],
                tokenizer, max_len: int, desc: str = "tokenize") -> "PreTokenizedDS":
    import torch
    print(f"   [{desc}] tokenizing {len(texts):,} samples (max_len={max_len}) ...",
          end=" ", flush=True)
    t0  = time.perf_counter()
    enc = tokenizer(texts, truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt")
    lbl = torch.tensor(labels, dtype=torch.long)
    mem = (enc["input_ids"].nbytes + enc["attention_mask"].nbytes + lbl.nbytes) / 1024**2
    print(f"done ({time.perf_counter()-t0:.1f}s, ~{mem:.0f} MB)")
    return PreTokenizedDS(enc["input_ids"], enc["attention_mask"], lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except ImportError: pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError: pass


def resolve_device(pref: str):
    import torch
    if pref == "cpu":   return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            sys.exit("[ERROR] DEVICE='cuda' nhưng không phát hiện CUDA.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(name: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name, torch.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, amp_dtype=None, desc="eval") -> Dict:
    import torch
    from sklearn.metrics import (
        f1_score, roc_auc_score, accuracy_score,
        precision_score, recall_score, confusion_matrix,
    )
    model.eval()
    all_logits, all_labels = [], []
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)
            if USE_AMP and amp_dtype and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(input_ids=ids, attention_mask=mask)
            else:
                out = model(input_ids=ids, attention_mask=mask)
            all_logits.append(out.logits.detach().float().cpu())
            all_labels.append(lbl.detach().cpu())
    pbar.close()
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds  = logits.argmax(dim=1).numpy()
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "precision":   float(precision_score(labels, preds, zero_division=0)),
        "recall":      float(recall_score(labels, preds, zero_division=0)),
        "f1_binary":   float(f1_score(labels, preds, zero_division=0)),
        "f1_macro":    float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "auc_roc":     float(roc_auc_score(labels, probs)),
        "fpr":         float(fp/(fp+tn)) if (fp+tn) > 0 else 0.0,
        "fnr":         float(fn/(fn+tp)) if (fn+tp) > 0 else 0.0,
        "n_pos":       int((labels==1).sum()),
        "n_neg":       int((labels==0).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Train one run
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(model_name: str,
                  train_texts, train_labels,
                  val_texts,   val_labels,
                  test_texts,  test_labels,
                  device, out_dir: Path) -> Dict:

    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    tag     = model_name.replace("/","__") + "__scenario3"
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    amp_dtype = resolve_amp_dtype(AMP_DTYPE) if USE_AMP else None
    amp_label = f"AMP {AMP_DTYPE.upper()}" if (USE_AMP and device.type=="cuda") else "FP32"

    print(f"\n{'='*72}")
    print(f"▶  RUN: {model_name}  |  Scenario 3  |  {amp_label}")
    print(f"   out_dir = {run_dir}")
    print(f"{'='*72}")

    # ── CodeCarbon ────────────────────────────────────────────────────────
    tracker     = None
    energy_info = {"energy_kwh": None, "co2_kg": None, "tracker_ok": False}
    if MEASURE_ENERGY:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(
                project_name=tag, output_dir=str(run_dir),
                log_level="error", save_to_file=True,
                allow_multiple_runs=True,
            )
            tracker.start()
            energy_info["tracker_ok"] = True
        except Exception as e:
            print(f"   [WARN] CodeCarbon: {e}")

    t_start = time.perf_counter()

    # ── Load tokenizer + model ────────────────────────────────────────────
    print(f"   Loading tokenizer + model ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2).to(device)
    except Exception as e:
        if tracker:
            try: tracker.stop()
            except Exception: pass
        return {"model": model_name, "scenario": "3", "status": f"FAILED at load: {e}",
                "wall_time_sec": 0, **energy_info}

    total_p = sum(p.numel() for p in model.parameters())
    print(f"   Params: {total_p/1e6:.1f}M")

    # ── Pre-tokenize ──────────────────────────────────────────────────────
    train_ds = pretokenize(train_texts, train_labels, tokenizer, MAX_LEN, "train")
    val_ds   = pretokenize(val_texts,   val_labels,   tokenizer, MAX_LEN, "val  ")
    test_ds  = pretokenize(test_texts,  test_labels,  tokenizer, MAX_LEN, "test ")

    nw  = 0   # num_workers=0 trên Windows
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2,   shuffle=False,
                              num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2,   shuffle=False,
                              num_workers=nw, pin_memory=pin)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS_MAX
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    use_scaler = (USE_AMP and device.type=="cuda" and AMP_DTYPE=="fp16")
    scaler     = torch.cuda.amp.GradScaler() if use_scaler else None

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1       = -1.0
    best_epoch        = -1
    epochs_no_improve = 0
    best_state        = None
    history           = []

    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        ep_t0           = time.perf_counter()
        running_loss    = 0.0
        running_correct = 0
        running_seen    = 0
        n_batches       = 0
        n_total         = len(train_loader)

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:>2}/{EPOCHS_MAX} [train]",
                    leave=False, unit="batch", dynamic_ncols=True, total=n_total)

        for batch in pbar:
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if USE_AMP and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(input_ids=ids, attention_mask=msk, labels=lbl)
                loss = out.loss
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                out  = model(input_ids=ids, attention_mask=msk, labels=lbl)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            running_loss    += loss.item()
            n_batches       += 1

            with torch.no_grad():
                pred_b = out.logits.argmax(dim=1)
                running_correct += (pred_b == lbl).sum().item()
                running_seen    += lbl.size(0)

            if n_batches % 10 == 0 or n_batches == n_total:
                pbar.set_postfix({
                    "loss": f"{running_loss/n_batches:.4f}",
                    "acc" : f"{running_correct/running_seen:.4f}",
                    "lr"  : f"{scheduler.get_last_lr()[0]:.2e}",
                })

        pbar.close()
        avg_loss  = running_loss    / max(n_batches, 1)
        train_acc = running_correct / max(running_seen, 1)
        ep_time   = time.perf_counter() - ep_t0

        # Eval val
        val_m = evaluate(model, val_loader, device, amp_dtype,
                         desc=f"Ep {epoch:>2}/{EPOCHS_MAX} [val]  ")
        history.append({
            "epoch":          epoch,
            "train_loss":     avg_loss,
            "train_acc":      train_acc,
            "val_f1":         val_m["f1_binary"],
            "val_auc":        val_m["auc_roc"],
            "val_acc":        val_m["accuracy"],
            "val_prec":       val_m["precision"],
            "val_rec":        val_m["recall"],
            "epoch_time_sec": ep_time,
        })

        improved = val_m["f1_binary"] > best_val_f1
        marker   = " ★ BEST" if improved else ""
        print(
            f"   ep {epoch:>2}/{EPOCHS_MAX} | "
            f"loss={avg_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_f1={val_m['f1_binary']:.4f}  "
            f"auc={val_m['auc_roc']:.4f}  "
            f"acc={val_m['accuracy']:.4f}  "
            f"prec={val_m['precision']:.4f}  "
            f"rec={val_m['recall']:.4f} | "
            f"{ep_time:.1f}s{marker}"
        )

        if improved:
            best_val_f1       = val_m["f1_binary"]
            best_epoch        = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            print(f"   (patience {epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"   [EARLY STOP] best epoch={best_epoch}  "
                      f"best val_f1={best_val_f1:.4f}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        del best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Lưu best model + tokenizer để test_external.py load được
    print(f"   Saving best model → {run_dir / 'best_model'}")
    model.save_pretrained(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))
    with open(run_dir / "best_model" / "train_info.json", "w", encoding="utf-8") as _f:
        json.dump({
            "model_name": model_name,
            "scenario":   "3",
            "max_len":    MAX_LEN,
            "best_epoch": best_epoch,
            "amp":        amp_label,
        }, _f, indent=2)

    # Eval test
    print("   Evaluating on TEST set ...")
    test_m = evaluate(model, test_loader, device, amp_dtype, desc="[test]        ")
    print(
        f"   ── TEST RESULT ──────────────────────────────────────────\n"
        f"   f1={test_m['f1_binary']:.4f}  "
        f"auc={test_m['auc_roc']:.4f}  "
        f"acc={test_m['accuracy']:.4f}  "
        f"prec={test_m['precision']:.4f}  "
        f"rec={test_m['recall']:.4f}  "
        f"fpr={test_m['fpr']:.4f}  "
        f"fnr={test_m['fnr']:.4f}"
    )

    t_total = time.perf_counter() - t_start

    # Stop CodeCarbon
    if tracker is not None:
        try:
            emissions = tracker.stop()
            energy_info["co2_kg"] = float(emissions) if emissions else 0.0
            csv_path = run_dir / "emissions.csv"
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        energy_info["energy_kwh"] = float(
                            rows[-1].get("energy_consumed", 0) or 0)
        except Exception as e:
            print(f"   [WARN] tracker.stop(): {e}")

    result = {
        "model":         model_name,
        "scenario":      "3 (domain + TLD + 11 features)",
        "amp":           amp_label,
        "status":        "OK",
        "best_epoch":    best_epoch,
        "epochs_run":    history[-1]["epoch"] if history else 0,
        "wall_time_sec": t_total,
        "test":          test_m,
        "history":       history,
        **energy_info,
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Giải phóng VRAM
    del model, tokenizer
    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del optimizer, scheduler
    if scaler: del scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. Reporting
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_txt(results: List[Dict], txt_path: Path, meta: Dict):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    def fmt(x, nd=4):
        if x is None: return "N/A"
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    SEP  = "=" * 95
    SEP2 = "-" * 95
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write("  BÁO CÁO TỔNG HỢP — Scenario 3: domain + TLD + 11 features\n")
        f.write(SEP + "\n\n")
        f.write("Metadata:\n")
        for k, v in meta.items():
            f.write(f"  {k:<22}: {v}\n")
        f.write("\n")

        # Bảng 1: test results
        f.write(SEP2 + "\n")
        f.write("BẢNG 1. KẾT QUẢ TRÊN TẬP TEST\n")
        f.write(SEP2 + "\n")
        hdrs = ["Model",               "AMP",   "F1",    "AUC",   "Acc",   "Prec",  "Rec",   "FPR",   "FNR"]
        ws   = [36,                    10,      8,       8,       8,       8,       8,       8,       8]
        f.write("  ".join(h.ljust(w) for h,w in zip(hdrs,ws)) + "\n")
        f.write(SEP2 + "\n")
        for r in results:
            if r["status"] != "OK":
                f.write(f"  {r['model'][:35]:<35}  FAILED: {r['status']}\n"); continue
            t = r["test"]
            row = [r["model"][:35], r.get("amp",""),
                   fmt(t["f1_binary"]), fmt(t["auc_roc"]), fmt(t["accuracy"]),
                   fmt(t["precision"]), fmt(t["recall"]), fmt(t["fpr"]), fmt(t["fnr"])]
            f.write("  ".join(v.ljust(w) for v,w in zip(row,ws)) + "\n")
        f.write("\n")

        # Bảng 2: thời gian + năng lượng
        f.write(SEP2 + "\n")
        f.write("BẢNG 2. THỜI GIAN & NĂNG LƯỢNG\n")
        f.write(SEP2 + "\n")
        h2 = ["Model",               "Wall-time(s)", "Energy(kWh)", "CO2(kg)",  "Best ep", "Ep run"]
        w2 = [36,                    14,             14,            11,         9,         7]
        f.write("  ".join(h.ljust(w) for h,w in zip(h2,w2)) + "\n")
        f.write(SEP2 + "\n")
        for r in results:
            if r["status"] != "OK": continue
            row = [r["model"][:35], f"{r['wall_time_sec']:.1f}",
                   fmt(r["energy_kwh"],6), fmt(r["co2_kg"],6),
                   str(r["best_epoch"]),   str(r["epochs_run"])]
            f.write("  ".join(v.ljust(w) for v,w in zip(row,w2)) + "\n")
        f.write("\n")

        # Chi tiết từng run
        f.write(SEP2 + "\n")
        f.write("CHI TIẾT TỪNG RUN\n")
        f.write(SEP2 + "\n\n")
        for r in results:
            f.write(f"### {r['model']}  |  Scenario 3  |  {r.get('amp','')}\n")
            if r["status"] != "OK":
                f.write(f"   STATUS: {r['status']}\n\n"); continue
            t = r["test"]
            for k,v in [("F1 (binary)",   fmt(t["f1_binary"])),
                        ("F1 (macro)",    fmt(t["f1_macro"])),
                        ("F1 (weighted)", fmt(t["f1_weighted"])),
                        ("AUC-ROC",       fmt(t["auc_roc"])),
                        ("Accuracy",      fmt(t["accuracy"])),
                        ("Precision",     fmt(t["precision"])),
                        ("Recall",        fmt(t["recall"])),
                        ("FPR",           fmt(t["fpr"])),
                        ("FNR",           fmt(t["fnr"])),
                        ("n_pos/n_neg",   f"{t['n_pos']} / {t['n_neg']}"),
                        ("Wall time",     f"{r['wall_time_sec']:.1f} s"),
                        ("Energy",        f"{fmt(r['energy_kwh'],6)} kWh"),
                        ("CO2",           f"{fmt(r['co2_kg'],6)} kg"),
                        ("Best epoch",    f"{r['best_epoch']} / {r['epochs_run']} run")]:
                f.write(f"   {k:<16}: {v}\n")
            f.write(f"\n   History:\n")
            f.write(f"   {'ep':>4}  {'loss':>8}  {'tr_acc':>7}  "
                    f"{'v_f1':>7}  {'v_auc':>7}  {'v_acc':>7}  "
                    f"{'v_prec':>7}  {'v_rec':>7}  {'time(s)':>8}\n")
            f.write(f"   {'-'*76}\n")
            for h in r["history"]:
                f.write(
                    f"   {h['epoch']:>4}  "
                    f"{h['train_loss']:>8.4f}  "
                    f"{h['train_acc']:>7.4f}  "
                    f"{h['val_f1']:>7.4f}  "
                    f"{h['val_auc']:>7.4f}  "
                    f"{h['val_acc']:>7.4f}  "
                    f"{h['val_prec']:>7.4f}  "
                    f"{h['val_rec']:>7.4f}  "
                    f"{h['epoch_time_sec']:>8.1f}\n"
                )
            f.write("\n")
        f.write(SEP + "\n")
        f.write("Hết báo cáo.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    data_dir = Path(DATA_DIR)
    dict_dir = Path(DICT_DIR)
    out_dir  = Path(OUT_DIR);  out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = Path(SUMMARY_TXT)

    # Kiểm tra file CSV
    for sp in ("train", "val", "test"):
        p = data_dir / f"{sp}.csv"
        if not p.is_file():
            sys.exit(f"[ERROR] Không tìm thấy {p}.")

    # Load từ điển
    dictDGA, dictOnlyDGA, dictOnlyNLTK, dictEng = load_all_dicts(dict_dir)

    # Đọc dữ liệu
    print(f"\n[INFO] Đọc dữ liệu từ {data_dir} ...")
    train_d, train_y = read_split(data_dir / "train.csv")
    val_d,   val_y   = read_split(data_dir / "val.csv")
    test_d,  test_y  = read_split(data_dir / "test.csv")
    print(f"   train={len(train_d):,}  val={len(val_d):,}  test={len(test_d):,}")
    print(f"   DGA rate — train={sum(train_y)/len(train_y):.2%}  "
          f"val={sum(val_y)/len(val_y):.2%}  "
          f"test={sum(test_y)/len(test_y):.2%}")

    # Build input Scenario 3: domain + TLD + feature string
    print(f"\n[INFO] Xây dựng input Scenario 3 (domain + TLD + 11 features) ...")
    t_feat = time.perf_counter()
    train_texts = build_inputs_batch(train_d, dictDGA, dictOnlyDGA, dictOnlyNLTK, "train")
    val_texts   = build_inputs_batch(val_d,   dictDGA, dictOnlyDGA, dictOnlyNLTK, "val  ")
    test_texts  = build_inputs_batch(test_d,  dictDGA, dictOnlyDGA, dictOnlyNLTK, "test ")
    print(f"   Feature extraction: {time.perf_counter()-t_feat:.1f}s")

    # In ví dụ để verify
    print(f"\n   Ví dụ input (3 mẫu đầu train):")
    for i in range(min(3, len(train_texts))):
        print(f"   [{train_y[i]}] {train_texts[i][:100]}")

    device = resolve_device(DEVICE)
    amp_label = f"AMP {AMP_DTYPE.upper()}" if (USE_AMP and device.type=="cuda") else "FP32"
    print(f"\n[INFO] Device={device}  |  {amp_label}  |  MAX_LEN={MAX_LEN}  BATCH={BATCH_SIZE}")

    # Train
    all_results = []
    for model_name in MODELS:
        try:
            res = train_one_run(
                model_name=model_name,
                train_texts=train_texts, train_labels=train_y,
                val_texts=val_texts,     val_labels=val_y,
                test_texts=test_texts,   test_labels=test_y,
                device=device, out_dir=out_dir,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"model": model_name, "scenario": "3",
                   "status": f"FAILED: {e}", "wall_time_sec": 0,
                   "energy_kwh": None, "co2_kg": None, "amp": amp_label}
        all_results.append(res)

    # Report
    import torch
    meta = {
        "scenario":        "3 — domain + TLD + 11 features (paper Scenario 3)",
        "data_dir":        str(data_dir),
        "dict_dir":        str(dict_dir),
        "n_train":         len(train_d),
        "n_val":           len(val_d),
        "n_test":          len(test_d),
        "device":          str(device),
        "amp":             amp_label,
        "max_len":         MAX_LEN,
        "batch_size":      BATCH_SIZE,
        "lr":              LR,
        "weight_decay":    WEIGHT_DECAY,
        "epochs_max":      EPOCHS_MAX,
        "patience":        PATIENCE,
        "warmup_ratio":    WARMUP_RATIO,
        "seed":            SEED,
        "torch_ver":       torch.__version__,
        "cuda_ver":        torch.version.cuda or "N/A",
        "dictDGA_size":    len(dictDGA),
        "dictOnlyDGA_size":len(dictOnlyDGA),
        "dictOnlyNLTK_size":len(dictOnlyNLTK),
        "dictEng_size":    len(dictEng),
    }
    write_summary_txt(all_results, txt_path, meta)
    with open(out_dir / "summary_s3.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2, default=str)

    print(f"\n[INFO] Summary TXT : {txt_path}")
    print(f"[INFO] Summary JSON: {out_dir/'summary_s3.json'}")
    print("Xong. ✓")


if __name__ == "__main__":
    main()
