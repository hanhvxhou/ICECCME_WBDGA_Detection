#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_external.py
================
Test 3 model best (Scenario 3) trên tập dữ liệu ngoài:
  - 4 họ wDGA, mỗi họ 1 file .txt có header "domain"
  - Load model + tokenizer từ best_model/ đã lưu bởi train_3.py
  - Dùng cùng pipeline feature extraction (split_meaningful_words + 11 features)

Cách dùng:
    python test_external.py

Không cần argument — đường dẫn hard-code trong CONFIG.

Output:
  - In bảng tổng hợp: mỗi họ × mỗi model → Detection Rate (DR)
  - Lưu chi tiết từng domain vào CSV
  - Lưu bảng tổng hợp ra file .txt
"""

import csv
import json
import os
import re
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# ── Giảm log noise ──────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
logging.getLogger("codecarbon").setLevel(logging.ERROR)
try:
    import transformers.safetensors_conversion as _sc
    _sc.auto_conversion = lambda *a, **k: None
except Exception:
    pass

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# =============================================================================
# CONFIG — SỬA Ở ĐÂY
# =============================================================================

# Thư mục chứa 3 thư mục best_model (output của train_3.py)
# Mỗi thư mục: results_s3/<model_tag>/best_model/
RESULTS_S3_DIR = r"D:\job\pycharm\Word-BasedDGA\results_s3"

# Thư mục chứa 4 file .txt họ DGA ngoài (mỗi file 1 cột "domain")
EXTERNAL_DIR   = r"D:\job\pycharm\Word-BasedDGA\dga_test_external"

# Thư mục chứa 4 từ điển
DICT_DIR       = r"D:\job\pycharm\Word-BasedDGA\dictionary"

# Thư mục output (CSV chi tiết + bảng tổng hợp)
OUT_DIR        = r"D:\job\pycharm\Word-BasedDGA\results_external"

# Batch size khi inference (tăng nếu VRAM dư)
INFER_BATCH    = 128

# Ngưỡng xác suất để phân loại là DGA (mặc định 0.5)
THRESHOLD      = 0.5

# =============================================================================
# (Hết CONFIG)
# =============================================================================


# ── Feature extraction (copy từ train_3.py) ──────────────────────────────────

def split_meaningful_words(domain: str, dictionary: set):
    domain = re.sub(r'[^a-zA-Z0-9]', '', domain).lower()
    words_found, current_word = [], ""
    meaningful_count, total_meaningful_length = 0, 0
    i = len(domain)
    while i > 0:
        best_match, match_index = "", -1
        for j in range(0, i):
            word = domain[j:i]
            if word in dictionary and len(word) > len(best_match):
                best_match, match_index = word, j
        if best_match:
            meaningful_count += 1
            total_meaningful_length += len(best_match)
            words_found.insert(0, best_match)
            i = match_index + 1
        i -= 1
    if current_word:
        words_found.insert(0, current_word)
    return words_found, meaningful_count, total_meaningful_length


_ONES = ["zero","one","two","three","four","five","six","seven","eight","nine",
         "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
         "seventeen","eighteen","nineteen","twenty"]
_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def _int_to_words(n: int) -> str:
    if n <= 20: return _ONES[n]
    if n < 100:
        t = _TENS[n // 10]; o = _ONES[n % 10] if n % 10 else ""
        return (t + " " + o).strip() if o else t
    if n < 1000:
        h = _ONES[n // 100] + " hundred"; r = n % 100
        return (h + " " + _int_to_words(r)).strip() if r else h
    t = _int_to_words(n // 1000) + " thousand"; r = n % 1000
    return (t + " " + _int_to_words(r)).strip() if r else t


def load_dict(path: Path) -> set:
    if not path.is_file():
        print(f"  [WARN] Từ điển không tồn tại: {path}", file=sys.stderr)
        return set()
    words = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w: words.add(w)
    return words


def extract_11_features(fqdn: str, dictDGA, dictOnlyDGA, dictOnlyNLTK) -> dict:
    no_tld = fqdn.rsplit(".", 1)[0] if "." in fqdn else fqdn
    ls_dga,       _, len_dga       = split_meaningful_words(no_tld, dictDGA)
    ls_only_nltk, _, len_only_nltk = split_meaningful_words(no_tld, dictOnlyNLTK)
    ls_only_dga,  _, len_only_dga  = split_meaningful_words(no_tld, dictOnlyDGA)
    return {
        "f1":  len(no_tld),
        "f2":  ls_dga,       "f3":  ls_only_nltk,   "f4":  ls_only_dga,
        "f5":  len_dga,      "f6":  len_only_nltk,   "f7":  len_only_dga,
        "f8":  max((len(w) for w in ls_dga),       default=0),
        "f9":  max((len(w) for w in ls_only_nltk), default=0),
        "f10": max((len(w) for w in ls_only_dga),  default=0),
        "f11": any(c.isdigit() for c in no_tld),
    }


def features_to_string(f: dict) -> str:
    return " ".join([
        _int_to_words(f["f1"]),
        " ".join(f["f2"]) if f["f2"] else "none",
        " ".join(f["f3"]) if f["f3"] else "none",
        " ".join(f["f4"]) if f["f4"] else "none",
        _int_to_words(f["f5"]), _int_to_words(f["f6"]),
        _int_to_words(f["f7"]), _int_to_words(f["f8"]),
        _int_to_words(f["f9"]), _int_to_words(f["f10"]),
        "yes" if f["f11"] else "no",
    ])


def build_bert_input(fqdn: str, dictDGA, dictOnlyDGA, dictOnlyNLTK) -> str:
    feats = extract_11_features(fqdn, dictDGA, dictOnlyDGA, dictOnlyNLTK)
    return f"{fqdn} {features_to_string(feats)}"


# ── Đọc file external ─────────────────────────────────────────────────────────

def read_external_file(path: Path) -> List[str]:
    """
    Đọc file .txt có header "domain" (1 cột).
    Tự động bỏ qua dòng trống và dòng comment (#).
    """
    domains = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return domains
        # Tìm cột tên là 'domain' (case-insensitive)
        col = next((c for c in reader.fieldnames
                    if c.strip().lower() == "domain"), None)
        if col is None:
            print(f"  [WARN] {path.name}: không tìm thấy cột 'domain'. "
                  f"Cột hiện có: {reader.fieldnames}", file=sys.stderr)
            return domains
        seen = set()
        for row in reader:
            d = row[col].strip().lower().rstrip(".")
            if not d or d.startswith("#") or d in seen:
                continue
            seen.add(d)
            domains.append(d)
    return domains


# ── Load model ────────────────────────────────────────────────────────────────

def discover_models(results_dir: Path) -> List[Dict]:
    """
    Quét results_dir tìm các thư mục có best_model/train_info.json.
    Trả về list dict: {model_name, max_len, best_epoch, model_dir}
    """
    found = []
    for d in sorted(results_dir.iterdir()):
        info_path = d / "best_model" / "train_info.json"
        if not info_path.is_file():
            continue
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        found.append({
            "model_name": info["model_name"],
            "model_dir":  d / "best_model",
            "max_len":    info.get("max_len", 128),
            "best_epoch": info.get("best_epoch", "?"),
            "amp":        info.get("amp", "?"),
            "tag":        d.name,
        })
    return found


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(texts: List[str], model, tokenizer,
                  max_len: int, device, amp_dtype=None) -> Tuple[List[int], List[float]]:
    """
    Trả về (preds, probs_dga):
      preds     : list[int] — 0=benign, 1=dga
      probs_dga : list[float] — xác suất thuộc lớp DGA
    """
    import torch
    model.eval()
    all_preds, all_probs = [], []

    for i in range(0, len(texts), INFER_BATCH):
        batch_texts = texts[i:i + INFER_BATCH]
        enc = tokenizer(batch_texts, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            if amp_dtype and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids=input_ids,
                                   attention_mask=attention_mask).logits
            else:
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask).logits

        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
        preds = [1 if p >= THRESHOLD else 0 for p in probs]
        all_preds.extend(preds)
        all_probs.extend(probs)

    return all_preds, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP  = "=" * 72
    SEP2 = "-" * 72
    buf  = []
    def p(text=""):
        print(text); buf.append(str(text))

    p(SEP)
    p("  TEST EXTERNAL — Scenario 3 models trên dữ liệu ngoài")
    p(SEP)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"\nDevice : {device}")
    if device.type == "cuda":
        p(f"GPU    : {torch.cuda.get_device_name(0)}")

    try:
        amp_dtype = torch.bfloat16
    except Exception:
        amp_dtype = None

    # ── Load từ điển ──────────────────────────────────────────────────────
    p(f"\nLoad từ điển từ: {DICT_DIR}")
    dict_dir = Path(DICT_DIR)
    t0 = time.perf_counter()
    dictDGA      = load_dict(dict_dir / "dictDGA.txt")
    dictOnlyDGA  = load_dict(dict_dir / "dictOnlyDGA.txt")
    dictOnlyNLTK = load_dict(dict_dir / "dictOnlyNLTK.txt")
    p(f"  dictDGA={len(dictDGA):,}  dictOnlyDGA={len(dictOnlyDGA):,}  "
      f"dictOnlyNLTK={len(dictOnlyNLTK):,}  ({time.perf_counter()-t0:.1f}s)")

    # ── Discover models ───────────────────────────────────────────────────
    results_dir = Path(RESULTS_S3_DIR)
    models_info = discover_models(results_dir)
    if not models_info:
        sys.exit(f"[ERROR] Không tìm thấy best_model nào trong {results_dir}.\n"
                 f"        Hãy chạy train_3.py trước.")
    p(f"\nTìm thấy {len(models_info)} model đã train:")
    for m in models_info:
        p(f"  • {m['model_name']:<35} best_ep={m['best_epoch']}  "
          f"max_len={m['max_len']}  [{m['tag']}]")

    # ── Đọc file external ─────────────────────────────────────────────────
    ext_dir = Path(EXTERNAL_DIR)
    if not ext_dir.is_dir():
        sys.exit(f"[ERROR] EXTERNAL_DIR không tồn tại: {ext_dir}")

    family_files = sorted(p2 for p2 in ext_dir.iterdir() if p2.is_file()
                          and p2.suffix.lower() in (".txt", ".csv"))
    if not family_files:
        sys.exit(f"[ERROR] Không tìm thấy file nào trong {ext_dir}")

    p(f"\nĐọc {len(family_files)} file external từ: {ext_dir}")
    families: Dict[str, List[str]] = {}
    for fp in family_files:
        fam   = fp.stem
        doms  = read_external_file(fp)
        families[fam] = doms
        p(f"  • {fam:<25} : {len(doms):,} domain")

    # ── Build BERT input (feature string) ─────────────────────────────────
    p(f"\nBuild feature string (Scenario 3) ...")
    bert_inputs: Dict[str, List[str]] = {}
    for fam, doms in families.items():
        t1 = time.perf_counter()
        inputs = []
        pbar = tqdm(doms, desc=f"  {fam}", leave=False, unit="domain")
        for d in pbar:
            inputs.append(build_bert_input(d, dictDGA, dictOnlyDGA, dictOnlyNLTK))
        bert_inputs[fam] = inputs
        p(f"  {fam:<25}: {len(inputs):,} inputs ({time.perf_counter()-t1:.1f}s)")

    # ── Inference qua từng model ───────────────────────────────────────────
    p(f"\n{SEP2}")
    p(f"INFERENCE — {len(models_info)} model × {len(families)} họ")
    p(SEP2)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # results[model_tag][family] = {"n":int, "dr":float, "preds":[], "probs":[]}
    results: Dict[str, Dict] = {}

    for minfo in models_info:
        tag       = minfo["tag"]
        mname     = minfo["model_name"]
        mdir      = minfo["model_dir"]
        max_len   = minfo["max_len"]
        results[tag] = {"model_name": mname, "families": {}}

        p(f"\n▶ {mname}")
        p(f"  Đang load từ {mdir} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(mdir))
            model     = AutoModelForSequenceClassification.from_pretrained(
                str(mdir)).to(device)
        except Exception as e:
            p(f"  [ERROR] Load model thất bại: {e}")
            continue

        model.eval()
        for fam, doms in families.items():
            inputs = bert_inputs[fam]
            t2 = time.perf_counter()
            preds, probs = run_inference(inputs, model, tokenizer,
                                         max_len, device, amp_dtype)
            n_correct = sum(preds)          # label DGA = 1, tất cả là DGA
            dr        = n_correct / len(preds) if preds else 0.0
            elapsed   = time.perf_counter() - t2
            results[tag]["families"][fam] = {
                "n": len(preds), "n_detected": n_correct,
                "dr": dr, "preds": preds, "probs": probs,
            }
            p(f"  {fam:<25}: DR={dr:.4f}  ({n_correct}/{len(preds)})  {elapsed:.1f}s")

        # Giải phóng VRAM
        del model, tokenizer
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Bảng tổng hợp DR ──────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("BẢNG TỔNG HỢP — Detection Rate (DR) theo họ × model")
    p(SEP)

    fam_list   = list(families.keys())
    model_tags = list(results.keys())

    # header
    col_w  = 26
    m_w    = 14
    header = f"  {'Họ DGA':<{col_w}}" + "".join(
        f"{results[t]['model_name'].split('/')[-1][:m_w]:<{m_w+2}}"
        for t in model_tags)
    p(header)
    p("  " + "-" * (col_w + (m_w + 2) * len(model_tags)))

    for fam in fam_list:
        row = f"  {fam:<{col_w}}"
        for t in model_tags:
            info = results[t]["families"].get(fam)
            if info:
                row += f"{info['dr']:.4f} ({info['n_detected']}/{info['n']})  "
            else:
                row += f"{'N/A':<{m_w+2}}"
        p(row)

    # Trung bình
    p("  " + "-" * (col_w + (m_w + 2) * len(model_tags)))
    avg_row = f"  {'Trung bình':<{col_w}}"
    for t in model_tags:
        drs = [results[t]["families"][f]["dr"]
               for f in fam_list if f in results[t]["families"]]
        avg_row += f"{sum(drs)/len(drs):.4f}{'':<{m_w-4+2}}" if drs else " " * (m_w + 2)
    p(avg_row)
    p(SEP)

    # ── Ghi CSV chi tiết ──────────────────────────────────────────────────
    p(f"\nGhi CSV chi tiết ...")
    for t in model_tags:
        mname_short = results[t]["model_name"].replace("/", "__")
        csv_path = out_dir / f"detail_{mname_short}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["family", "domain", "pred", "prob_dga", "detected"])
            for fam in fam_list:
                if fam not in results[t]["families"]:
                    continue
                info = results[t]["families"][fam]
                for i, (dom, pred, prob) in enumerate(
                        zip(families[fam], info["preds"], info["probs"])):
                    w.writerow([fam, dom, pred, f"{prob:.6f}", pred == 1])
        p(f"  → {csv_path}")

    # ── Ghi summary .txt ──────────────────────────────────────────────────
    summary_path = out_dir / "summary_external.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))
    p(f"\n→ Summary: {summary_path}")
    p("Xong. ✓")


if __name__ == "__main__":
    main()
