#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
========
Train + đánh giá 3 mô hình BERT cho bài toán phát hiện wDGA botnet:

  1) bert-base-uncased
  2) ehsanaghaei/SecureBERT
  3) distilbert-base-uncased

Mỗi mô hình được train trong 2 SCENARIO:
  - with_tld    : domain giữ nguyên TLD,  vd "superior-generators.it"
  - without_tld : domain bỏ TLD,          vd "superior-generators"

→ Tổng cộng 6 lần train. Mỗi lần in ra:
  • F1-score (macro / binary / weighted)
  • AUC-ROC
  • Accuracy, Precision, Recall
  • Wall-clock time (train + eval)
  • Năng lượng tiêu thụ (kWh) + CO₂ (kg) qua CodeCarbon

Tất cả kết quả được ghi vào MỘT file .txt tổng hợp ở cuối.

Có Early Stopping (patience trên val F1-score).

CÁCH DÙNG
---------
  Sửa CONFIG bên dưới (đường dẫn train/val/test, hyperparam) rồi:

      python train.py

YÊU CẦU
-------
  pip install torch transformers scikit-learn pandas codecarbon

  File train.csv / val.csv / test.csv (tạo bởi build_dataset.py) với
  header  domain;label  và separator ';'.
"""

import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

# Giảm log noise trên Windows / khi không có HF token
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# tqdm cho progress bar; fallback no-op nếu không cài
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(iterable=None, **kwargs):
        # No-op fallback: chỉ trả về iterable, hỗ trợ .set_postfix() / .update() / .close()
        class _Dummy:
            def __init__(self, it): self.it = it
            def __iter__(self): return iter(self.it) if self.it is not None else iter([])
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def set_postfix(self, *a, **k): pass
            def set_description(self, *a, **k): pass
            def update(self, *a, **k): pass
            def close(self): pass
        return _Dummy(iterable)

# ============================================================
# CONFIG — SỬA Ở ĐÂY
# ============================================================

# Thư mục chứa train.csv / val.csv / test.csv (output của build_dataset.py)
DATA_DIR = r"C:\cDGA\Word-BasedDGA\dataout"

# Thư mục output cho kết quả + log + report
OUT_DIR = "results"

# File .txt tổng hợp kết quả cuối cùng
SUMMARY_TXT = "results/summary.txt"

# 3 model HuggingFace muốn so sánh.
# Lưu ý: SecureBERT (RoBERTa-base) lớn hơn DistilBERT, cần GPU >=8GB.
MODELS: List[str] = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "ehsanaghaei/SecureBERT",
]

# 2 scenario về TLD
SCENARIOS: List[str] = ["with_tld", "without_tld"]

# Hyperparameters (chung cho mọi model)
MAX_LEN       = 64        # đa số domain ngắn; tăng nếu input có thêm features
BATCH_SIZE    = 64        # giảm xuống 32 nếu OOM
LR            = 2e-5
WEIGHT_DECAY  = 0.01
EPOCHS_MAX    = 10        # giới hạn trên; early stopping sẽ dừng sớm
WARMUP_RATIO  = 0.1
PATIENCE      = 2         # số epoch không cải thiện val F1 thì dừng
SEED          = 42

# Cố định device:
#   "auto" → CUDA nếu có, không thì CPU
#   "cuda" / "cpu" → ép buộc
DEVICE = "auto"

# Đo điện năng: True → dùng CodeCarbon (cần `pip install codecarbon`)
# Nếu không cài được hoặc lỗi runtime → script vẫn chạy, mục năng lượng = N/A.
MEASURE_ENERGY = True

# ============================================================
# (Hết phần CONFIG)
# ============================================================


# ---------- 0. Setup ----------

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic nhất có thể (đánh đổi tốc độ)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(pref: str):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            sys.exit("[ERROR] CONFIG ép DEVICE='cuda' nhưng không có CUDA.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 1. Data loading ----------

def read_split(path: Path) -> Tuple[List[str], List[int]]:
    """Đọc CSV (separator=';', header=domain;label) → (domains, labels)."""
    domains, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            d = row["domain"].strip().lower()
            y = int(row["label"])
            domains.append(d)
            labels.append(y)
    return domains, labels


def strip_tld(d: str) -> str:
    """
    Bỏ TLD đơn giản: cắt phần sau dấu chấm cuối cùng.
    LƯU Ý: không xử lý đúng các TLD ghép như co.uk (vẫn còn 'co' phía sau).
    Để chính xác hơn có thể dùng tldextract, nhưng vì paper gốc cũng dùng
    cách đơn giản "domain without TLD", giữ logic này cho khớp.
    """
    if "." in d:
        return d.rsplit(".", 1)[0]
    return d


def prepare_texts(domains: List[str], scenario: str) -> List[str]:
    """Chuẩn bị input text cho BERT theo scenario."""
    if scenario == "with_tld":
        return domains
    elif scenario == "without_tld":
        return [strip_tld(d) for d in domains]
    raise ValueError(f"Unknown scenario: {scenario}")


# ---------- 2. Torch Dataset ----------
#
# CRITICAL: phải định nghĩa class ở MODULE-LEVEL (không lồng trong hàm)
# để Windows multiprocessing pickle được khi DataLoader dùng num_workers>0.

try:
    import torch as _torch_for_ds
    from torch.utils.data import Dataset as _Dataset

    class DomainDS(_Dataset):
        """Tokenise on-the-fly. texts: list[str]; labels: list[int]."""
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = self.tokenizer(
                self.texts[i],
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         _torch_for_ds.tensor(self.labels[i],
                                                       dtype=_torch_for_ds.long),
            }
except ImportError:
    # torch chưa cài — sẽ báo lỗi rõ ràng khi vào main()
    DomainDS = None


# ---------- 3. Train / Eval ----------

def evaluate(model, loader, device, desc="eval"):
    """Trả về dict các metric trên loader."""
    import torch
    from sklearn.metrics import (
        f1_score, roc_auc_score, accuracy_score,
        precision_score, recall_score, confusion_matrix
    )

    model.eval()
    all_logits, all_labels = [], []
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch",
                dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            all_logits.append(outputs.logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds  = logits.argmax(dim=1).numpy()

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy":     accuracy_score(labels, preds),
        "precision":    precision_score(labels, preds, zero_division=0),
        "recall":       recall_score(labels, preds, zero_division=0),
        "f1_binary":    f1_score(labels, preds, zero_division=0),
        "f1_macro":     f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted":  f1_score(labels, preds, average="weighted", zero_division=0),
        "auc_roc":      roc_auc_score(labels, probs),
        "fpr":          fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr":          fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "n_pos":        int((labels == 1).sum()),
        "n_neg":        int((labels == 0).sum()),
    }


def train_one_run(model_name: str,
                  scenario: str,
                  train_texts, train_labels,
                  val_texts,   val_labels,
                  test_texts,  test_labels,
                  device,
                  out_dir: Path) -> Dict:
    """
    Train 1 lần (1 model × 1 scenario). Trả về dict kết quả.
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    tag = f"{model_name.replace('/', '__')}__{scenario}"
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"▶  RUN: model={model_name}  scenario={scenario}")
    print(f"   out_dir={run_dir}")
    print(f"{'='*70}")

    # ---- Energy tracker (CodeCarbon) ----
    tracker = None
    energy_info = {"energy_kwh": None, "co2_kg": None, "tracker_ok": False}
    if MEASURE_ENERGY:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(
                project_name=tag,
                output_dir=str(run_dir),
                log_level="error",
                save_to_file=True,
                allow_multiple_runs=True,
            )
            tracker.start()
            energy_info["tracker_ok"] = True
        except Exception as e:
            print(f"   [WARN] CodeCarbon không khởi tạo được: {e}")
            tracker = None

    t_start = time.perf_counter()

    # ---- Tokenizer + Model ----
    print(f"   Loading tokenizer + model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)
    except Exception as e:
        # Cleanup tracker trước khi return lỗi
        if tracker is not None:
            try: tracker.stop()
            except Exception: pass
        return {
            "model": model_name, "scenario": scenario,
            "status": f"FAILED at load: {e}",
            "wall_time_sec": 0,
            **energy_info,
        }

    # ---- DataLoaders ----
    train_ds = DomainDS(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds   = DomainDS(val_texts,   val_labels,   tokenizer, MAX_LEN)
    test_ds  = DomainDS(test_texts,  test_labels,  tokenizer, MAX_LEN)

    # num_workers: 0 trên Windows (spawn chậm + dataset nhẹ, không cần
    # multi-process). Trên Linux có thể dùng 2 worker.
    nw = 2 if (device.type == "cuda" and sys.platform != "win32") else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=nw, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=nw, pin_memory=(device.type == "cuda"))

    # ---- Optimizer + Scheduler ----
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS_MAX
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # ---- Training loop với Early Stopping ----
    best_val_f1 = -1.0
    best_epoch  = -1
    epochs_no_improve = 0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        n_batches = 0
        n_total_batches = len(train_loader)

        # Progress bar cho từng batch trong epoch
        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:>2}/{EPOCHS_MAX} [train]",
                    leave=False, unit="batch", dynamic_ncols=True,
                    total=n_total_batches)
        for batch in pbar:
            optimizer.zero_grad()
            labels_gpu = batch["labels"].to(device)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels_gpu,
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += out.loss.item()
            n_batches += 1
            # Theo dõi accuracy live (rẻ, chỉ argmax)
            with torch.no_grad():
                preds = out.logits.argmax(dim=1)
                running_correct += (preds == labels_gpu).sum().item()
                running_seen    += labels_gpu.size(0)
            # Cập nhật postfix của tqdm: loss trung bình + acc + LR
            if n_batches % 10 == 0 or n_batches == n_total_batches:
                pbar.set_postfix({
                    "loss": f"{running_loss / n_batches:.4f}",
                    "acc":  f"{running_correct / running_seen:.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                })
        pbar.close()
        avg_loss = running_loss / max(n_batches, 1)
        train_acc = running_correct / max(running_seen, 1)
        ep_time = time.perf_counter() - ep_t0

        # Eval trên val
        val_m = evaluate(model, val_loader, device,
                         desc=f"Ep {epoch:>2}/{EPOCHS_MAX} [val] ")
        history.append({
            "epoch":    epoch,
            "train_loss":   avg_loss,
            "train_acc":    train_acc,
            "val_f1":   val_m["f1_binary"],
            "val_auc":  val_m["auc_roc"],
            "val_acc":  val_m["accuracy"],
            "val_prec": val_m["precision"],
            "val_rec":  val_m["recall"],
            "epoch_time_sec": ep_time,
        })
        improved = val_m["f1_binary"] > best_val_f1
        marker = " ★ BEST" if improved else ""
        print(f"   ep {epoch:>2}/{EPOCHS_MAX} | "
              f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f} | "
              f"val: f1={val_m['f1_binary']:.4f} "
              f"auc={val_m['auc_roc']:.4f} "
              f"acc={val_m['accuracy']:.4f} "
              f"prec={val_m['precision']:.4f} "
              f"rec={val_m['recall']:.4f} | "
              f"{ep_time:.1f}s{marker}")

        # Early stopping check
        if improved:
            best_val_f1 = val_m["f1_binary"]
            best_epoch  = epoch
            epochs_no_improve = 0
            # Save best state vào RAM (tiết kiệm disk; epoch ngắn nên OK)
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            print(f"   (val F1 không cải thiện — patience "
                  f"{epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"   [EARLY STOP] Không cải thiện sau {PATIENCE} "
                      f"epoch. Best epoch={best_epoch} f1={best_val_f1:.4f}")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        del best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Final eval trên test ----
    test_m = evaluate(model, test_loader, device, desc="[final test]")
    print(f"   TEST: f1={test_m['f1_binary']:.4f} | "
          f"auc={test_m['auc_roc']:.4f} | "
          f"acc={test_m['accuracy']:.4f} | "
          f"prec={test_m['precision']:.4f} | "
          f"rec={test_m['recall']:.4f} | "
          f"fpr={test_m['fpr']:.4f} | fnr={test_m['fnr']:.4f}")

    t_total = time.perf_counter() - t_start

    # ---- Stop tracker ----
    if tracker is not None:
        try:
            emissions = tracker.stop()  # kg CO2
            # CodeCarbon ghi data chi tiết vào emissions.csv;
            # đọc cột energy_consumed nếu cần (kWh).
            energy_info["co2_kg"] = float(emissions) if emissions else 0.0
            csv_path = run_dir / "emissions.csv"
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        energy_info["energy_kwh"] = float(
                            rows[-1].get("energy_consumed", 0) or 0
                        )
        except Exception as e:
            print(f"   [WARN] tracker.stop() lỗi: {e}")

    # ---- Lưu kết quả run ----
    result = {
        "model": model_name,
        "scenario": scenario,
        "status": "OK",
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "wall_time_sec": t_total,
        "test": test_m,
        "history": history,
        **energy_info,
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Giải phóng VRAM
    del model, tokenizer, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------- 4. Reporting ----------

def write_summary_txt(results: List[Dict], txt_path: Path, meta: Dict):
    """Ghi file .txt tổng hợp."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x, nd=4):
        if x is None: return "N/A"
        if isinstance(x, float): return f"{x:.{nd}f}"
        return str(x)

    with open(txt_path, "w", encoding="utf-8") as f:
        # ---- Header ----
        f.write("=" * 90 + "\n")
        f.write("  BÁO CÁO TỔNG HỢP — wDGA Detection với BERT / SecureBERT / DistilBERT\n")
        f.write("=" * 90 + "\n\n")

        f.write("Metadata:\n")
        for k, v in meta.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\n")

        # ---- Bảng tóm tắt ----
        f.write("-" * 90 + "\n")
        f.write("BẢNG 1. KẾT QUẢ TRÊN TẬP TEST\n")
        f.write("-" * 90 + "\n\n")
        headers = ["Model", "Scenario", "F1", "AUC", "Acc", "Prec", "Rec", "FPR", "FNR"]
        widths  = [32,      14,         8,    8,     8,     8,      7,     7,     7]
        line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        f.write(line + "\n")
        f.write("-" * len(line) + "\n")
        for r in results:
            if r["status"] != "OK":
                row = [r["model"][:31], r["scenario"], r["status"]]
                f.write("  ".join(str(x).ljust(w) for x, w in zip(row, widths[:3])) + "\n")
                continue
            t = r["test"]
            row = [
                r["model"][:31].ljust(widths[0]),
                r["scenario"].ljust(widths[1]),
                fmt(t["f1_binary"]).ljust(widths[2]),
                fmt(t["auc_roc"]).ljust(widths[3]),
                fmt(t["accuracy"]).ljust(widths[4]),
                fmt(t["precision"]).ljust(widths[5]),
                fmt(t["recall"]).ljust(widths[6]),
                fmt(t["fpr"]).ljust(widths[7]),
                fmt(t["fnr"]).ljust(widths[8]),
            ]
            f.write("  ".join(row) + "\n")
        f.write("\n")

        # ---- Bảng thời gian + năng lượng ----
        f.write("-" * 90 + "\n")
        f.write("BẢNG 2. THỜI GIAN & NĂNG LƯỢNG TIÊU THỤ\n")
        f.write("-" * 90 + "\n\n")
        headers2 = ["Model", "Scenario", "Wall-time", "Energy",   "CO2",       "Best epoch", "Epochs run"]
        widths2  = [32,      14,         11,          12,         11,          11,           10]
        units    = ["",      "",         "(seconds)", "(kWh)",    "(kg)",      "",           ""]
        line = "  ".join(h.ljust(w) for h, w in zip(headers2, widths2))
        f.write(line + "\n")
        f.write("  ".join(u.ljust(w) for u, w in zip(units, widths2)) + "\n")
        f.write("-" * len(line) + "\n")
        for r in results:
            if r["status"] != "OK":
                continue
            row = [
                r["model"][:31].ljust(widths2[0]),
                r["scenario"].ljust(widths2[1]),
                f"{r['wall_time_sec']:.1f}".ljust(widths2[2]),
                fmt(r["energy_kwh"], 6).ljust(widths2[3]),
                fmt(r["co2_kg"],     6).ljust(widths2[4]),
                str(r["best_epoch"]).ljust(widths2[5]),
                str(r["epochs_run"]).ljust(widths2[6]),
            ]
            f.write("  ".join(row) + "\n")
        f.write("\n")

        # ---- Chi tiết từng run ----
        f.write("-" * 90 + "\n")
        f.write("CHI TIẾT TỪNG RUN\n")
        f.write("-" * 90 + "\n\n")
        for r in results:
            f.write(f"### {r['model']} | {r['scenario']}\n")
            if r["status"] != "OK":
                f.write(f"   STATUS: {r['status']}\n\n")
                continue
            t = r["test"]
            f.write(f"   F1 (binary)   : {fmt(t['f1_binary'])}\n")
            f.write(f"   F1 (macro)    : {fmt(t['f1_macro'])}\n")
            f.write(f"   F1 (weighted) : {fmt(t['f1_weighted'])}\n")
            f.write(f"   AUC-ROC       : {fmt(t['auc_roc'])}\n")
            f.write(f"   Accuracy      : {fmt(t['accuracy'])}\n")
            f.write(f"   Precision     : {fmt(t['precision'])}\n")
            f.write(f"   Recall        : {fmt(t['recall'])}\n")
            f.write(f"   FPR           : {fmt(t['fpr'])}\n")
            f.write(f"   FNR           : {fmt(t['fnr'])}\n")
            f.write(f"   n_pos / n_neg : {t['n_pos']} / {t['n_neg']}\n")
            f.write(f"   Wall time     : {r['wall_time_sec']:.1f} s\n")
            f.write(f"   Energy        : {fmt(r['energy_kwh'], 6)} kWh\n")
            f.write(f"   CO2           : {fmt(r['co2_kg'],     6)} kg\n")
            f.write(f"   Best epoch    : {r['best_epoch']} / "
                    f"epochs run = {r['epochs_run']}\n")
            f.write(f"   History (val F1 mỗi epoch):\n")
            for h in r["history"]:
                f.write(f"      ep {h['epoch']:>2}: "
                        f"loss={h['train_loss']:.4f}  "
                        f"val_f1={h['val_f1']:.4f}  "
                        f"val_auc={h['val_auc']:.4f}  "
                        f"({h['epoch_time_sec']:.1f}s)\n")
            f.write("\n")

        f.write("=" * 90 + "\n")
        f.write("Hết báo cáo.\n")


# ---------- 5. Main ----------

def main():
    set_seed(SEED)

    data_dir = Path(DATA_DIR)
    out_dir  = Path(OUT_DIR);  out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = Path(SUMMARY_TXT)

    # ---- Kiểm tra file dữ liệu ----
    for sp in ("train", "val", "test"):
        p = data_dir / f"{sp}.csv"
        if not p.is_file():
            sys.exit(f"[ERROR] Không tìm thấy {p}. Hãy chạy build_dataset.py trước.")

    print(f"[INFO] Đọc dữ liệu từ {data_dir} ...")
    train_d, train_y = read_split(data_dir / "train.csv")
    val_d,   val_y   = read_split(data_dir / "val.csv")
    test_d,  test_y  = read_split(data_dir / "test.csv")
    print(f"   train={len(train_d):,}  val={len(val_d):,}  test={len(test_d):,}")
    print(f"   train DGA rate = {sum(train_y)/len(train_y):.2%}")

    device = resolve_device(DEVICE)
    print(f"[INFO] Device = {device}")

    # ---- Loop qua model × scenario ----
    all_results = []
    for scenario in SCENARIOS:
        tr_texts = prepare_texts(train_d, scenario)
        va_texts = prepare_texts(val_d,   scenario)
        te_texts = prepare_texts(test_d,  scenario)
        for model_name in MODELS:
            try:
                res = train_one_run(
                    model_name=model_name,
                    scenario=scenario,
                    train_texts=tr_texts, train_labels=train_y,
                    val_texts=va_texts,   val_labels=val_y,
                    test_texts=te_texts,  test_labels=test_y,
                    device=device,
                    out_dir=out_dir,
                )
            except Exception as e:
                print(f"[ERROR] Run {model_name}/{scenario} FAILED: {e}")
                res = {
                    "model": model_name, "scenario": scenario,
                    "status": f"FAILED: {e}",
                    "wall_time_sec": 0,
                    "energy_kwh": None, "co2_kg": None,
                }
            all_results.append(res)

    # ---- Ghi report ----
    meta = {
        "data_dir":     str(data_dir),
        "n_train":      len(train_d),
        "n_val":        len(val_d),
        "n_test":       len(test_d),
        "device":       str(device),
        "max_len":      MAX_LEN,
        "batch_size":   BATCH_SIZE,
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs_max":   EPOCHS_MAX,
        "patience":     PATIENCE,
        "warmup_ratio": WARMUP_RATIO,
        "seed":         SEED,
    }
    write_summary_txt(all_results, txt_path, meta)
    # Ghi thêm JSON gọn để parse sau
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2,
                  default=str)

    print(f"\n[INFO] Báo cáo tổng hợp đã ghi vào: {txt_path}")
    print(f"[INFO] JSON đã ghi vào: {out_dir/'summary.json'}")
    print("Xong. ✓")


if __name__ == "__main__":
    main()
