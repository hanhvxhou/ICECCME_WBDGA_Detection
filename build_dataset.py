#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset.py
================
Xây bộ dữ liệu train/val/test (tỉ lệ 70:15:15) cho bài toán phát hiện
wDGA botnet, với 2 cơ chế chống rò rỉ dữ liệu (data leakage):

  (1) Dedupe theo FQDN (toàn bộ domain, lowercase, bỏ trailing dot).
  (2) Group-aware split theo e2LD (registered domain): tất cả các FQDN
      cùng e2LD sẽ rơi vào CÙNG MỘT tập (train / val / test).

Lấy mẫu (sau khi dedupe + group):
  - Benign  : N_BENIGN mẫu  (label = 0)  — mặc định 78,000
  - Mỗi họ DGA : N_PER_FAMILY mẫu (label = 1) — mặc định 6,000

Stratified theo họ DGA:
  - Mỗi họ tự chia 70 / 15 / 15 (xấp xỉ; tôn trọng nhóm e2LD).
  - Benign cũng chia 70 / 15 / 15 (group-aware theo e2LD).

Đầu ra:
  - train.csv / val.csv / test.csv  (separator: ';',  header: domain;label)
  - stats.json (khi WRITE_STATS = True)

Cách dùng
---------
  Sửa các đường dẫn trong block CONFIG ngay phía dưới rồi:

      python build_dataset.py
"""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG — SỬA Ở ĐÂY
# ============================================================

# Đường dẫn tới file benign (mỗi dòng 1 domain, label = 0)
BENIGN_PATH = r"E:\DGABotnet2k25\Word-BasedDGA\legit-1000000.txt"

# Thư mục chứa các file họ DGA (mỗi file = 1 họ; mỗi dòng 1 domain)
DGA_DIR = r"E:\DGABotnet2k25\Word-BasedDGA\Dataset\wordBased"

# Thư mục output (3 file CSV + stats.json sẽ được ghi vào đây)
OUT_DIR = "E:\DGABotnet2k25\Word-BasedDGA\dataout"

# Số benign cần lấy
N_BENIGN = 78_000

# Số mẫu cần lấy mỗi họ DGA
N_PER_FAMILY = 6_000

# Random seed (đảm bảo tái lập kết quả)
SEED = 42

# Nếu DGA_DIR có nhiều file không liên quan, chỉ nhận file có đuôi này.
# Để trống ("") nếu muốn nhận mọi file thường.
DGA_EXT = ".txt"        # vd: ".txt"

# Có ghi stats.json không?
WRITE_STATS = True

# ============================================================
# (Hết phần CONFIG — không cần sửa gì phía dưới)
# ============================================================


# ---------- 1. Helpers ----------

def _try_import_tldextract():
    """Trả về một callable (fqdn -> e2LD) dùng tldextract nếu có."""
    try:
        import tldextract
        # cache_dir=False để không ghi cache; suffix_list_urls=None để
        # không gọi mạng (dùng PSL đã đóng gói trong package).
        extractor = tldextract.TLDExtract(suffix_list_urls=None, cache_dir=False)

        def _e2ld(fqdn: str) -> str:
            ext = extractor(fqdn)
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}".lower()
            # Không tách được (vd "localhost") → trả về chính fqdn,
            # group sẽ chỉ có 1 phần tử, không ảnh hưởng leakage.
            return fqdn.lower()

        return _e2ld, True
    except Exception:
        return None, False


def _fallback_e2ld(fqdn: str) -> str:
    """
    Fallback: lấy 2 nhãn cuối. Sai với co.uk / com.vn / ac.id... nên
    chỉ dùng khi không có tldextract.
    """
    parts = fqdn.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:]).lower()
    return fqdn.lower()


def normalize_domain(line: str) -> str:
    """Strip + lowercase + bỏ trailing dot. Trả về '' nếu không hợp lệ."""
    s = line.strip().lower().rstrip(".")
    # Bỏ dòng comment, dòng rỗng, hoặc có khoảng trắng giữa domain.
    if not s or s.startswith("#") or any(c.isspace() for c in s):
        return ""
    return s


def read_domain_file(path: Path) -> list:
    """Đọc 1 file text, trả về list FQDN đã normalize + dedupe (giữ thứ tự)."""
    seen = set()
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            d = normalize_domain(line)
            if not d or d in seen:
                continue
            seen.add(d)
            out.append(d)
    return out


# ---------- 2. Group-aware sampling + split ----------

def group_by_e2ld(fqdns: list, e2ld_fn) -> dict:
    """Trả về dict: e2LD -> list[FQDN]."""
    groups = defaultdict(list)
    for d in fqdns:
        groups[e2ld_fn(d)].append(d)
    return groups


def sample_groups_until(groups: dict, target_n: int, rng: random.Random):
    """
    Shuffle các group theo seed, lấy lần lượt cho đến khi đủ target_n mẫu
    (mẫu = FQDN). Group cuối KHÔNG bị cắt — để giữ tính toàn vẹn group
    (chống leakage). Vì vậy số mẫu thực tế có thể >= target_n một chút.
    """
    keys = list(groups.keys())
    rng.shuffle(keys)
    chosen = []
    total = 0
    for k in keys:
        if total >= target_n:
            break
        chosen.append((k, groups[k]))
        total += len(groups[k])
    return chosen, total


def split_groups_70_15_15(groups_list, rng: random.Random):
    """
    Chia danh sách (e2ld, [fqdns]) thành 3 tập theo TỈ LỆ MẪU (FQDN) ≈
    70 / 15 / 15. Group không bị tách. Thuật toán 'rót vào ly': đổ vào
    train cho đến khi đầy quota → val → test. Quota tính theo tổng FQDN.
    """
    rng.shuffle(groups_list)
    total = sum(len(v) for _, v in groups_list)
    quota_train = round(total * 0.70)
    quota_val   = round(total * 0.15)

    train, val, test = [], [], []
    n_train = n_val = 0
    for g in groups_list:
        sz = len(g[1])
        if n_train + sz <= quota_train or not train:
            # 'or not train': đảm bảo train không rỗng nếu group đầu lớn
            train.append(g); n_train += sz
        elif n_val + sz <= quota_val or not val:
            val.append(g); n_val += sz
        else:
            test.append(g)

    # Sanity: nếu test/val rỗng (do dữ liệu quá ít hoặc group cực to),
    # nhường bớt từ train để 3 tập đều khác rỗng.
    if not test and len(train) > 1:
        test.append(train.pop())
    if not val and len(train) > 1:
        val.append(train.pop())

    return train, val, test


def flatten(groups_list):
    """(e2ld, [fqdns]) -> list[fqdn]."""
    out = []
    for _, fqdns in groups_list:
        out.extend(fqdns)
    return out


# ---------- 3. Pipeline chính ----------

def process_source(name, fqdns, target_n, e2ld_fn, rng):
    """
    Trả về dict {'train': [...], 'val': [...], 'test': [...]} cho 1 nguồn
    (benign hoặc 1 họ DGA). Đã group-aware + sample target_n.
    """
    groups = group_by_e2ld(fqdns, e2ld_fn)
    chosen, total_chosen = sample_groups_until(groups, target_n, rng)

    if total_chosen < target_n:
        print(f"  [WARN] {name}: chỉ có {total_chosen} FQDN khả dụng "
              f"(yêu cầu {target_n}). Sẽ dùng toàn bộ.", file=sys.stderr)

    train, val, test = split_groups_70_15_15(chosen, rng)
    return {
        "train": flatten(train),
        "val":   flatten(val),
        "test":  flatten(test),
        "_n_groups": len(chosen),
        "_n_total": total_chosen,
    }


def write_csv(path: Path, rows):
    """rows: iterable of (domain, label). Separator: ';'."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["domain", "label"])
        for d, lab in rows:
            w.writerow([d, lab])


def main():
    # --- Kiểm tra path ---
    benign_path = Path(BENIGN_PATH)
    dga_dir     = Path(DGA_DIR)
    out_dir     = Path(OUT_DIR)

    if not benign_path.is_file():
        sys.exit(f"[ERROR] BENIGN_PATH không tồn tại hoặc không phải file: "
                 f"{benign_path}")
    if not dga_dir.is_dir():
        sys.exit(f"[ERROR] DGA_DIR không phải thư mục: {dga_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] BENIGN_PATH  = {benign_path}")
    print(f"[CONFIG] DGA_DIR      = {dga_dir}")
    print(f"[CONFIG] OUT_DIR      = {out_dir}")
    print(f"[CONFIG] N_BENIGN     = {N_BENIGN:,}")
    print(f"[CONFIG] N_PER_FAMILY = {N_PER_FAMILY:,}")
    print(f"[CONFIG] SEED         = {SEED}")

    # --- Thiết lập e2LD extractor ---
    e2ld_fn, has_tldex = _try_import_tldextract()
    if not has_tldex:
        print("[WARN] Không tìm thấy `tldextract` — dùng fallback đơn giản "
              "(2 nhãn cuối). Khuyến nghị: pip install tldextract",
              file=sys.stderr)
        e2ld_fn = _fallback_e2ld
    else:
        print("[INFO] Dùng tldextract để tách e2LD (chính xác qua PSL).")

    # --- 1. Đọc benign ---
    print(f"\n[1/4] Đọc benign từ {benign_path} ...")
    benign_fqdns = read_domain_file(benign_path)
    print(f"      Sau dedupe FQDN: {len(benign_fqdns):,} domain duy nhất.")

    # --- 2. Đọc các họ DGA ---
    print(f"\n[2/4] Đọc các họ DGA từ {dga_dir} ...")
    family_files = sorted(p for p in dga_dir.iterdir()
                          if p.is_file()
                          and (not DGA_EXT or p.suffix == DGA_EXT))
    if not family_files:
        sys.exit(f"[ERROR] Không tìm thấy file nào trong {dga_dir}"
                 + (f" (lọc theo đuôi {DGA_EXT!r})" if DGA_EXT else ""))

    family_fqdns = {}  # family_name -> list[fqdn]
    for fp in family_files:
        name = fp.stem  # bỏ đuôi làm tên họ
        fqdns = read_domain_file(fp)
        family_fqdns[name] = fqdns
        print(f"      - {name:<20} : {len(fqdns):>7,} FQDN duy nhất")

    print(f"      Tổng: {len(family_fqdns)} họ.")
    if len(family_fqdns) != 13:
        print(f"      [WARN] Đề bài yêu cầu 13 họ, đang có "
              f"{len(family_fqdns)}.", file=sys.stderr)

    # --- 2b. Dedupe cross-family (một FQDN chỉ thuộc 1 họ DGA) ---
    # Nếu cùng 1 domain xuất hiện ở nhiều họ, chỉ giữ ở 1 họ (chọn
    # deterministically theo seed). Việc này CẦN THIẾT để tránh leakage:
    # nếu không, cùng 1 FQDN có thể rơi vào train (qua họ A) và val (qua họ B).
    print(f"\n[2b/4] Dedupe cross-family (một FQDN chỉ thuộc 1 họ) ...")
    rng_dedup = random.Random(SEED - 1)
    fqdn_to_family = {}   # fqdn -> family được giữ
    # Duyệt theo thứ tự shuffle của họ để công bằng
    fam_order = sorted(family_fqdns.keys())
    rng_dedup.shuffle(fam_order)
    dup_counts = defaultdict(int)
    for fam in fam_order:
        for d in family_fqdns[fam]:
            if d in fqdn_to_family:
                dup_counts[(fqdn_to_family[d], fam)] += 1
            else:
                fqdn_to_family[d] = fam
    # Tái tạo family_fqdns sau dedupe
    family_fqdns_dedup = {fam: [] for fam in family_fqdns}
    for d, fam in fqdn_to_family.items():
        family_fqdns_dedup[fam].append(d)
    # Báo cáo
    total_dup = sum(dup_counts.values())
    if total_dup > 0:
        print(f"      Phát hiện {total_dup:,} FQDN xuất hiện ở >1 họ. "
              f"Đã giữ duy nhất 1 bản.")
        for (fam_keep, fam_drop), c in sorted(dup_counts.items(),
                                              key=lambda x: -x[1])[:5]:
            print(f"        • {c:>5,} FQDN bị loại khỏi {fam_drop} "
                  f"(đã thuộc {fam_keep})")
    else:
        print(f"      Không có FQDN nào xuất hiện ở >1 họ. ✓")
    for fam in family_fqdns:
        before = len(family_fqdns[fam])
        after  = len(family_fqdns_dedup[fam])
        if before != after:
            print(f"      - {fam:<20} : {before:>7,} → {after:>7,} "
                  f"(loại {before-after})")
    family_fqdns = family_fqdns_dedup

    # --- 2c. Loại benign có trùng với DGA ---
    # Nếu 1 domain xuất hiện cả ở benign và DGA, ưu tiên giữ ở DGA
    # (vì nhãn DGA là 'ground truth chắc chắn hơn', và để tránh
    # tình huống cùng FQDN có 2 nhãn khác nhau trong dataset).
    all_dga_fqdns = set()
    for fqdns in family_fqdns.values():
        all_dga_fqdns.update(fqdns)
    before = len(benign_fqdns)
    benign_fqdns = [d for d in benign_fqdns if d not in all_dga_fqdns]
    removed = before - len(benign_fqdns)
    if removed > 0:
        print(f"      Loại {removed:,} FQDN khỏi benign vì trùng với DGA "
              f"(giữ ở phía DGA). Còn lại: {len(benign_fqdns):,}.")
    else:
        print(f"      Không có FQDN benign nào trùng với DGA. ✓")

    # --- 3. Sampling + split từng nguồn (group-aware) ---
    print(f"\n[3/4] Sampling + split 70:15:15 (group-aware theo e2LD) ...")

    # Benign — RNG con riêng để stability không phụ thuộc thứ tự xử lý họ.
    rng_benign = random.Random(SEED)
    print(f"  ├─ Benign (target {N_BENIGN:,})")
    benign_split = process_source("benign", benign_fqdns, N_BENIGN,
                                  e2ld_fn, rng_benign)
    print(f"  │   groups chọn: {benign_split['_n_groups']:,}, "
          f"FQDN: {benign_split['_n_total']:,} "
          f"(train {len(benign_split['train']):,} / "
          f"val {len(benign_split['val']):,} / "
          f"test {len(benign_split['test']):,})")

    # Mỗi họ DGA — RNG riêng theo family để tái lập được
    dga_splits = {}
    for i, fam in enumerate(sorted(family_fqdns.keys())):
        rng_fam = random.Random(SEED + 1 + i)
        s = process_source(fam, family_fqdns[fam], N_PER_FAMILY,
                           e2ld_fn, rng_fam)
        dga_splits[fam] = s
        print(f"  ├─ {fam:<20} groups {s['_n_groups']:>5,}  "
              f"FQDN {s['_n_total']:>6,}  "
              f"(tr {len(s['train']):>5,} / va {len(s['val']):>5,} / "
              f"te {len(s['test']):>5,})")

    # --- 4. Gộp + sanity check leakage + ghi file ---
    print(f"\n[4/4] Gộp & kiểm tra leakage ...")

    rows = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for d in benign_split[split]:
            rows[split].append((d, 0))
        for fam, s in dga_splits.items():
            for d in s[split]:
                rows[split].append((d, 1))

    # Kiểm tra leakage: không FQDN nào, cũng không e2LD nào, xuất hiện ở >1 split.
    def check_no_overlap(level_fn, level_name):
        sets = {sp: set(level_fn(d) for d, _ in rows[sp]) for sp in rows}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = sets[a] & sets[b]
            if inter:
                sample = list(inter)[:3]
                sys.exit(f"[FATAL] Leakage ở mức {level_name} giữa "
                         f"{a} và {b}: {len(inter)} phần tử trùng "
                         f"(ví dụ: {sample})")
        print(f"      ✓ Không có overlap ở mức {level_name}.")

    check_no_overlap(lambda d: d, "FQDN")
    check_no_overlap(e2ld_fn,    "e2LD")

    # Shuffle cuối cùng để label không tụm khối
    rng_final = random.Random(SEED + 999)
    for sp in rows:
        rng_final.shuffle(rows[sp])

    # Ghi CSV
    for sp in ("train", "val", "test"):
        out = out_dir / f"{sp}.csv"
        write_csv(out, rows[sp])
        n0 = sum(1 for _, l in rows[sp] if l == 0)
        n1 = sum(1 for _, l in rows[sp] if l == 1)
        print(f"      → {out}  ({len(rows[sp]):,} dòng | "
              f"benign {n0:,} / dga {n1:,})")

    # Stats
    if WRITE_STATS:
        stats = {
            "seed": SEED,
            "tldextract_used": has_tldex,
            "config": {
                "benign_path": str(benign_path),
                "dga_dir": str(dga_dir),
                "n_benign": N_BENIGN,
                "n_per_family": N_PER_FAMILY,
            },
            "splits": {sp: {
                "total": len(rows[sp]),
                "benign": sum(1 for _, l in rows[sp] if l == 0),
                "dga":    sum(1 for _, l in rows[sp] if l == 1),
            } for sp in ("train", "val", "test")},
            "benign_groups_chosen": benign_split["_n_groups"],
            "families": {fam: {
                "groups_chosen": s["_n_groups"],
                "fqdns_chosen":  s["_n_total"],
                "train": len(s["train"]),
                "val":   len(s["val"]),
                "test":  len(s["test"]),
            } for fam, s in dga_splits.items()},
        }
        sp = out_dir / "stats.json"
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"      → {sp}")

    print("\nXong. ✓")


if __name__ == "__main__":
    main()
