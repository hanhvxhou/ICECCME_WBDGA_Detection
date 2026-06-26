import random

# ==========================================
# Word-based DGA Generator
# Sinh 2 họ:
#   - gen2k26ver1.txt
#   - gen2k26ver2.txt
# ==========================================

TLDS = ["com", "net", "org", "inter", "ai"]

DICT1_FILE = r"D:\job\pycharm\Word-BasedDGA\dictionary\dictDGA.txt"
DICT2_FILE = r"D:\job\pycharm\Word-BasedDGA\dictionary\dictEng.txt"

OUTPUT_VER1 = r"D:\job\pycharm\Word-BasedDGA\dga_test_external\gen2k26ver1.txt"
OUTPUT_VER2 = r"D:\job\pycharm\Word-BasedDGA\dga_test_external\gen2k26ver2.txt"

NUM_DOMAINS = 500


# ==========================================
# Đọc dictionary
# Chỉ lấy từ có độ dài từ 4-8 ký tự
# ==========================================

def load_words(filename):

    words = []

    with open(filename, "r", encoding="utf-8") as f:

        for line in f:

            word = line.strip().lower()

            # Chỉ lấy ký tự alphabet
            if not word.isalpha():
                continue

            # Độ dài 4-8
            if 4 <= len(word) <= 8:
                words.append(word)

    # Loại bỏ trùng
    words = list(set(words))

    return words


# ==========================================
# Sinh domain
# Cấu trúc:
# word1-word2-word3.tld
# ==========================================

def generate_domains(words, count):

    domains = set()

    while len(domains) < count:

        w1 = random.choice(words)
        w2 = random.choice(words)
        w3 = random.choice(words)

        tld = random.choice(TLDS)

        domain = f"{w1}-{w2}-{w3}.{tld}"

        domains.add(domain)

    return list(domains)


# ==========================================
# Ghi file
# ==========================================

def save_domains(filename, domains):

    with open(filename, "w", encoding="utf-8") as f:

        for domain in domains:
            f.write(domain + "\n")


# ==========================================
# Main
# ==========================================

def main():

    # Đọc 2 bộ từ điển
    words_ver1 = load_words(DICT1_FILE)
    words_ver2 = load_words(DICT2_FILE)

    # Kiểm tra dữ liệu
    if len(words_ver1) < 3:
        print("dic1.txt không đủ từ hợp lệ")
        return

    if len(words_ver2) < 3:
        print("dic2.txt không đủ từ hợp lệ")
        return

    # Sinh domain
    domains_ver1 = generate_domains(words_ver1, NUM_DOMAINS)
    domains_ver2 = generate_domains(words_ver2, NUM_DOMAINS)

    # Ghi file
    save_domains(OUTPUT_VER1, domains_ver1)
    save_domains(OUTPUT_VER2, domains_ver2)

    print(f"Đã tạo {OUTPUT_VER1}: {len(domains_ver1)} domains")
    print(f"Đã tạo {OUTPUT_VER2}: {len(domains_ver2)} domains")


if __name__ == "__main__":
    main()