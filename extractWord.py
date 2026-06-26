import nltk
from nltk.corpus import words
import re

# Hàm để tách các từ có nghĩa trong một tên miền
def split_meaningful_words(domain, dictionary):
    # Loại bỏ các ký tự không phải chữ và số
    domain = re.sub(r'[^a-zA-Z0-9]', '', domain).lower()
    words_found = []
    current_word = ""
    meaningful_count = 0  # Đếm số từ có nghĩa
    total_meaningful_length = 0  # Tổng số ký tự của các từ có nghĩa
    i = len(domain)

    while i > 0:
        best_match = ""  # Lưu từ dài nhất tìm thấy
        match_index = -1  # Lưu vị trí kết thúc của từ tìm thấy

        # Kiểm tra tất cả các chuỗi con kết thúc tại vị trí `i`
        for j in range(0, i):
            word = domain[j:i]

            # Kiểm tra trong từ điển
            if word in dictionary and len(word) > len(best_match):
                best_match = word
                match_index = j

        if best_match:  # Nếu tìm thấy từ có nghĩa
            meaningful_count += 1  # Tăng số lượng từ có nghĩa
            total_meaningful_length += len(best_match)  # Cộng độ dài từ có nghĩa
            # if current_word:  # Thêm phần chưa tách vào danh sách
            #     words_found.insert(0, current_word)
            #     current_word = ""  # Reset phần chưa tách
            words_found.insert(0, best_match)  # Thêm từ dài nhất vào danh sách
            i = match_index + 1  # Cập nhật vị trí kiểm tra tiếp
        # else:
        #     current_word = domain[i - 1] + current_word  # Thêm ký tự vào phần chưa tách
        i -= 1

    # Thêm phần còn lại nếu còn từ chưa tách
    if current_word:
        words_found.insert(0, current_word)

    return words_found, meaningful_count, total_meaningful_length
