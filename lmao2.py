def process_stock_file(input_file, output_file):
    # Mở file để đọc
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Xử lý từng dòng
    processed_lines = []
    for line in lines:
        # Tách dòng thành các từ
        words = line.split()
        # Kiểm tra có ít nhất 2 từ
        if len(words) > 0:
            # Giữ lại từ thứ 2 và đóng gói trong cặp dấu ""
            processed_line = f'"{words[0]}"'
            processed_lines.append(processed_line)

    # Nối các dòng lại với nhau và thay thế \n bằng \t
    final_output = ','.join(processed_lines)

    # Ghi kết quả vào file đầu ra
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)

# Đường dẫn tới file đầu vào và đầu ra
input_file = 'stockList.txt'
output_file = 'processed_stock.txt'

# Gọi hàm xử lý
process_stock_file(input_file, output_file)
