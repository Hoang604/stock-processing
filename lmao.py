def replace_newline_and_special_patterns(filename):
    try:
        # Đọc nội dung file
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Khởi tạo biến để lưu nội dung đã xử lý
        modified_content = []
        newline_count = 0  # Đếm số lần xuất hiện của '\n'

        # Chuẩn bị các cặp thay thế cho ",A.", ",B.", ",C.", ",D."
        replacements = {
            ',A.': '\t\t1\t\t\t',
            ',B.': '\t\t2\t\t\t',
            ',C.': '\t\t3\t\t\t',
            ',D.': '\t\t4\t\t\t'
        }

        i = 0  # Vị trí con trỏ trong chuỗi nội dung
        while i < len(content):
            # Xử lý các ký tự đặc biệt ",A.", ",B.", ",C.", ",D."
            if content[i:i+3] in replacements:
                modified_content.append(replacements[content[i:i+3]])
                i += 3  # Bỏ qua các ký tự vừa xử lý
                continue
            
            # Xử lý ký tự '\n'
            if content[i] == '\n':
                newline_count += 1
                if newline_count % 5 == 1:
                    # Lần đầu tiên trong chu kỳ 4, thay thế '\n' bằng '\t\t'
                    modified_content.append('\t\t')
                elif newline_count % 5 == 0:
                    # Lần thứ 5, giữ nguyên ký tự '\n'
                    modified_content.append('\n')
                else:
                    # Thay thế 3 ký tự '\n' kế tiếp thành '\t'
                    modified_content.append('\t')
            else:
                # Thêm các ký tự khác vào danh sách
                modified_content.append(content[i])
            
            i += 1  # Tiếp tục xử lý ký tự tiếp theo

        # Chuyển danh sách thành chuỗi
        modified_content = ''.join(modified_content)

        # Ghi nội dung đã thay thế trở lại file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(modified_content)

        print(f"Đã xử lý và thay thế các ký tự đặc biệt trong file {filename}.")
    
    except FileNotFoundError:
        print(f"File {filename} không tồn tại.")

# Gọi hàm với tên file đầu vào
replace_newline_and_special_patterns('chapter1.txt')
