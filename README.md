# NHÓM 11 – ĐỀ TÀI 7: PHÂN LOẠI HOA (FIT4013)

**Thành viên**
- Đỗ Tiến Sơn
- Đỗ Huy Anh Duy

## 1. Mục tiêu
Xây dựng pipeline phân loại ảnh hoa trên bộ dữ liệu Oxford Flowers 102; thực hiện EDA, tiền xử lý ảnh, trích xuất đặc trưng, huấn luyện, đánh giá và phân tích lỗi.

## 2. Dữ liệu
Theo đường dẫn Đề tài 7 trong tài liệu FIT4013, nhóm sử dụng Oxford Flowers 102:
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Có thể tải tự động bằng `torchvision.datasets.Flowers102`. Bộ dữ liệu chuẩn gồm 102 lớp; train 1.020 ảnh, validation 1.020 ảnh, test 6.149 ảnh. Giữ nguyên split chuẩn.

## 3. Các thí nghiệm
1. Baseline: ảnh RGB resize 32×32, chuẩn hóa pixel, K-NN.
2. K-NN với đặc trưng HOG (ảnh grayscale resize 64×64).
3. Decision Tree với đặc trưng HOG.

Tham số được chọn trên validation; tập test chỉ dùng để đánh giá cuối cùng. Các phép biến đổi ảnh không học tham số từ test.

## 4. Chạy trên Google Colab (khuyến nghị)
1. Tải và giải nén repository.
2. Mở `notebooks/Nhom11_PhanLoaiHoa.ipynb` bằng Google Colab.
3. Chọn Runtime → Change runtime type → CPU hoặc GPU (pipeline cổ điển chủ yếu dùng CPU).
4. Chạy lần lượt các cell. Cell đầu cài thư viện nếu cần.
5. Dữ liệu được tải tự động vào `./data`.
6. Kết quả, biểu đồ và mô hình được lưu trong `./outputs`.

Hoặc chạy local:
```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt

# 2. Huấn luyện toàn bộ pipeline và xuất kết quả
python src/train.py

# 3. Khởi chạy giao diện Web trên trình duyệt (Khuyến nghị)
python src/web_app.py

# 4. Khởi chạy giao diện Desktop tương tác (GUI Tkinter)
python src/gui_app.py

# (Tùy chọn) Chạy chế độ headless sinh ảnh minh chứng giao diện
python src/gui_app.py --demo
```

## 5. Cấu trúc
- `notebooks/`: notebook chạy toàn bộ pipeline.
- `src/train.py`: mã nguồn huấn luyện pipeline độc lập.
- `src/web_app.py`: giao diện ứng dụng Web tương tác trên trình duyệt.
- `src/gui_app.py`: giao diện Desktop GUI Tkinter.
- `report/main.tex`: báo cáo LaTeX theo sáu phần nội dung bắt buộc.
- `outputs/`: được tạo sau khi chạy, chứa CSV, biểu đồ, mô hình và báo cáo metric.
- `AI_USAGE.md`: khai báo sử dụng AI.

## 6. Lưu ý quan trọng
Không điền số liệu giả vào báo cáo. Sau khi chạy notebook/script, lấy số liệu thực tế từ `outputs/metrics.csv`, `outputs/classification_report_*.csv` và các hình trong `outputs/` để hoàn thiện phần kết quả. Nhóm cần kiểm tra, hiểu và giải thích mã nguồn trước khi nộp.

## 7. Trích dẫn dữ liệu
Nilsback, M.-E. and Zisserman, A. (2008). *Automated flower classification over a large number of classes*. Indian Conference on Computer Vision, Graphics and Image Processing.
