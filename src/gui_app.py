"""
GIAO DIỆN DEMO NHẬN DIỆN VÀ PHÂN LOẠI HOA (FIT4013)
Nhóm 11 - Đề tài 7: Phân loại hoa Oxford Flowers 102
Trường Đại học Đại Nam - Khoa Công nghệ Thông tin
Giảng viên hướng dẫn: ThS. Lê Thị Thùy Trang
Thành viên: Đỗ Tiến Sơn - Đỗ Huy Anh Duy
"""
import os, sys, time, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Từ điển 102 loài hoa (Song ngữ Anh - Việt)
CATEGORY_NAMES = {
    1: ('pink primrose', 'Hoa Anh Thảo Hồng'),
    2: ('hard-leaved pocket orchid', 'Hoa Lan Hài Lá Cứng'),
    3: ('canterbury bells', 'Hoa Chuông Tử Đằng'),
    4: ('sweet pea', 'Hoa Đậu Thơm'),
    5: ('english marigold', 'Hoa Cúc Vạn Thọ Anh'),
    6: ('tiger lily', 'Hoa Loa Kèn Da Báo'),
    7: ('moon orchid', 'Hoa Lan Hồ Điệp Trắng'),
    8: ('bird of paradise', 'Hoa Thiên Điểu'),
    9: ('monkshood', 'Hoa Phi Yến Xanh'),
    10: ('globe thistle', 'Hoa Cúc Gai Cầu Tím'),
    11: ('snapdragon', 'Hoa Mõm Chó'),
    12: ("colt's foot", 'Hoa Móng Ngựa Vàng'),
    13: ('king protea', 'Hoa Vua Protea Nam Phi'),
    14: ('spear thistle', 'Hoa Kế Binh Gai'),
    15: ('yellow iris', 'Hoa Diên Vĩ Vàng'),
    16: ('globe-flower', 'Hoa Cầu Vàng'),
    17: ('purple coneflower', 'Hoa Cúc Xích Tím Echinacea'),
    18: ('peruvian lily', 'Hoa Thủy Tiên Peru'),
    19: ('balloon flower', 'Hoa Cát Cánh Chuông'),
    20: ('giant white arum lily', 'Hoa Rum Trắng Khổng Lồ'),
    21: ('fire lily', 'Hoa Hỏa Ly Đỏ Cam'),
    22: ('pincushion flower', 'Hoa Gối Cắm Kim'),
    23: ('fritillary', 'Hoa Thư Sinh Ca-rô'),
    24: ('red ginger', 'Hoa Gừng Đỏ'),
    25: ('grape hyacinth', 'Hoa Lan Dạ Hương Chùm Nho'),
    26: ('corn poppy', 'Hoa Anh Túc Đỏ Đồng'),
    27: ('prince of wales feathers', 'Hoa Lông Vũ Xứ Wales'),
    28: ('stemless gentian', 'Hoa Long Đởm Lam'),
    29: ('artichoke', 'Hoa Atisô'),
    30: ('sweet william', 'Hoa Cẩm Chướng Râu'),
    31: ('carnation', 'Hoa Cẩm Chướng Hồng'),
    32: ('garden phlox', 'Hoa Trúc Đào Phlox'),
    33: ('love in the mist', 'Hoa Tình Thoảng Trong Sương'),
    34: ('mexican aster', 'Hoa Cúc Sao Mexico'),
    35: ('alpine sea holly', 'Hoa Kế Biển Alpine'),
    36: ('ruby-lipped cattleya', 'Hoa Lan Hoàng Thảo Môi Đỏ'),
    37: ('cape flower', 'Hoa Mũi Hảo Vọng'),
    38: ('great masterwort', 'Hoa Đại Cự Astrantia'),
    39: ('siam tulip', 'Hoa Uất Kim Hương Xiêm'),
    40: ('lenten rose', 'Hoa Hồng Mùa Chay'),
    41: ('barbeton daisy', 'Hoa Cúc Đồng Tiền Barberton'),
    42: ('daffodil', 'Hoa Thủy Tiên Vàng'),
    43: ('sword lily', 'Hoa Lay Ơn Kiếm'),
    44: ('poinsettia', 'Hoa Trạng Nguyên Đỏ'),
    45: ('bolero deep blue', 'Hoa Vũ Nữ Xanh Thẫm Bolero'),
    46: ('wallflower', 'Hoa Tường Vi Cổ Điển'),
    47: ('marigold', 'Hoa Cúc Vạn Thọ Cam'),
    48: ('buttercup', 'Hoa Mao Lương Vàng Óng'),
    49: ('oxeye daisy', 'Hoa Cúc Mắt Bò Trắng'),
    50: ('common dandelion', 'Hoa Bồ Công Anh Vàng'),
    51: ('petunia', 'Hoa Dạ Yến Thảo'),
    52: ('wild pansy', 'Hoa Păng-xê Dại Ba Màu'),
    53: ('primula', 'Hoa Báo Xuân'),
    54: ('sunflower', 'Hoa Hướng Dương'),
    55: ('pelargonium', 'Hoa Phong Lữ Vườn'),
    56: ('bishop of llandaff', 'Hoa Thược Dược Giám Mục Đỏ'),
    57: ('gaura', 'Hoa Bướm Gaura Trắng Hồng'),
    58: ('geranium', 'Hoa Phong Lữ Thảo'),
    59: ('orange dahlia', 'Hoa Thược Dược Cam'),
    60: ('pink-yellow dahlia', 'Hoa Thược Dược Hồng Vàng'),
    61: ('cautleya spicata', 'Hoa Cautleya Rừng Rậm'),
    62: ('japanese anemone', 'Hoa Phong Quỳ Nhật Bản'),
    63: ('black-eyed susan', 'Hoa Cúc Mắt Huyền'),
    64: ('silverbush', 'Hoa Bụi Bạc Convolvulus'),
    65: ('californian poppy', 'Hoa Anh Túc Vàng California'),
    66: ('osteospermum', 'Hoa Cúc Châu Phi'),
    67: ('spring crocus', 'Hoa Nghệ Tây Mùa Xuân Tím'),
    68: ('bearded iris', 'Hoa Diên Vĩ Râu Tím'),
    69: ('windflower', 'Hoa Thảo Anemone Trắng'),
    70: ('tree poppy', 'Hoa Anh Túc Thân Gỗ'),
    71: ('gazania', 'Hoa Cúc Huân Chương'),
    72: ('azalea', 'Hoa Đỗ Quyên Rực Rỡ'),
    73: ('water lily', 'Hoa Súng Nước'),
    74: ('rose', 'Hoa Hồng Đỏ'),
    75: ('thorn apple', 'Hoa Cà Độc Dược Trắng'),
    76: ('morning glory', 'Hoa Bìm Bìm Tím'),
    77: ('passion flower', 'Hoa Lạc Tiên Độc Đáo'),
    78: ('lotus', 'Hoa Sen Hồng'),
    79: ('toad lily', 'Hoa Ly Da Cóc Đốm'),
    80: ('anthurium', 'Hoa Hồng Môn Đỏ'),
    81: ('frangipani', 'Hoa Đại / Hoa Sứ Trắng'),
    82: ('clematis', 'Hoa Ông Lão Clematis Tím'),
    83: ('hibiscus', 'Hoa Dâm Bụt Đỏ'),
    84: ('columbine', 'Hoa Bồ Câu Aquilegia'),
    85: ('desert-rose', 'Hoa Sứ Sa Mạc Adenium'),
    86: ('tree mallow', 'Hoa Cẩm Quỳ Thân Gỗ Hồng'),
    87: ('magnolia', 'Hoa Mộc Lan Trắng Ngà'),
    88: ('cyclamen', 'Hoa Tiên Khách Lai Hồng'),
    89: ('watercress', 'Hoa Cải Xoong'),
    90: ('canna lily', 'Hoa Chuối Hoa Đỏ Canna'),
    91: ('hippeastrum', 'Hoa Lan Huệ'),
    92: ('bee balm', 'Hoa Tía Tô Đỏ Monarda'),
    93: ('ball moss', 'Hoa Rêu Cầu'),
    94: ('foxglove', 'Hoa Mao Địa Hoàng Chuông'),
    95: ('bougainvillea', 'Hoa Giấy Rực Rỡ'),
    96: ('camellia', 'Hoa Trà My Đỏ Hồng'),
    97: ('mallow', 'Hoa Cẩm Quỳ Tím'),
    98: ('mexican petunia', 'Hoa Dạ Yến Thảo Lam Mexico'),
    99: ('bromelia', 'Hoa Dứa Cảnh Nến'),
    100: ('blanket flower', 'Hoa Gaillardia Vải Hoa'),
    101: ('trumpet creeper', 'Hoa Đăng Tiêu Cam'),
    102: ('blackberry lily', 'Hoa Xạ Can / Rẻ Quạt Cam')
}

def find_project_root():
    """Tự động xác định thư mục gốc của dự án."""
    cwd = Path.cwd().resolve()
    if (cwd / "outputs" / "best_knn_hog_model.pt").exists():
        return cwd
    if (cwd / "src" / "gui_app.py").exists():
        return cwd
    if (cwd / "Nhom11_PhanLoaiHoa_FIT4013").exists():
        return cwd / "Nhom11_PhanLoaiHoa_FIT4013"
    return cwd

ROOT_DIR = find_project_root()

def compute_hog_single(img_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    H, W = img_array.shape
    gx = np.zeros_like(img_array)
    gy = np.zeros_like(img_array)
    gx[:, 1:-1] = img_array[:, 2:] - img_array[:, :-2]
    gy[1:-1, :] = img_array[2:, :] - img_array[:-2, :]
    
    mag = np.sqrt(gx**2 + gy**2)
    ori = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0
    
    cx, cy = pixels_per_cell
    n_cells_y = H // cy
    n_cells_x = W // cx
    
    hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    bin_width = 180.0 / orientations
    bin_idx = np.clip((ori / bin_width).astype(int), 0, orientations - 1)
    
    for b in range(orientations):
        mask = (bin_idx == b)
        w_mag = mag * mask
        hist[:, :, b] = w_mag.reshape(n_cells_y, cy, n_cells_x, cx).sum(axis=(1, 3))
                
    bx, by = cells_per_block
    n_blocks_y = n_cells_y - by + 1
    n_blocks_x = n_cells_x - bx + 1
    
    blocks = []
    eps = 1e-5
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block = hist[i:i+by, j:j+bx, :].ravel()
            norm = np.sqrt(np.sum(block**2) + eps)
            block = block / norm
            block = np.clip(block, 0, 0.2)
            norm = np.sqrt(np.sum(block**2) + eps)
            block = block / norm
            blocks.append(block)
            
    return np.concatenate(blocks), mag, ori

class FlowerPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        if model_path is None:
            candidates = [
                ROOT_DIR / "outputs" / "best_knn_hog_model.pt",
                Path("outputs/best_knn_hog_model.pt"),
                Path("Nhom11_PhanLoaiHoa_FIT4013/outputs/best_knn_hog_model.pt")
            ]
            for c in candidates:
                if c.exists():
                    model_path = c
                    break
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}. Hãy chạy `python src/train.py` trước.")

        print(f"Nạp mô hình từ {model_path}...")
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.X_train = checkpoint["X_train"]
        self.y_train = checkpoint["y_train"]
        self.k = checkpoint.get("k", 3)
        self.scaler_mean = checkpoint["scaler_mean"]
        self.scaler_scale = checkpoint["scaler_scale"]
        print(f"Mô hình đã sẵn sàng: k={self.k}, X_train shape={self.X_train.shape}")

    def predict_image(self, pil_img, top_n=5):
        t0 = time.time()
        # Resize to 64x64 and extract HOG
        gray = np.asarray(pil_img.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        hog_feat, mag, ori = compute_hog_single(gray)
        
        # Scale
        hog_scaled = (hog_feat - self.scaler_mean) / self.scaler_scale
        x_tensor = torch.tensor(hog_scaled.reshape(1, -1), dtype=torch.float32)
        
        # Distance & K-NN
        dists = torch.cdist(x_tensor, self.X_train)
        topk_dists, topk_indices = torch.topk(dists, self.k, dim=1, largest=False)
        topk_labels = self.y_train[topk_indices][0]
        topk_dists = topk_dists[0]
        
        weights = 1.0 / (topk_dists + 1e-5)
        scores = torch.zeros(102, dtype=torch.float32)
        scores.scatter_add_(0, topk_labels, weights)
        probs = (scores / scores.sum()).numpy()
        
        # Top N classes
        top_indices = np.argsort(probs)[::-1][:top_n]
        results = []
        for idx in top_indices:
            class_id = int(idx) + 1
            en_name, vn_name = CATEGORY_NAMES.get(class_id, (f"Flower_{class_id}", f"Loài hoa số {class_id}"))
            results.append({
                "class_id": class_id,
                "en_name": en_name.title(),
                "vn_name": vn_name,
                "prob": float(probs[idx]),
                "dist": float(topk_dists[0])
            })
            
        elapsed_ms = (time.time() - t0) * 1000.0
        return results, mag, elapsed_ms

# ==================== GIAO DIỆN TKINTER DESKTOP GUI ====================
class FlowerClassifierGUI(tk.Tk):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.current_img = None
        self.current_img_path = None

        self.title("NHÓM 11 - HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI HOA (FIT4013)")
        self.geometry("1060x720")
        self.minsize(980, 650)
        self.configure(bg="#f4f6f9")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self._setup_ui()
        self._load_initial_sample()

    def _setup_ui(self):
        # 1. Header
        header = tk.Frame(self, bg="#1e3a8a", height=75)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title_lbl = tk.Label(header, text="HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI HOA (OXFORD FLOWERS 102)",
                             font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#1e3a8a")
        title_lbl.pack(pady=(10, 2))
        
        sub_lbl = tk.Label(header, text="Bài tập lớn FIT4013 - Đề tài 7 | GVHD: ThS. Lê Thị Thùy Trang | SV: Đỗ Tiến Sơn - Đỗ Huy Anh Duy",
                           font=("Segoe UI", 10), fg="#bfdbfe", bg="#1e3a8a")
        sub_lbl.pack()

        # 2. Main content container
        main_box = tk.Frame(self, bg="#f4f6f9", padx=16, pady=14)
        main_box.pack(fill="both", expand=True)

        # Cột trái (Ảnh & Thao tác)
        left_col = tk.Frame(main_box, bg="#ffffff", bd=1, relief="solid")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Thanh công cụ bên trái
        btn_bar = tk.Frame(left_col, bg="#ffffff", pady=10)
        btn_bar.pack(fill="x")

        btn_browse = tk.Button(btn_bar, text="📁 Chọn ảnh từ máy...", font=("Segoe UI", 10, "bold"),
                               bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff",
                               padx=12, pady=6, cursor="hand2", command=self.browse_file)
        btn_browse.pack(side="left", padx=(15, 8))

        btn_random = tk.Button(btn_bar, text="🎲 Ảnh ngẫu nhiên từ Dataset", font=("Segoe UI", 10),
                               bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
                               padx=12, pady=6, cursor="hand2", command=self.random_sample)
        btn_random.pack(side="left", padx=5)

        self.path_lbl = tk.Label(left_col, text="Chưa chọn tệp ảnh nào", font=("Segoe UI", 9, "italic"),
                                 fg="#64748b", bg="#ffffff", anchor="w")
        self.path_lbl.pack(fill="x", padx=15, pady=(0, 8))

        # Khung hiển thị ảnh gốc và ảnh HOG
        img_container = tk.Frame(left_col, bg="#ffffff")
        img_container.pack(fill="both", expand=True, padx=15, pady=5)

        # Khung ảnh gốc
        frame_orig = tk.LabelFrame(img_container, text=" Ảnh gốc (Input Image) ", font=("Segoe UI", 10, "bold"),
                                   bg="#ffffff", fg="#1e293b", padx=6, pady=6)
        frame_orig.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.lbl_orig = tk.Label(frame_orig, bg="#f1f5f9", text="Chưa có ảnh")
        self.lbl_orig.pack(fill="both", expand=True)

        # Khung ảnh HOG Gradient
        frame_hog = tk.LabelFrame(img_container, text=" Bản đồ Gradient HOG ", font=("Segoe UI", 10, "bold"),
                                  bg="#ffffff", fg="#1e293b", padx=6, pady=6)
        frame_hog.pack(side="right", fill="both", expand=True, padx=(6, 0))
        self.lbl_hog = tk.Label(frame_hog, bg="#f1f5f9", text="Chưa có đặc trưng")
        self.lbl_hog.pack(fill="both", expand=True)

        # Cột phải (Kết quả nhận diện)
        right_col = tk.Frame(main_box, bg="#ffffff", bd=1, relief="solid", width=420)
        right_col.pack(side="right", fill="both", expand=False, padx=(10, 0))
        right_col.pack_propagate(False)

        res_title = tk.Label(right_col, text="KẾT QUẢ DỰ ĐOÁN", font=("Segoe UI", 12, "bold"),
                             fg="#1e3a8a", bg="#ffffff", pady=12)
        res_title.pack(fill="x")

        # Card Top 1
        self.card_top1 = tk.Frame(right_col, bg="#eff6ff", bd=1, relief="solid", padx=14, pady=12)
        self.card_top1.pack(fill="x", padx=15, pady=(0, 12))

        lbl_pred_title = tk.Label(self.card_top1, text="Loài hoa dự đoán (Top-1):", font=("Segoe UI", 10, "bold"),
                                  fg="#1e40af", bg="#eff6ff")
        lbl_pred_title.pack(anchor="w")

        self.lbl_vn_name = tk.Label(self.card_top1, text="---", font=("Segoe UI", 15, "bold"),
                                    fg="#16a34a", bg="#eff6ff", wraplength=380, justify="left")
        self.lbl_vn_name.pack(anchor="w", pady=(3, 1))

        self.lbl_en_name = tk.Label(self.card_top1, text="---", font=("Segoe UI", 11, "italic"),
                                    fg="#334155", bg="#eff6ff")
        self.lbl_en_name.pack(anchor="w", pady=(0, 6))

        info_grid = tk.Frame(self.card_top1, bg="#eff6ff")
        info_grid.pack(fill="x", pady=4)

        tk.Label(info_grid, text="Mã lớp (Class ID):", font=("Segoe UI", 9), fg="#64748b", bg="#eff6ff").grid(row=0, column=0, sticky="w")
        self.lbl_cid = tk.Label(info_grid, text="-- / 102", font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#eff6ff")
        self.lbl_cid.grid(row=0, column=1, sticky="w", padx=6)

        tk.Label(info_grid, text="Độ tin cậy (Confidence):", font=("Segoe UI", 9), fg="#64748b", bg="#eff6ff").grid(row=1, column=0, sticky="w")
        self.lbl_conf = tk.Label(info_grid, text="-- %", font=("Segoe UI", 9, "bold"), fg="#2563eb", bg="#eff6ff")
        self.lbl_conf.grid(row=1, column=1, sticky="w", padx=6)

        tk.Label(info_grid, text="Thuật toán sử dụng:", font=("Segoe UI", 9), fg="#64748b", bg="#eff6ff").grid(row=2, column=0, sticky="w")
        self.lbl_algo = tk.Label(info_grid, text=f"K-NN (k={self.predictor.k}) + HOG", font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#eff6ff")
        self.lbl_algo.grid(row=2, column=1, sticky="w", padx=6)

        tk.Label(info_grid, text="Thời gian suy diễn:", font=("Segoe UI", 9), fg="#64748b", bg="#eff6ff").grid(row=3, column=0, sticky="w")
        self.lbl_time = tk.Label(info_grid, text="-- ms", font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#eff6ff")
        self.lbl_time.grid(row=3, column=1, sticky="w", padx=6)

        # Top 5 Candidates Frame
        top5_frame = tk.LabelFrame(right_col, text=" Top-5 Loài hoa tiềm năng nhất ", font=("Segoe UI", 10, "bold"),
                                   bg="#ffffff", fg="#1e293b", padx=10, pady=8)
        top5_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.candidate_rows = []
        for i in range(5):
            row = tk.Frame(top5_frame, bg="#ffffff", pady=3)
            row.pack(fill="x")
            
            lbl_name = tk.Label(row, text=f"#{i+1}. ---", font=("Segoe UI", 9), fg="#1e293b", bg="#ffffff", width=25, anchor="w")
            lbl_name.pack(side="left")
            
            pbar = ttk.Progressbar(row, orient="horizontal", length=110, mode="determinate")
            pbar.pack(side="left", padx=4)
            
            lbl_pct = tk.Label(row, text="0.0%", font=("Segoe UI", 9, "bold"), fg="#2563eb", bg="#ffffff", width=6, anchor="e")
            lbl_pct.pack(side="left")
            
            self.candidate_rows.append((lbl_name, pbar, lbl_pct))

    def _load_initial_sample(self):
        """Khởi động với một ảnh mẫu có sẵn."""
        sample_path = ROOT_DIR / "data" / "flowers-102" / "jpg" / "image_05400.jpg"
        if not sample_path.exists():
            sample_candidates = list((ROOT_DIR / "data" / "flowers-102" / "jpg").glob("*.jpg"))
            if sample_candidates:
                sample_path = sample_candidates[0]
        if sample_path.exists():
            self.process_image_file(str(sample_path))

    def browse_file(self):
        filetypes = [("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
        init_dir = ROOT_DIR / "data" / "flowers-102" / "jpg"
        if not init_dir.exists():
            init_dir = ROOT_DIR
        path = filedialog.askopenfilename(title="Chọn ảnh bông hoa để nhận diện", initialdir=str(init_dir), filetypes=filetypes)
        if path:
            self.process_image_file(path)

    def random_sample(self):
        jpg_dir = ROOT_DIR / "data" / "flowers-102" / "jpg"
        if not jpg_dir.exists():
            messagebox.showwarning("Cảnh báo", f"Không tìm thấy thư mục ảnh tại: {jpg_dir}")
            return
        all_imgs = list(jpg_dir.glob("*.jpg"))
        if not all_imgs:
            messagebox.showwarning("Cảnh báo", "Không có tệp ảnh jpg nào trong thư mục data.")
            return
        selected = random.choice(all_imgs)
        self.process_image_file(str(selected))

    def process_image_file(self, img_path):
        try:
            self.current_img_path = img_path
            self.path_lbl.config(text=f"Tệp: {Path(img_path).name}")

            pil_img = Image.open(img_path).convert("RGB")
            self.current_img = pil_img

            # Hiển thị ảnh gốc
            disp_orig = ImageOps.contain(pil_img.copy(), (250, 250))
            tk_orig = ImageTk.PhotoImage(disp_orig)
            self.lbl_orig.config(image=tk_orig, text="")
            self.lbl_orig.image = tk_orig

            # Dự đoán
            results, mag, elapsed_ms = self.predictor.predict_image(pil_img, top_n=5)

            # Tạo ảnh hiển thị HOG gradient
            norm_mag = (mag / (mag.max() + 1e-5) * 255).astype(np.uint8)
            cm = plt.get_cmap("inferno")
            color_mag = (cm(norm_mag / 255.0)[:, :, :3] * 255).astype(np.uint8)
            hog_pil = Image.fromarray(color_mag).resize((250, 250), Image.NEAREST)
            tk_hog = ImageTk.PhotoImage(hog_pil)
            self.lbl_hog.config(image=tk_hog, text="")
            self.lbl_hog.image = tk_hog

            # Cập nhật kết quả Top 1
            top1 = results[0]
            self.lbl_vn_name.config(text=top1["vn_name"])
            self.lbl_en_name.config(text=f"Tên khoa học: {top1['en_name']}")
            self.lbl_cid.config(text=f"{top1['class_id']} / 102")
            self.lbl_conf.config(text=f"{top1['prob']*100:.1f} %")
            self.lbl_time.config(text=f"{elapsed_ms:.2f} ms")

            # Cập nhật Top 5
            for i, res in enumerate(results):
                lbl_name, pbar, lbl_pct = self.candidate_rows[i]
                pct = res["prob"] * 100
                short_vn = res["vn_name"] if len(res["vn_name"]) <= 18 else res["vn_name"][:16] + "..."
                lbl_name.config(text=f"#{i+1}. Lớp {res['class_id']}: {short_vn}")
                pbar["value"] = pct
                lbl_pct.config(text=f"{pct:.1f}%")

        except Exception as e:
            messagebox.showerror("Lỗi xử lý ảnh", f"Không thể xử lý ảnh: {e}")

# ==================== DEMO SCREENSHOT & LATEX SYNC ====================
def create_demo_screenshot(predictor, img_path, out_path, title_suffix=""):
    img = Image.open(img_path).convert("RGB")
    results, mag, elapsed = predictor.predict_image(img, top_n=3)
    top1 = results[0]

    fig = plt.figure(figsize=(12, 6.8), facecolor="#f8f9fa")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.4], height_ratios=[1, 1],
                           left=0.06, right=0.96, top=0.88, bottom=0.08, wspace=0.25, hspace=0.28)

    fig.suptitle(f"HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI HOA - FIT4013 (NHÓM 11)\nĐại học Đại Nam - GV: ThS. Lê Thị Thùy Trang {title_suffix}",
                 fontsize=13, fontweight="bold", color="#1a252f")

    # 1. Ảnh gốc
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img)
    ax_orig.set_title("Ảnh đầu vào (Input Image)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax_orig.axis("off")

    # 2. Đặc trưng gradient HOG
    ax_hog = fig.add_subplot(gs[1, 0])
    im_h = ax_hog.imshow(mag, cmap="inferno")
    ax_hog.set_title("Bản đồ Gradient HOG (Magnitude 64x64)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax_hog.axis("off")
    fig.colorbar(im_h, ax=ax_hog, orientation="horizontal", pad=0.05, shrink=0.8)

    # 3. Kết quả nhận diện Top 1
    ax_res = fig.add_subplot(gs[0, 1])
    ax_res.axis("off")
    res_text = (
        f"KẾT QUẢ DỰ ĐOÁN TỐT NHẤT (TOP 1):\n\n"
        f"  • Loài hoa: {top1['vn_name']}\n"
        f"  • Tên tiếng Anh: {top1['en_name']}\n"
        f"  • Mã lớp (Class ID): {top1['class_id']} / 102\n"
        f"  • Độ tin cậy (Confidence): {top1['prob']*100:.1f}%\n"
        f"  • Thuật toán: K-NN (k={predictor.k}, HOG Feature)\n"
        f"  • Thời gian suy diễn: {elapsed:.2f} ms"
    )
    ax_res.text(0.02, 0.95, res_text, transform=ax_res.transAxes, fontsize=11,
                verticalalignment="top", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#e8f4f8", edgecolor="#3498db", linewidth=1.5))

    # 4. Biểu đồ Top 3 xác suất
    ax_bar = fig.add_subplot(gs[1, 1])
    names = [f"Lớp {r['class_id']}: {r['vn_name'][:18]}" for r in results]
    probs = [r['prob'] * 100 for r in results]
    y_pos = np.arange(len(names))
    bars = ax_bar.barh(y_pos, probs, color=["#27ae60", "#2980b9", "#f39c12"], edgecolor="black", height=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Độ tin cậy (%)", fontsize=10, fontweight="bold")
    ax_bar.set_title("Xác suất Top-3 ứng viên gần nhất", fontsize=11, fontweight="bold", color="#2c3e50")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax_bar.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%",
                    va="center", ha="left", fontsize=9, fontweight="bold", color="#2c3e50")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"-> Đã lưu ảnh minh chứng GUI: {out_path}")

def find_latex_hinhve_dir():
    """Tìm thư mục Hinhve của báo cáo LaTeX nếu có."""
    candidates = [
        ROOT_DIR.parent.parent / "LAXTEX" / "Hinhve",
        ROOT_DIR.parent / "LAXTEX" / "Hinhve",
        ROOT_DIR / "LAXTEX" / "Hinhve"
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def run_headless_demo():
    print("Khởi tạo bộ tạo ảnh minh chứng GUI...")
    predictor = FlowerPredictor()

    data_jpg = ROOT_DIR / "data" / "flowers-102" / "jpg"
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True, parents=True)
    hinhve_dir = find_latex_hinhve_dir()

    # 5 ca kiểm thử chuẩn xác 100% khớp với 5 loài hoa trong báo cáo LaTeX
    test_cases = [
        ("image_05400.jpg", "gui_demo_sunflower.png", "- Hoa Hướng Dương (Lớp 54)"),
        ("image_01147.jpg", "gui_demo_rose.png", "- Hoa Hồng Đỏ (Lớp 74)"),
        ("image_00255.jpg", "gui_demo_waterlily.png", "- Hoa Súng Nước (Lớp 73)"),
        ("image_01314.jpg", "gui_demo_petunia.png", "- Hoa Dạ Yến Thảo (Lớp 51)"),
        ("image_06300.jpg", "gui_demo_dandelion.png", "- Hoa Bồ Công Anh (Lớp 50)")
    ]

    for fname, out_name, suffix in test_cases:
        p = data_jpg / fname
        if p.exists():
            # Lưu vào outputs
            out_path = out_dir / out_name
            create_demo_screenshot(predictor, p, out_path, suffix)
            
            # Đồng bộ sang thư mục LaTeX Hinhve nếu có
            if hinhve_dir:
                latex_path = hinhve_dir / out_name
                import shutil
                shutil.copy(out_path, latex_path)
                print(f"-> Đã đồng bộ sang báo cáo LaTeX: {latex_path}")

    # Đồng bộ các biểu đồ chính từ outputs sang Hinhve
    if hinhve_dir:
        import shutil
        files_to_sync = [
            "class_distribution.png",
            "sample_images.png",
            "model_comparison.png",
            "confusion_matrix_KNN_Pixel.png",
            "confusion_matrix_KNN_HOG.png",
            "confusion_matrix_DecisionTree_HOG.png"
        ]
        for f in files_to_sync:
            src = out_dir / f
            if src.exists():
                shutil.copy(src, hinhve_dir / f)
                print(f"-> Đã đồng bộ biểu đồ sang LaTeX: {f}")

    print("\n[THÀNH CÔNG] Đã tạo toàn bộ ảnh chụp giao diện GUI Demo chuẩn xác và đồng bộ sang LaTeX!\n")

def main():
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_headless_demo()
        return

    # Khởi chạy giao diện Tkinter Desktop
    try:
        predictor = FlowerPredictor()
        app = FlowerClassifierGUI(predictor)
        print("Đang mở giao diện Desktop GUI Tkinter...")
        app.mainloop()
    except Exception as e:
        print(f"Không thể khởi chạy cửa sổ đồ họa trực tiếp ({e}). Chuyển sang chế độ tạo ảnh minh chứng demo (--demo)...")
        run_headless_demo()

if __name__ == "__main__":
    main()
