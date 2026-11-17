"""
NHÓM 11 - FIT4013 - ĐỀ TÀI 7: PHÂN LOẠI HOA
Oxford Flowers 102 | Pixel + HOG | KNN + Decision Tree
Trường Đại học Đại Nam - Khoa Công nghệ Thông tin
Giảng viên hướng dẫn: ThS. Lê Thị Thùy Trang
"""
from pathlib import Path
import time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision.datasets import Flowers102

warnings.filterwarnings("ignore")

# Xác định thư mục gốc dự án một cách an toàn và linh hoạt
FILE_DIR = Path(__file__).resolve().parent
ROOT = FILE_DIR.parent if (FILE_DIR.name == "src") else FILE_DIR
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)
RANDOM_STATE = 42
PIXEL_SIZE = (32, 32)
HOG_SIZE = (64, 64)

# ==================== DATASET LOADING ====================
def load_split(split):
    try:
        ds = Flowers102(root=str(DATA_DIR), split=split, download=False)
    except Exception:
        print(f"-> Chưa có dữ liệu cục bộ cho tập {split}, đang tự động tải...")
        ds = Flowers102(root=str(DATA_DIR), split=split, download=True)
    images, labels = [], []
    for img, label in ds:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        images.append(img.copy())
        labels.append(int(label))
    return images, np.asarray(labels, dtype=np.int64)

def summarize_split(name, labels):
    values, counts = np.unique(labels, return_counts=True)
    return {
        "split": name,
        "images": len(labels),
        "classes": len(values),
        "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max())
    }

# ==================== FEATURE EXTRACTION ====================
def pixel_features(images):
    feats = []
    for img in images:
        arr = np.asarray(img.resize(PIXEL_SIZE), dtype=np.float32) / 255.0
        feats.append(arr.reshape(-1))
    return np.asarray(feats, dtype=np.float32)

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
            
    return np.concatenate(blocks)

def hog_features(images):
    feats = []
    for img in images:
        gray = np.asarray(img.convert("L").resize(HOG_SIZE), dtype=np.float32) / 255.0
        f = compute_hog_single(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        feats.append(f)
    return np.asarray(feats, dtype=np.float32)

# ==================== STANDARDIZATION ====================
class Scaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ==================== CLASSIFIERS ====================
class TorchKNN:
    def __init__(self, n_neighbors=5, weights="distance"):
        self.k = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.n_classes_ = 102

    def fit(self, X, y):
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)
        self.n_classes_ = int(np.max(y) + 1)
        return self

    def predict_proba(self, X):
        X_test = torch.tensor(X, dtype=torch.float32)
        probas = []
        chunk_size = 500
        for i in range(0, len(X_test), chunk_size):
            chunk = X_test[i:i+chunk_size]
            dists = torch.cdist(chunk, self.X_train)
            topk_dists, topk_indices = torch.topk(dists, self.k, dim=1, largest=False)
            topk_labels = self.y_train[topk_indices]
            
            if self.weights == "distance":
                weights = 1.0 / (topk_dists + 1e-5)
            else:
                weights = torch.ones_like(topk_dists)

            chunk_probs = torch.zeros((len(chunk), self.n_classes_), dtype=torch.float32)
            for j in range(len(chunk)):
                chunk_probs[j].scatter_add_(0, topk_labels[j], weights[j])
                s = chunk_probs[j].sum()
                if s > 0:
                    chunk_probs[j] /= s
            probas.append(chunk_probs.numpy())
        return np.vstack(probas)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, probas=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probas = probas

    @property
    def is_leaf(self):
        return self.value is not None

class CARTDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=6, n_classes=102, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        counts = np.bincount(y, minlength=self.n_classes)
        leaf_val = int(np.argmax(counts))
        leaf_prob = counts / n_samples

        if (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1 or n_samples < self.min_samples_split:
            return Node(value=leaf_val, probas=leaf_prob)

        # Evaluate candidate features (subsample for efficiency)
        n_candidates = min(120, n_features)
        feat_idxs = np.random.choice(n_features, size=n_candidates, replace=False)
        best_feat, best_thresh = None, None
        best_gain = -1.0
        current_gini = self._gini(y)

        for feat in feat_idxs:
            col = X[:, feat]
            thresholds = np.percentile(col, [20, 35, 50, 65, 80])
            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                nl, nr = np.sum(left_mask), np.sum(right_mask)
                if nl < 2 or nr < 2:
                    continue
                gini_split = (nl / n_samples) * self._gini(y[left_mask]) + (nr / n_samples) * self._gini(y[right_mask])
                gain = current_gini - gini_split
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 0 or best_feat is None:
            return Node(value=leaf_val, probas=leaf_prob)

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _predict_row(self, node, x):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def _predict_row_proba(self, node, x):
        if node.is_leaf:
            return node.probas
        if x[node.feature] <= node.threshold:
            return self._predict_row_proba(node.left, x)
        return self._predict_row_proba(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_row_proba(self.root, x) for x in X])

# ==================== EVALUATION METRICS ====================
def calc_metrics(y_true, y_pred, n_classes=102):
    acc = float(np.mean(y_true == y_pred))
    cm = np.bincount(n_classes * y_true + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    
    precisions = []
    recalls = []
    f1s = []
    class_report_rows = []

    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = np.sum(cm[c, :])

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        class_report_rows.append({
            "class": c + 1,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1-score": round(f, 4),
            "support": int(support)
        })

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f = float(np.mean(f1s))

    return {
        "accuracy": acc,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f,
        "confusion_matrix": cm,
        "class_report": pd.DataFrame(class_report_rows)
    }

# ==================== MAIN PIPELINE ====================
def main():
    print("=========================================================")
    print(" BÀI TẬP LỚN FIT4013: PHÂN LOẠI HOA (OXFORD FLOWERS 102)")
    print(" Nhóm 11 - Đề tài 7 | ThS. Lê Thị Thùy Trang")
    print("=========================================================\n")

    # 1) LOAD DATA
    print("1) NẠP DỮ LIỆU TỪ OXFORD FLOWERS 102...")
    t0 = time.time()
    train_imgs, y_train = load_split("train")
    val_imgs, y_val = load_split("val")
    test_imgs, y_test = load_split("test")
    print(f"-> Nạp xong trong {round(time.time() - t0, 2)}s.")
    print(f"-> Train: {len(y_train)} ảnh | Val: {len(y_val)} ảnh | Test: {len(y_test)} ảnh.\n")

    # Lưu thống kê EDA
    split_summary = pd.DataFrame([
        summarize_split("train", y_train),
        summarize_split("validation", y_val),
        summarize_split("test", y_test)
    ])
    split_summary.to_csv(OUT / "dataset_summary.csv", index=False)
    print("--- BẢNG THỐNG KÊ CẤU TRÚC SPLIT ---")
    print(split_summary.to_string(index=False), "\n")

    # Biểu đồ phân phối nhãn
    all_labels = np.concatenate([y_train, y_val, y_test])
    counts = pd.Series(all_labels + 1).value_counts().sort_index()
    plt.figure(figsize=(16, 4))
    counts.plot(kind="bar", width=0.85, color="steelblue")
    plt.title("Phân phối số lượng ảnh trên 102 lớp hoa (Oxford Flowers 102)", fontsize=13, fontweight="bold")
    plt.xlabel("Mã lớp hoa (1-102)"); plt.ylabel("Số lượng mẫu")
    plt.tight_layout(); plt.savefig(OUT / "class_distribution.png", dpi=200); plt.close()

    # Lưu mẫu ảnh EDA
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, idx in zip(axes.ravel(), np.linspace(0, len(train_imgs) - 1, 10, dtype=int)):
        ax.imshow(train_imgs[idx])
        ax.set_title(f"Lớp: {y_train[idx] + 1}", fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Một số ảnh mẫu đại diện trong tập huấn luyện (Train Set)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT / "sample_images.png", dpi=200); plt.close()

    # 2) TRÍCH XUẤT ĐẶC TRƯNG
    print("2) TRÍCH XUẤT ĐẶC TRƯNG (PIXEL & HOG)...")
    t0 = time.time()
    Xtr_pix = pixel_features(train_imgs)
    Xv_pix = pixel_features(val_imgs)
    Xte_pix = pixel_features(test_imgs)
    print(f"-> Đặc trưng Pixel (32x32 RGB): Train {Xtr_pix.shape}, Val {Xv_pix.shape}, Test {Xte_pix.shape}")

    Xtr_hog = hog_features(train_imgs)
    Xv_hog = hog_features(val_imgs)
    Xte_hog = hog_features(test_imgs)
    print(f"-> Đặc trưng HOG (64x64): Train {Xtr_hog.shape}, Val {Xv_hog.shape}, Test {Xte_hog.shape}")
    print(f"-> Thời gian trích xuất: {round(time.time() - t0, 2)}s.\n")

    # 3) CHỌN SIÊU THAM SỐ TRÊN VALIDATION SET (CHỐNG DATA LEAKAGE)
    print("3) THỬ NGHIỆM VÀ TÌM THAM SỐ TRÊN TẬP VALIDATION...")
    candidates = []

    # Scaler fit duy nhất trên train
    scaler_pix = Scaler().fit(Xtr_pix)
    Xtr_pix_sc = scaler_pix.transform(Xtr_pix)
    Xv_pix_sc = scaler_pix.transform(Xv_pix)

    scaler_hog = Scaler().fit(Xtr_hog)
    Xtr_hog_sc = scaler_hog.transform(Xtr_hog)
    Xv_hog_sc = scaler_hog.transform(Xv_hog)

    # Họ 1: KNN Pixel (Baseline)
    for k in [1, 3, 5, 7]:
        knn = TorchKNN(n_neighbors=k, weights="distance")
        knn.fit(Xtr_pix_sc, y_train)
        pred = knn.predict(Xv_pix_sc)
        m = calc_metrics(y_val, pred)
        candidates.append(("KNN_Pixel", k, m["f1_macro"], m["accuracy"], "pixel"))
        print(f"   [Baseline] KNN Pixel (k={k}): Val Acc = {m['accuracy']*100:.2f}%, Val F1 = {m['f1_macro']:.4f}")

    # Họ 2: KNN HOG
    for k in [1, 3, 5, 7]:
        knn = TorchKNN(n_neighbors=k, weights="distance")
        knn.fit(Xtr_hog_sc, y_train)
        pred = knn.predict(Xv_hog_sc)
        m = calc_metrics(y_val, pred)
        candidates.append(("KNN_HOG", k, m["f1_macro"], m["accuracy"], "hog"))
        print(f"   [Đặc trưng HOG] KNN (k={k}): Val Acc = {m['accuracy']*100:.2f}%, Val F1 = {m['f1_macro']:.4f}")

    # Họ 3: Decision Tree HOG
    for depth in [10, 20, None]:
        dt = CARTDecisionTree(max_depth=depth, random_state=RANDOM_STATE)
        dt.fit(Xtr_hog_sc, y_train)
        pred = dt.predict(Xv_hog_sc)
        m = calc_metrics(y_val, pred)
        candidates.append(("DecisionTree_HOG", str(depth), m["f1_macro"], m["accuracy"], "hog"))
        print(f"   [Đặc trưng HOG] Cây quyết định (depth={depth}): Val Acc = {m['accuracy']*100:.2f}%, Val F1 = {m['f1_macro']:.4f}")

    val_df = pd.DataFrame([
        {"experiment": c[0], "parameter": str(c[1]), "validation_accuracy": c[3], "validation_macro_f1": c[2]}
        for c in candidates
    ]).sort_values("validation_macro_f1", ascending=False)
    val_df.to_csv(OUT / "validation_model_selection.csv", index=False)
    print("\n--- BẢNG KẾT QUẢ TÌM THAM SỐ TRÊN TẬP VALIDATION ---")
    print(val_df.to_string(index=False), "\n")

    # Chọn cấu hình tốt nhất của từng họ mô hình
    selected = {}
    for fam in ["KNN_Pixel", "KNN_HOG", "DecisionTree_HOG"]:
        fam_list = [c for c in candidates if c[0] == fam]
        selected[fam] = max(fam_list, key=lambda x: x[2])

    print("--- CẤU HÌNH TỐI ƯU ĐƯỢC CHỌN ---")
    for fam, s in selected.items():
        print(f"   * {fam}: tham số = {s[1]} (Val F1 = {s[2]:.4f})")
    print()

    # 4) HUẤN LUYỆN CUỐI CÙNG TRÊN (TRAIN + VAL) VÀ ĐÁNH GIÁ TRÊN TEST
    print("4) HUẤN LUYỆN LẠI VÀ ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (6.149 ẢNH)...")
    y_fit = np.concatenate([y_train, y_val])
    X_fit_pix = np.concatenate([Xtr_pix, Xv_pix])
    X_fit_hog = np.concatenate([Xtr_hog, Xv_hog])

    # Fit scaler trên (train+val)
    sc_pix_final = Scaler().fit(X_fit_pix)
    X_fit_pix_sc = sc_pix_final.transform(X_fit_pix)
    X_te_pix_sc = sc_pix_final.transform(Xte_pix)

    sc_hog_final = Scaler().fit(X_fit_hog)
    X_fit_hog_sc = sc_hog_final.transform(X_fit_hog)
    X_te_hog_sc = sc_hog_final.transform(Xte_hog)

    test_rows = []
    saved_models = {}

    for fam, (name, param, val_f1, _, feat_type) in selected.items():
        print(f"\n--> Đang huấn luyện và đánh giá: {fam} (tham số {param})...")
        t_start = time.time()
        
        if fam == "KNN_Pixel":
            model = TorchKNN(n_neighbors=int(param), weights="distance")
            model.fit(X_fit_pix_sc, y_fit)
            fit_time = time.time() - t_start
            
            t_pred = time.time()
            y_pred = model.predict(X_te_pix_sc)
            pred_time = time.time() - t_pred
            saved_models[fam] = (model, sc_pix_final, "pixel")

        elif fam == "KNN_HOG":
            model = TorchKNN(n_neighbors=int(param), weights="distance")
            model.fit(X_fit_hog_sc, y_fit)
            fit_time = time.time() - t_start
            
            t_pred = time.time()
            y_pred = model.predict(X_te_hog_sc)
            pred_time = time.time() - t_pred
            saved_models[fam] = (model, sc_hog_final, "hog")

        else: # DecisionTree_HOG
            depth_val = None if param == "None" else int(param)
            model = CARTDecisionTree(max_depth=depth_val, random_state=RANDOM_STATE)
            model.fit(X_fit_hog_sc, y_fit)
            fit_time = time.time() - t_start
            
            t_pred = time.time()
            y_pred = model.predict(X_te_hog_sc)
            pred_time = time.time() - t_pred
            saved_models[fam] = (model, sc_hog_final, "hog")

        # Tính toán metric
        m = calc_metrics(y_test, y_pred)
        row = {
            "experiment": fam,
            "feature": feat_type,
            "selected_parameter": str(param),
            "validation_macro_f1": round(val_f1, 4),
            "accuracy": round(m["accuracy"], 4),
            "precision_macro": round(m["precision_macro"], 4),
            "recall_macro": round(m["recall_macro"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "fit_seconds": round(fit_time, 2),
            "predict_seconds": round(pred_time, 2)
        }
        test_rows.append(row)
        print(f"    Kết quả Test: Accuracy = {m['accuracy']*100:.2f}%, F1-macro = {m['f1_macro']:.4f}, Fit: {fit_time:.2f}s, Pred: {pred_time:.2f}s")

        # Lưu báo cáo chi tiết từng lớp
        m["class_report"].to_csv(OUT / f"classification_report_{fam}.csv", index=False)

        # Vẽ và lưu Confusion Matrix
        plt.figure(figsize=(14, 12))
        sns.heatmap(m["confusion_matrix"], cmap="Blues", cbar=True)
        plt.title(f"Confusion Matrix (102x102) - {fam}", fontsize=14, fontweight="bold")
        plt.xlabel("Nhãn dự đoán (Predicted Class)", fontsize=11)
        plt.ylabel("Nhãn thực tế (True Class)", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT / f"confusion_matrix_{fam}.png", dpi=200)
        plt.close()

        # Lưu danh sách mẫu dự đoán sai
        wrong_idx = np.where(y_pred != y_test)[0]
        err_df = pd.DataFrame({
            "test_sample_index": wrong_idx,
            "true_class": y_test[wrong_idx] + 1,
            "predicted_class": y_pred[wrong_idx] + 1
        })
        err_df.to_csv(OUT / f"errors_{fam}.csv", index=False)

        # Phân tích các cặp nhầm lẫn phổ biến nhất
        pair_df = pd.DataFrame({"true": y_test[wrong_idx] + 1, "pred": y_pred[wrong_idx] + 1})
        if len(pair_df) > 0:
            top_pairs = (pair_df.value_counts().rename("count").reset_index()
                         .sort_values("count", ascending=False).head(20))
            top_pairs.to_csv(OUT / f"top_confusions_{fam}.csv", index=False)

    # Lưu bảng tổng hợp kết quả cuối cùng
    metrics_df = pd.DataFrame(test_rows).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    print("\n================== BẢNG TỔNG KẾT KẾT QUẢ ĐỊNH LƯỢNG TRÊN TẬP TEST ==================")
    print(metrics_df.to_string(index=False))
    print("===================================================================================\n")

    # Vẽ biểu đồ so sánh mô hình
    ax = metrics_df.set_index("experiment")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(
        kind="bar", figsize=(11, 5), ylim=(0, 1), colormap="viridis", edgecolor="black", width=0.7
    )
    plt.title("So sánh hiệu năng giữa các mô hình trên tập kiểm thử (Test Set)", fontsize=13, fontweight="bold")
    plt.ylabel("Điểm số (Score)", fontsize=11)
    plt.xticks(rotation=15, fontsize=10, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=200)
    plt.close()

    # Lưu config
    metadata = {
        "dataset": "Oxford Flowers 102",
        "random_state": RANDOM_STATE,
        "pixel_size": PIXEL_SIZE,
        "hog_size": HOG_SIZE,
        "split": "Train 1020, Val 1020, Test 6149",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    (OUT / "run_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Lưu trọng số mô hình tốt nhất (KNN_HOG)
    best_knn = saved_models["KNN_HOG"][0]
    best_scaler = saved_models["KNN_HOG"][1]
    torch.save({
        "X_train": best_knn.X_train,
        "y_train": best_knn.y_train,
        "k": best_knn.k,
        "weights": best_knn.weights,
        "scaler_mean": best_scaler.mean,
        "scaler_scale": best_scaler.scale
    }, OUT / "best_knn_hog_model.pt")

    print(f"\n[HOÀN THÀNH] Toàn bộ dữ liệu, biểu đồ và mô hình đã được lưu tại: {OUT.resolve()}")

    # Tự động đồng bộ các biểu đồ sang thư mục Hinhve của báo cáo LaTeX (nếu có)
    latex_candidates = [
        ROOT.parent.parent / "LAXTEX" / "Hinhve",
        ROOT.parent / "LAXTEX" / "Hinhve",
        ROOT / "LAXTEX" / "Hinhve"
    ]
    hinhve_dir = None
    for cand in latex_candidates:
        if cand.exists():
            hinhve_dir = cand
            break
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
            src = OUT / f
            if src.exists():
                shutil.copy(src, hinhve_dir / f)
        print(f"-> Đã tự động đồng bộ {len(files_to_sync)} biểu đồ sang báo cáo LaTeX tại: {hinhve_dir}\n")
    else:
        print()

if __name__ == "__main__":
    main()
