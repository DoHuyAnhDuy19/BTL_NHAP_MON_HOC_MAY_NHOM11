"""
GIAO DIỆN WEB DEMO NHẬN DIỆN VÀ PHÂN LOẠI HOA (FIT4013)
Nhóm 11 - Đề tài 7: Phân loại hoa Oxford Flowers 102
Trường Đại học Đại Nam - Khoa Công nghệ Thông tin
Chạy trực tiếp trên trình duyệt Web (Chrome, Edge, Firefox, Cốc Cốc)
"""
import os, sys, time, json, random, base64, io, webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Nhập logic mô hình và từ điển từ gui_app
FILE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FILE_DIR.parent))

from src.gui_app import FlowerPredictor, CATEGORY_NAMES, ROOT_DIR

predictor = FlowerPredictor()

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nhóm 11 - Nhận diện và Phân loại Hoa Oxford 102</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: #f1f5f9; color: #1e293b; line-height: 1.5; }
    header { background: linear-gradient(135deg, #1e3a8a, #2563eb); color: white; padding: 22px 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    header h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
    header p { font-size: 0.9rem; opacity: 0.9; }
    .container { max-width: 1200px; margin: 24px auto; padding: 0 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 860px) { .container { grid-template-columns: 1fr; } }
    .card { background: white; border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; }
    .card h2 { font-size: 1.15rem; font-weight: 700; color: #1e3a8a; margin-bottom: 16px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }
    .btn-group { display: flex; gap: 12px; margin-bottom: 16px; }
    button, label.btn { cursor: pointer; padding: 10px 18px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; border: none; transition: all 0.2s; display: inline-flex; align-items: center; justify-content: center; gap: 6px; }
    .btn-primary { background-color: #2563eb; color: white; }
    .btn-primary:hover { background-color: #1d4ed8; }
    .btn-success { background-color: #059669; color: white; }
    .btn-success:hover { background-color: #047857; }
    input[type="file"] { display: none; }
    .images-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
    .img-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; text-align: center; }
    .img-box h3 { font-size: 0.85rem; color: #64748b; margin-bottom: 8px; font-weight: 600; }
    .img-box img { max-width: 100%; max-height: 240px; object-fit: contain; border-radius: 6px; }
    .top1-box { background: #eff6ff; border: 2px solid #3b82f6; border-radius: 10px; padding: 18px; margin-bottom: 18px; }
    .top1-badge { display: inline-block; background-color: #2563eb; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    .top1-title { font-size: 1.45rem; font-weight: 800; color: #16a34a; margin-bottom: 2px; }
    .top1-sub { font-size: 1rem; color: #475569; font-style: italic; margin-bottom: 12px; }
    .meta-row { display: flex; justify-content: space-between; font-size: 0.9rem; padding: 4px 0; border-top: 1px dashed #cbd5e1; }
    .meta-label { color: #64748b; }
    .meta-val { font-weight: 700; color: #0f172a; }
    .candidates-list { display: flex; flex-direction: column; gap: 10px; }
    .candidate-item { display: flex; align-items: center; justify-content: space-between; font-size: 0.9rem; gap: 10px; }
    .cand-name { width: 170px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 600; }
    .cand-bar-wrap { flex: 1; height: 10px; background-color: #e2e8f0; border-radius: 6px; overflow: hidden; }
    .cand-bar { height: 100%; background: linear-gradient(90deg, #2563eb, #10b981); border-radius: 6px; transition: width 0.4s; }
    .cand-pct { width: 55px; text-align: right; font-weight: 700; color: #2563eb; }
    .loading { display: none; text-align: center; padding: 15px; color: #2563eb; font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <h1>HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI HOA (OXFORD FLOWERS 102)</h1>
    <p>BTL Nhập môn Học máy (FIT4013) – Trường Đại học Đại Nam | Nhóm 11 – Đề tài 7 | GVHD: ThS. Lê Thị Thùy Trang</p>
  </header>

  <div class="container">
    <!-- Cột trái: Tải ảnh & Hiển thị -->
    <div class="card">
      <h2>1. Chọn ảnh đầu vào</h2>
      <div class="btn-group">
        <label class="btn btn-primary">
          📁 Chọn ảnh từ máy...
          <input type="file" id="fileInput" accept="image/*" onchange="uploadImage(event)">
        </label>
        <button class="btn btn-success" onclick="fetchRandom()">
          🎲 Ảnh ngẫu nhiên từ Dataset
        </button>
      </div>

      <div id="loading" class="loading">Đang trích xuất đặc trưng HOG và suy luận...</div>

      <div class="images-grid">
        <div class="img-box">
          <h3>Ảnh gốc (Input Image)</h3>
          <img id="imgOrig" src="" alt="Chưa có ảnh" style="display:none;">
          <div id="placeholderOrig" style="padding: 60px 0; color: #94a3b8;">Chưa chọn ảnh</div>
        </div>
        <div class="img-box">
          <h3>Bản đồ Gradient HOG</h3>
          <img id="imgHog" src="" alt="Chưa có đặc trưng" style="display:none;">
          <div id="placeholderHog" style="padding: 60px 0; color: #94a3b8;">Chưa có đặc trưng</div>
        </div>
      </div>
    </div>

    <!-- Cột phải: Kết quả nhận diện -->
    <div class="card">
      <h2>2. Kết quả phân loại</h2>
      
      <!-- Card Top 1 -->
      <div class="top1-box" id="top1Card" style="display:none;">
        <span class="top1-badge">Dự đoán tốt nhất (Top 1)</span>
        <div class="top1-title" id="resVn">---</div>
        <div class="top1-sub" id="resEn">---</div>
        
        <div class="meta-row">
          <span class="meta-label">Mã lớp (Class ID):</span>
          <span class="meta-val" id="resCid">-- / 102</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Độ tin cậy (Confidence):</span>
          <span class="meta-val" id="resConf" style="color: #2563eb;">-- %</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Thuật toán sử dụng:</span>
          <span class="meta-val">K-NN (k=3, HOG 64x64)</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Thời gian suy diễn:</span>
          <span class="meta-val" id="resTime">-- ms</span>
        </div>
      </div>

      <!-- Top 5 -->
      <h3 style="font-size: 0.95rem; color: #475569; margin-bottom: 12px; font-weight: 700;">Top 5 Loài hoa tiềm năng nhất:</h3>
      <div class="candidates-list" id="candList">
        <div style="color: #94a3b8; font-style: italic;">Chọn một ảnh để xem kết quả.</div>
      </div>
    </div>
  </div>

  <script>
    function renderResult(data) {
      document.getElementById('loading').style.display = 'none';
      if (!data || !data.results || data.results.length === 0) return;

      const top1 = data.results[0];
      document.getElementById('top1Card').style.display = 'block';
      document.getElementById('resVn').innerText = top1.vn_name;
      document.getElementById('resEn').innerText = 'Tên tiếng Anh: ' + top1.en_name;
      document.getElementById('resCid').innerText = top1.class_id + ' / 102';
      document.getElementById('resConf').innerText = (top1.prob * 100).toFixed(1) + ' %';
      document.getElementById('resTime').innerText = data.elapsed_ms.toFixed(2) + ' ms';

      // Ảnh
      if (data.orig_b64) {
        const orig = document.getElementById('imgOrig');
        orig.src = 'data:image/jpeg;base64,' + data.orig_b64;
        orig.style.display = 'block';
        document.getElementById('placeholderOrig').style.display = 'none';
      }
      if (data.hog_b64) {
        const hog = document.getElementById('imgHog');
        hog.src = 'data:image/png;base64,' + data.hog_b64;
        hog.style.display = 'block';
        document.getElementById('placeholderHog').style.display = 'none';
      }

      // Danh sách Top 5
      const candList = document.getElementById('candList');
      candList.innerHTML = '';
      data.results.forEach((r, idx) => {
        const pct = (r.prob * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'candidate-item';
        item.innerHTML = `
          <div class="cand-name" title="${r.vn_name}">#${idx+1}. Lớp ${r.class_id}: ${r.vn_name}</div>
          <div class="cand-bar-wrap"><div class="cand-bar" style="width: ${pct}%"></div></div>
          <div class="cand-pct">${pct}%</div>
        `;
        candList.appendChild(item);
      });
    }

    async function fetchRandom() {
      document.getElementById('loading').style.display = 'block';
      try {
        const res = await fetch('/api/random');
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert('Lỗi khi gọi API: ' + err);
        document.getElementById('loading').style.display = 'none';
      }
    }

    function uploadImage(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = async function(e) {
        const b64 = e.target.result.split(',')[1];
        document.getElementById('loading').style.display = 'block';
        
        // Hiện ảnh gốc ngay
        const orig = document.getElementById('imgOrig');
        orig.src = e.target.result;
        orig.style.display = 'block';
        document.getElementById('placeholderOrig').style.display = 'none';

        try {
          const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: b64 })
          });
          const data = await res.json();
          renderResult(data);
        } catch (err) {
          alert('Lỗi suy diễn: ' + err);
          document.getElementById('loading').style.display = 'none';
        }
      };
      reader.readAsDataURL(file);
    }

    // Tự động load ảnh mẫu khi mở trang
    window.addEventListener('load', fetchRandom);
  </script>
</body>
</html>
"""

def make_hog_image_base64(mag):
    norm_mag = (mag / (mag.max() + 1e-5) * 255).astype(np.uint8)
    cm = plt.get_cmap("inferno")
    color_mag = (cm(norm_mag / 255.0)[:, :, :3] * 255).astype(np.uint8)
    hog_pil = Image.fromarray(color_mag).resize((250, 250), Image.NEAREST)
    buf = io.BytesIO()
    hog_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class FlowerWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Tắt bớt log request ra console

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/api/random":
            jpg_dir = ROOT_DIR / "data" / "flowers-102" / "jpg"
            imgs = list(jpg_dir.glob("*.jpg"))
            if not imgs:
                self.send_error(404, "Không tìm thấy ảnh nào trong data.")
                return
            chosen = random.choice(imgs)
            pil_img = Image.open(chosen).convert("RGB")
            
            # Encode ảnh gốc
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            orig_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Dự đoán
            results, mag, elapsed = predictor.predict_image(pil_img, top_n=5)
            hog_b64 = make_hog_image_base64(mag)

            payload = {
                "results": results,
                "elapsed_ms": elapsed,
                "orig_b64": orig_b64,
                "hog_b64": hog_b64,
                "file_name": chosen.name
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            b64_str = data.get("image_b64", "")
            img_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            results, mag, elapsed = predictor.predict_image(pil_img, top_n=5)
            hog_b64 = make_hog_image_base64(mag)

            payload = {
                "results": results,
                "elapsed_ms": elapsed,
                "hog_b64": hog_b64
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

def main():
    port = 8501
    server_address = ("127.0.0.1", port)
    
    # Thử bind port
    try:
        httpd = HTTPServer(server_address, FlowerWebHandler)
    except OSError:
        port = 8080
        server_address = ("127.0.0.1", port)
        httpd = HTTPServer(server_address, FlowerWebHandler)

    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f" 🌐 MÁY CHỦ GIAO DIỆN WEB DEMO ĐANG CHẠY TẠI: {url}")
    print(f"=======================================================\n")
    
    # Mở tự động trình duyệt web (Chrome, Edge...)
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng máy chủ web.")
        httpd.server_close()

if __name__ == "__main__":
    main()
