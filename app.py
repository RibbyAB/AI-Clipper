#!/usr/bin/env python3
"""
AI Clipper - Web UI
====================
Antarmuka web lokal buat clipper.py. Jalankan:

    python web_app.py

lalu buka http://127.0.0.1:5000 di browser (biasanya kebuka otomatis).
"""

import io
import re
import json
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from pathlib import Path

# Command Prompt Windows sering pakai encoding lama (cp1252/charmap) yang
# tidak dukung banyak karakter unicode (mis. tanda hubung khusus di judul
# video). Alihkan stdout & stderr ke UTF-8 di awal, sebelum apa pun lain
# jalan, supaya print() dari mana pun (kode kita, Flask/Werkzeug, dll)
# tidak bikin proses crash gara-gara satu karakter yang tidak dikenali.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from flask import Flask, request, jsonify, send_from_directory, render_template, send_file

import clipper

app = Flask(__name__)

BASE_OUTPUT_DIR = Path("output")
WATERMARK_DIR = Path("watermark")
WATERMARK_PATH = WATERMARK_DIR / "logo.png"

# ---------------------------------------------------------------------------
# State global sederhana (aplikasi ini dipakai 1 orang di 1 browser lokal).
# Mendukung QUEUE: beberapa link diproses berurutan, tiap video jadi satu
# "job" dengan hasil klipnya masing-masing.
# ---------------------------------------------------------------------------

state = {
    "running": False,
    "done": False,
    "error": None,
    "log": [],
    "jobs": [],          # daftar {title, out_dir (nama folder saja), results: [...]}
    "queue_total": 0,
    "queue_index": 0,    # video ke berapa yang lagi diproses (1-based)
}
state_lock = threading.Lock()

MAX_LOG_LINES = 400


def _trim_log():
    if len(state["log"]) > MAX_LOG_LINES:
        state["log"] = state["log"][-MAX_LOG_LINES:]


def log(msg: str):
    with state_lock:
        state["log"].append(msg)
        _trim_log()
    try:
        print(msg, file=sys.__stdout__)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), file=sys.__stdout__)


class LogWriter(io.TextIOBase):
    """Menangkap semua print() dari clipper.py supaya muncul di web UI."""
    def write(self, s):
        s = s.strip("\n")
        if s:
            with state_lock:
                state["log"].append(s)
                _trim_log()
        return len(s)


def process_one_video(url: str, num_clips: int, subtitles: bool, aspect_ratios: list[str] | None = None, font_name: str = "Arial Black", use_watermark: bool = False, speaker_colors: bool = False, font_color: str | None = None, font_size_pct: float | None = None, pos_x_pct: float | None = None, pos_y_pct: float | None = None, smart_crop: bool = False, wm_size_pct: float | None = None, wm_pos_x_pct: float | None = None, wm_pos_y_pct: float | None = None):
    """Proses satu video sampai selesai, hasil ditambahkan sebagai satu job
    baru ke state['jobs']. Error di satu video TIDAK menghentikan video
    lain di antrian -- dicatat di log lalu lanjut ke berikutnya."""
    aspect_ratios = aspect_ratios or ["9:16"]
    settings = {
        "subtitles": subtitles, "font_name": font_name, "use_watermark": use_watermark,
        "speaker_colors": speaker_colors, "font_color": font_color,
        "font_size_pct": font_size_pct, "pos_x_pct": pos_x_pct, "pos_y_pct": pos_y_pct,
        "smart_crop": smart_crop,
        "wm_size_pct": wm_size_pct, "wm_pos_x_pct": wm_pos_x_pct, "wm_pos_y_pct": wm_pos_y_pct,
    }
    base_dir = BASE_OUTPUT_DIR
    log("Mengambil info video ...")
    try:
        info = clipper.get_video_info(url)
    except subprocess.CalledProcessError:
        info = {"title": "video", "uploader": "unknown"}
    out_dir = clipper.make_output_folder(base_dir, info["title"], info["uploader"])
    job = {"title": info["title"], "out_dir": out_dir.name, "results": [], "aspects": aspect_ratios, "url": url, "settings": settings}
    with state_lock:
        state["jobs"].append(job)
    log(f"Hasil akan disimpan di: {out_dir}")

    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp(prefix="ai-clipper-"))
    try:
        video_path = clipper.download_video(url, tmp_dir)
        segments = clipper.transcribe(video_path, tmp_dir)
        if speaker_colors:
            clipper.assign_pause_based_speakers(segments)
        total_duration = segments[-1]["end"] if segments else 0

        # simpan transkrip permanen di folder hasil -- supaya nanti bisa
        # generate rasio tambahan tanpa perlu transkrip ulang dari nol
        with open(out_dir / "segments.json", "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)

        moments = clipper.find_moments(segments, num_clips, total_duration)
        log(f"    Ditemukan {len(moments)} momen.")

        log(f"[4/5] Memotong {len(moments)} klip x {len(aspect_ratios)} rasio ({', '.join(aspect_ratios)}) ...")
        results = []
        for i, m in enumerate(moments, start=1):
            slug = clipper.slugify(m.title)
            files = {}
            for aspect in aspect_ratios:
                aspect_suffix = aspect.replace(":", "x")
                fname = f"{i:02d}-{slug}-{aspect_suffix}.mp4"
                out_path = out_dir / fname
                try:
                    wm_path = WATERMARK_PATH if (use_watermark and WATERMARK_PATH.is_file()) else None
                    clipper.cut_clip(video_path, m, segments, out_path, tmp_dir, subtitles, aspect, font_name, wm_path, speaker_colors, font_color, font_size_pct, pos_x_pct, pos_y_pct, smart_crop, wm_size_pct, wm_pos_x_pct, wm_pos_y_pct)
                    files[aspect] = fname
                    log(f"    [{i}/{len(moments)}] {fname}  ({m.duration:.0f}s) - {m.title} [{aspect}]")
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    log(f"    Gagal memotong klip {i} rasio {aspect}: {e}")

            if not files:
                continue  # semua rasio gagal untuk klip ini, skip total

            item = {
                "files": files,
                "title": m.title,
                "reason": m.reason,
                "caption": m.caption,
                "hashtags": m.hashtags,
                "duration": round(m.duration),
                "start": m.start,
                "end": m.end,
            }
            results.append(item)
            with state_lock:
                job["results"] = list(results)

        with open(out_dir / "moments.json", "w", encoding="utf-8") as f:
            json.dump({
                "aspects": aspect_ratios, "results": results,
                "url": url, "settings": settings,
            }, f, indent=2, ensure_ascii=False)

        log(f"[5/5] Selesai. {len(results)} klip tersimpan di: {out_dir.resolve()}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log("Video sumber & file temporary sudah dihapus.")


def run_queue(urls: list[str], num_clips: int, subtitles: bool, aspect_ratios: list[str] | None = None, font_name: str = "Arial Black", use_watermark: bool = False, speaker_colors: bool = False, font_color: str | None = None, font_size_pct: float | None = None, pos_x_pct: float | None = None, pos_y_pct: float | None = None, smart_crop: bool = False, wm_size_pct: float | None = None, wm_pos_x_pct: float | None = None, wm_pos_y_pct: float | None = None):
    real_stdout = sys.stdout
    sys.stdout = LogWriter()
    try:
        total = len(urls)
        for idx, url in enumerate(urls, start=1):
            with state_lock:
                state["queue_index"] = idx
            if total > 1:
                log(f"\n=== Video {idx}/{total}: {url} ===")
            try:
                process_one_video(url, num_clips, subtitles, aspect_ratios, font_name, use_watermark, speaker_colors, font_color, font_size_pct, pos_x_pct, pos_y_pct, smart_crop, wm_size_pct, wm_pos_x_pct, wm_pos_y_pct)
            except Exception as e:
                log(f"ERROR pada video {idx}/{total}: {e}")
                with state_lock:
                    state["error"] = str(e)
                # lanjut ke video berikutnya di antrian, tidak berhenti total

        with state_lock:
            state["done"] = True
    finally:
        sys.stdout = real_stdout
        with state_lock:
            state["running"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "error": "Masih ada proses berjalan."}), 400
        data = request.get_json(force=True)

        urls_raw = data.get("urls")
        if urls_raw is None:
            # kompatibilitas: dukung juga field lama "url" tunggal
            urls_raw = [data.get("url") or ""]
        urls = [u.strip() for u in urls_raw if u and u.strip()]

        num_clips = int(data.get("clips") or 8)
        subtitles = bool(data.get("subtitles"))
        VALID_ASPECTS = ("9:16", "16:9", "1:1")
        aspects_raw = data.get("aspects")
        if aspects_raw is None:
            # kompatibilitas: dukung juga field lama "aspect" tunggal
            aspects_raw = [data.get("aspect") or "9:16"]
        aspect_ratios = [a for a in aspects_raw if a in VALID_ASPECTS]
        if not aspect_ratios:
            aspect_ratios = ["9:16"]
        font_name = (data.get("font") or "Arial Black").strip()[:50] or "Arial Black"
        use_watermark = bool(data.get("watermark"))
        speaker_colors = bool(data.get("speaker_colors"))

        font_color = data.get("font_color")
        if font_color is not None:
            font_color = re.sub(r"[^0-9A-Fa-f]", "", str(font_color))[:6] or None

        def _num_or_none(key, lo, hi):
            val = data.get(key)
            if val is None or val == "":
                return None
            try:
                val = float(val)
            except (TypeError, ValueError):
                return None
            return max(lo, min(hi, val))

        font_size_pct = _num_or_none("font_size_pct", 20, 400)
        pos_x_pct = _num_or_none("pos_x_pct", 0, 100)
        pos_y_pct = _num_or_none("pos_y_pct", 0, 100)
        smart_crop = bool(data.get("smart_crop"))
        wm_size_pct = _num_or_none("wm_size_pct", 5, 60)
        wm_pos_x_pct = _num_or_none("wm_pos_x_pct", 0, 100)
        wm_pos_y_pct = _num_or_none("wm_pos_y_pct", 0, 100)

        if not urls:
            return jsonify({"ok": False, "error": "Link video kosong."}), 400

        state.update({
            "running": True, "done": False, "error": None,
            "log": [], "jobs": [], "queue_total": len(urls), "queue_index": 0,
        })

    thread = threading.Thread(target=run_queue, args=(urls, num_clips, subtitles, aspect_ratios, font_name, use_watermark, speaker_colors, font_color, font_size_pct, pos_x_pct, pos_y_pct, smart_crop, wm_size_pct, wm_pos_x_pct, wm_pos_y_pct), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "running": state["running"],
            "done": state["done"],
            "error": state["error"],
            "log": state["log"],
            "jobs": state["jobs"],
            "queue_total": state["queue_total"],
            "queue_index": state["queue_index"],
        })


@app.route("/api/history")
def api_history():
    """Baca ulang semua folder hasil dari disk (bukan dari memory) --
    jadi riwayat tetap ada walau server di-restart atau browser ditutup."""
    if not BASE_OUTPUT_DIR.exists():
        return jsonify({"jobs": []})

    entries = []
    for folder in BASE_OUTPUT_DIR.iterdir():
        if not folder.is_dir():
            continue
        moments_file = folder / "moments.json"
        if not moments_file.exists():
            continue
        try:
            with open(moments_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Format baru: {"aspects": [...], "results": [...], "url": ..., "settings": {...}}
        # Format lama (sebelum fitur multi-aspect): list hasil langsung.
        if isinstance(data, dict):
            results = data.get("results", [])
            aspects = data.get("aspects", ["9:16"])
            url = data.get("url")
            settings = data.get("settings", {})
        else:
            results = data
            aspects = ["9:16"]
            url = None
            settings = {}
        if not results:
            continue

        # nama folder formatnya "01 - Judul - Creator" -> ambil bagian judulnya
        name_parts = folder.name.split(" - ", 2)
        title = name_parts[1] if len(name_parts) >= 2 else folder.name

        entries.append({
            "title": title,
            "out_dir": folder.name,
            "results": results,
            "aspects": aspects,
            "url": url,
            "settings": settings,
            "modified": folder.stat().st_mtime,
        })

    entries.sort(key=lambda e: e["modified"], reverse=True)
    return jsonify({"jobs": entries})


@app.route("/api/thumbnail", methods=["POST"])
def api_thumbnail():
    """Ambil judul & thumbnail video (tanpa download) buat panel preview
    posisi watermark/subtitle sebelum mulai generate klip."""
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Link video kosong."}), 400
    try:
        info = clipper.get_video_info(url)
    except subprocess.CalledProcessError:
        return jsonify({"ok": False, "error": "Gagal mengambil info video. Cek link-nya benar."}), 400
    if not info.get("thumbnail"):
        return jsonify({"ok": False, "error": "Video ini tidak punya thumbnail yang bisa dipakai."}), 400
    return jsonify({"ok": True, "title": info["title"], "thumbnail": info["thumbnail"]})


@app.route("/api/watermark", methods=["GET"])
def watermark_status():
    return jsonify({"exists": WATERMARK_PATH.is_file()})


@app.route("/api/watermark", methods=["POST"])
def upload_watermark():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Tidak ada file yang dikirim."}), 400

    # Validasi tipe file lewat isi filenya (bukan cuma nama/ekstensi) --
    # cek magic bytes PNG di awal file.
    header = file.stream.read(8)
    file.stream.seek(0)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return jsonify({"ok": False, "error": "File harus berformat PNG."}), 400

    WATERMARK_DIR.mkdir(parents=True, exist_ok=True)
    file.save(str(WATERMARK_PATH))
    return jsonify({"ok": True})


@app.route("/api/watermark", methods=["DELETE"])
def delete_watermark():
    if WATERMARK_PATH.is_file():
        WATERMARK_PATH.unlink()
    return jsonify({"ok": True})


@app.route("/watermark/logo.png")
def serve_watermark():
    if not WATERMARK_PATH.is_file():
        return "Not found", 404
    return send_from_directory(WATERMARK_DIR, "logo.png")


@app.route("/clips/<path:relpath>")
def serve_clip(relpath):
    return send_from_directory(BASE_OUTPUT_DIR, relpath)


def _resolve_job_dir(job_dir: str) -> Path | None:
    """Pastikan job_dir tidak bisa dipakai buat keluar dari BASE_OUTPUT_DIR
    (path traversal, mis. job_dir="../../etc"). Balikin None kalau mencurigakan."""
    base = BASE_OUTPUT_DIR.resolve()
    candidate = (base / job_dir).resolve()
    if base not in candidate.parents and candidate != base:
        return None
    return candidate


@app.route("/api/download_zip/<path:job_dir>")
def download_zip(job_dir):
    folder = _resolve_job_dir(job_dir)
    if folder is None:
        return "Path tidak valid.", 400

    with state_lock:
        jobs = list(state["jobs"])
    job = next((j for j in jobs if j["out_dir"] == job_dir), None)
    results = job["results"] if job else None

    if results is None:
        # tidak ada di sesi aktif (mis. dibuka dari Riwayat) -- baca dari disk
        moments_file = folder / "moments.json"
        if not moments_file.exists():
            return "Belum ada klip untuk di-download.", 404
        with open(moments_file, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", []) if isinstance(data, dict) else data

    if not results:
        return "Belum ada klip untuk di-download.", 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            filenames = list(r["files"].values()) if "files" in r else [r["file"]]
            for fname in filenames:
                file_path = folder / fname
                if file_path.exists():
                    zf.write(file_path, arcname=fname)
    buf.seek(0)

    zip_name = job_dir + ".zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=zip_name)


@app.route("/api/open_folder/<path:job_dir>", methods=["POST"])
def open_folder(job_dir):
    folder = _resolve_job_dir(job_dir)
    if folder is None:
        return jsonify({"ok": False, "error": "Path tidak valid."}), 400
        return jsonify({"ok": False, "error": "Folder tidak ditemukan."}), 404

    try:
        if sys.platform == "win32":
            import os
            os.startfile(str(folder))  # noqa: S606 -- aman, path dari state internal, bukan input bebas user
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)