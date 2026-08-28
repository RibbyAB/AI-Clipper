#!/usr/bin/env python3
"""
AI Clipper - Web UI
====================
Antarmuka web lokal buat clipper.py. Jalankan:

    python web_app.py

lalu buka http://127.0.0.1:5000 di browser (biasanya kebuka otomatis).
"""

import io
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


def process_one_video(url: str, num_clips: int, subtitles: bool, aspect_ratio: str = "9:16"):
    """Proses satu video sampai selesai, hasil ditambahkan sebagai satu job
    baru ke state['jobs']. Error di satu video TIDAK menghentikan video
    lain di antrian -- dicatat di log lalu lanjut ke berikutnya."""
    base_dir = BASE_OUTPUT_DIR
    log("Mengambil info video ...")
    try:
        info = clipper.get_video_info(url)
    except subprocess.CalledProcessError:
        info = {"title": "video", "uploader": "unknown"}
    out_dir = clipper.make_output_folder(base_dir, info["title"], info["uploader"])
    job = {"title": info["title"], "out_dir": out_dir.name, "results": []}
    with state_lock:
        state["jobs"].append(job)
    log(f"Hasil akan disimpan di: {out_dir}")

    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp(prefix="ai-clipper-"))
    try:
        video_path = clipper.download_video(url, tmp_dir)
        segments = clipper.transcribe(video_path, tmp_dir)
        total_duration = segments[-1]["end"] if segments else 0

        moments = clipper.find_moments(segments, num_clips, total_duration)
        log(f"    Ditemukan {len(moments)} momen.")

        log(f"[4/5] Memotong {len(moments)} klip (rasio {aspect_ratio}) ...")
        results = []
        for i, m in enumerate(moments, start=1):
            fname = f"{i:02d}-{clipper.slugify(m.title)}.mp4"
            out_path = out_dir / fname
            try:
                clipper.cut_clip(video_path, m, segments, out_path, tmp_dir, subtitles, aspect_ratio)
                log(f"    [{i}/{len(moments)}] {fname}  ({m.duration:.0f}s) - {m.title}")
                item = {
                    "file": fname,
                    "title": m.title,
                    "reason": m.reason,
                    "caption": m.caption,
                    "hashtags": m.hashtags,
                    "duration": round(m.duration),
                }
                results.append(item)
                with state_lock:
                    job["results"] = list(results)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                log(f"    Gagal memotong klip {i}: {e}")

        with open(out_dir / "moments.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        log(f"[5/5] Selesai. {len(results)} klip tersimpan di: {out_dir.resolve()}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log("Video sumber & file temporary sudah dihapus.")


def run_queue(urls: list[str], num_clips: int, subtitles: bool, aspect_ratio: str = "9:16"):
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
                process_one_video(url, num_clips, subtitles, aspect_ratio)
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
        aspect_ratio = data.get("aspect") or "9:16"
        if aspect_ratio not in ("9:16", "16:9"):
            aspect_ratio = "9:16"
        if not urls:
            return jsonify({"ok": False, "error": "Link video kosong."}), 400

        state.update({
            "running": True, "done": False, "error": None,
            "log": [], "jobs": [], "queue_total": len(urls), "queue_index": 0,
        })

    thread = threading.Thread(target=run_queue, args=(urls, num_clips, subtitles, aspect_ratio), daemon=True)
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


@app.route("/clips/<path:relpath>")
def serve_clip(relpath):
    return send_from_directory(BASE_OUTPUT_DIR, relpath)


@app.route("/api/download_zip/<path:job_dir>")
def download_zip(job_dir):
    with state_lock:
        jobs = list(state["jobs"])
    job = next((j for j in jobs if j["out_dir"] == job_dir), None)
    if not job or not job["results"]:
        return "Belum ada klip untuk di-download.", 404

    folder = BASE_OUTPUT_DIR / job["out_dir"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in job["results"]:
            file_path = folder / r["file"]
            if file_path.exists():
                zf.write(file_path, arcname=r["file"])
    buf.seek(0)

    zip_name = job["out_dir"] + ".zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=zip_name)


@app.route("/api/open_folder/<path:job_dir>", methods=["POST"])
def open_folder(job_dir):
    folder = (BASE_OUTPUT_DIR / job_dir).resolve()
    if not folder.exists():
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