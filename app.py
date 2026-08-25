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

# ---------------------------------------------------------------------------
# State global sederhana (aplikasi ini dipakai 1 orang di 1 browser lokal,
# jadi tidak perlu sistem multi-user/job-queue yang rumit)
# ---------------------------------------------------------------------------

state = {
    "running": False,
    "done": False,
    "error": None,
    "log": [],
    "results": [],
    "out_dir": None,
}
state_lock = threading.Lock()


MAX_LOG_LINES = 300


def _trim_log():
    # simpan cuma N baris terakhir -- daftar log yang terus membesar bikin
    # payload /api/status makin besar tiap poll dan render browser makin berat.
    if len(state["log"]) > MAX_LOG_LINES:
        state["log"] = state["log"][-MAX_LOG_LINES:]


def log(msg: str):
    with state_lock:
        state["log"].append(msg)
        _trim_log()
    # tulis ke konsol asli (bukan print() biasa, supaya tidak ke-tangkap lagi
    # oleh LogWriter yang menggantikan sys.stdout selama pipeline berjalan)
    try:
        print(msg, file=sys.__stdout__)
    except UnicodeEncodeError:
        # Command Prompt Windows kadang tidak bisa nampilkan karakter unicode
        # tertentu (mis. tanda hubung panjang). Ini cuma cermin ke konsol
        # untuk debugging -- data yang dikirim ke web UI tetap utuh, jadi
        # aman untuk dilewati saja kalau gagal tampil di konsol.
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


def run_pipeline(url: str, num_clips: int, subtitles: bool):
    import sys
    real_stdout = sys.stdout
    sys.stdout = LogWriter()
    try:
        base_dir = Path("output")
        log("Mengambil info video ...")
        try:
            info = clipper.get_video_info(url)
        except subprocess.CalledProcessError:
            info = {"title": "video", "uploader": "unknown"}
        out_dir = clipper.make_output_folder(base_dir, info["title"], info["uploader"])
        with state_lock:
            state["out_dir"] = str(out_dir)
        log(f"Hasil akan disimpan di: {out_dir}")

        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix="ai-clipper-"))
        try:
            video_path = clipper.download_video(url, tmp_dir)
            segments = clipper.transcribe(video_path, tmp_dir)
            total_duration = segments[-1]["end"] if segments else 0

            moments = clipper.find_moments(segments, num_clips, total_duration)
            log(f"    Ditemukan {len(moments)} momen.")

            log(f"[4/5] Memotong & mem-vertikal-kan {len(moments)} klip ...")
            results = []
            for i, m in enumerate(moments, start=1):
                fname = f"{i:02d}-{clipper.slugify(m.title)}.mp4"
                out_path = out_dir / fname
                try:
                    clipper.cut_clip(video_path, m, segments, out_path, tmp_dir, subtitles)
                    log(f"    [{i}/{len(moments)}] {fname}  ({m.duration:.0f}s) - {m.title}")
                    item = {
                        "file": fname,
                        "title": m.title,
                        "reason": m.reason,
                        "duration": round(m.duration),
                    }
                    results.append(item)
                    with state_lock:
                        state["results"] = list(results)
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    log(f"    Gagal memotong klip {i}: {e}")

            with open(out_dir / "moments.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            log(f"[5/5] Selesai. {len(results)} klip tersimpan di: {out_dir.resolve()}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log("Video sumber & file temporary sudah dihapus.")

        with state_lock:
            state["done"] = True

    except Exception as e:
        log(f"ERROR: {e}")
        with state_lock:
            state["error"] = str(e)
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
        url = (data.get("url") or "").strip()
        num_clips = int(data.get("clips") or 8)
        subtitles = bool(data.get("subtitles"))
        if not url:
            return jsonify({"ok": False, "error": "Link video kosong."}), 400

        state.update({
            "running": True, "done": False, "error": None,
            "log": [], "results": [], "out_dir": None,
        })

    thread = threading.Thread(target=run_pipeline, args=(url, num_clips, subtitles), daemon=True)
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
            "results": state["results"],
            "out_dir": state["out_dir"],
        })


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    with state_lock:
        out_dir = state["out_dir"]
    if not out_dir:
        return "Not found", 404
    return send_from_directory(out_dir, filename)


@app.route("/api/download_zip")
def download_zip():
    with state_lock:
        out_dir = state["out_dir"]
        results = list(state["results"])

    if not out_dir or not results:
        return "Belum ada klip untuk di-download.", 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            file_path = Path(out_dir) / r["file"]
            if file_path.exists():
                zf.write(file_path, arcname=r["file"])
    buf.seek(0)

    zip_name = Path(out_dir).name + ".zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=zip_name)


def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)