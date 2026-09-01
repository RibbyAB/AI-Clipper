#!/usr/bin/env python3
"""
AI Clipper
==========
Ambil video dari link (YouTube, dll) -> transkrip otomatis -> AI cari momen
menarik -> potong jadi klip pendek vertikal (9:16) -> (opsional) subtitle.

Video sumber di-download sementara ke folder temp dan DIHAPUS otomatis
setelah proses selesai, jadi tidak menuhin storage.

Cara pakai:
    python clipper.py "https://youtube.com/watch?v=xxxx" --clips 8

Environment variable yang dibutuhkan:
    ANTHROPIC_API_KEY   -> untuk deteksi momen menarik (wajib)

Lihat README.md untuk detail instalasi.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = os.environ.get("CLIPPER_WHISPER_MODEL", "small")
GROQ_MODEL = os.environ.get("CLIPPER_GROQ_MODEL", "openai/gpt-oss-120b")
OUTPUT_DIR = Path("output")
# Folder berisi file .ttf yang dibundel bersama app -- supaya font-font
# trendi (Bebas Neue, Anton, dll) langsung bisa dipakai ffmpeg tanpa perlu
# diinstall manual ke sistem operasi.
FONTS_DIR = Path(__file__).resolve().parent / "fonts"
# File cascade wajah (buat smart crop) dibundel sendiri di project, TIDAK
# bergantung pada file bawaan paket OpenCV -- soalnya packaging OpenCV
# ternyata bisa berubah-ubah antar versi (mis. OpenCV 5.0 berhenti
# nyertain file cascade sama sekali di paketnya). Dengan bundling sendiri,
# smart crop tetap jalan konsisten berapa pun versi OpenCV yang terpasang,
# selama cv2.CascadeClassifier (class-nya) masih tersedia.
CASCADES_DIR = Path(__file__).resolve().parent / "cascades"
# Nama folder font versi lokal (di dalam tmp_dir tiap video) -- sengaja
# pendek tanpa spasi, supaya bisa dirujuk di filter ffmpeg TANPA tanda
# kutip sama sekali. Sebagian versi ffmpeg (terutama build Windows) gagal
# parse kalau ada 2 nilai filter yang sama-sama dibungkus kutip dalam satu
# filter (fontsdir='...' + force_style='...' sekaligus).
LOCAL_FONTS_DIRNAME = "fnt"


def _ensure_local_fontsdir(tmp_dir: Path) -> str | None:
    """Copy folder fonts/ ke dalam tmp_dir (sekali per video, dipakai ulang
    untuk semua klip video itu) supaya fontsdir bisa dirujuk pakai nama
    folder relatif pendek -- tanpa spasi, tanpa drive letter, tanpa titik
    dua -- yang aman dipakai di filter ffmpeg tanpa perlu tanda kutip."""
    if not FONTS_DIR.is_dir():
        return None
    local = tmp_dir / LOCAL_FONTS_DIRNAME
    if not local.exists():
        shutil.copytree(FONTS_DIR, local)
    return LOCAL_FONTS_DIRNAME


@dataclass
class Moment:
    start: float
    end: float
    title: str
    reason: str
    caption: str = ""
    hashtags: str = ""

    @property
    def duration(self):
        return self.end - self.start


# ---------------------------------------------------------------------------
# 1. Download video dari link (sementara, auto-hapus di akhir)
# ---------------------------------------------------------------------------

def get_video_info(url: str) -> dict:
    """Ambil judul, nama creator, & URL thumbnail video tanpa perlu download videonya."""
    cmd = ["yt-dlp", "--skip-download", "--print", "%(title)s", "--print", "%(uploader)s",
           "--print", "%(thumbnail)s", url]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (result.stdout or "").strip().split("\n")
    title = lines[0] if len(lines) > 0 else "video"
    uploader = lines[1] if len(lines) > 1 else "unknown"
    thumbnail = lines[2] if len(lines) > 2 and lines[2] != "NA" else None
    return {"title": title, "uploader": uploader, "thumbnail": thumbnail}


def make_output_folder(base_dir: Path, title: str, uploader: str) -> Path:
    """Bikin folder baru bernomor urut, format: 01 - Judul - Creator (gak akan
    numpuk/timpa folder hasil run sebelumnya)."""
    def clean(text: str, max_len: int) -> str:
        text = re.sub(r'[<>:"/\\|?*]', "", text).strip()
        return text[:max_len] if text else "unknown"

    safe_title = clean(title, 60)
    safe_uploader = clean(uploader, 30)
    base_dir.mkdir(parents=True, exist_ok=True)

    n = 1
    while True:
        folder_name = f"{n:02d} - {safe_title} - {safe_uploader}"
        candidate = base_dir / folder_name
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        n += 1


def download_video(url: str, tmp_dir: Path) -> Path:
    print(f"[1/5] Mengunduh video dari: {url}")
    out_template = str(tmp_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    candidates = list(tmp_dir.glob("source.*"))
    if not candidates:
        raise RuntimeError("Gagal menemukan file hasil download.")
    return candidates[0]


# ---------------------------------------------------------------------------
# 2. Transkrip pakai Whisper (dengan timestamp per segmen)
# ---------------------------------------------------------------------------

def transcribe(video_path: Path, tmp_dir: Path) -> list[dict]:
    from faster_whisper import WhisperModel  # import di sini biar startup script tetap cepat

    def _fmt(t: float) -> str:
        t = max(0, int(t))
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _run(device: str, compute_type: str) -> list[dict]:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
        segments_gen, info = model.transcribe(str(video_path), beam_size=5)
        total_dur = getattr(info, "duration", None)

        result = []
        start_time = time.time()
        for i, s in enumerate(segments_gen, start=1):
            result.append({"start": s.start, "end": s.end, "text": (s.text or "").strip()})
            if i % 25 == 0:
                elapsed = time.time() - start_time
                if total_dur:
                    pct = min(100, s.end / total_dur * 100)
                    rate = s.end / elapsed if elapsed > 0 else 0
                    eta = (total_dur - s.end) / rate if rate > 0 else None
                    eta_str = f" - sisa ~{_fmt(eta)}" if eta else ""
                    print(f"    [{pct:3.0f}%] {_fmt(s.end)} / {_fmt(total_dur)}{eta_str}")
                else:
                    print(f"    ... {i} segmen, sampai {_fmt(s.end)} ({_fmt(elapsed)} berjalan)")
        return result

    print(f"[2/5] Transkrip audio (model whisper: {WHISPER_MODEL_SIZE}) ...")
    try:
        # Error terkait GPU (termasuk file .dll CUDA yang hilang/rusak) kadang
        # baru muncul di tengah proses transkripsi, bukan cuma saat loading
        # model -- makanya seluruh proses dibungkus try/except, bukan cuma
        # baris pembuatan model-nya saja.
        segments = _run("cuda", "float16")
        print("    (berhasil pakai GPU)")
    except Exception as e:
        print(f"    GPU tidak bisa dipakai ({e}). Mengulang pakai CPU ...")
        segments = _run("cpu", "int8")

    # simpan transkrip mentah, berguna buat debug / dipakai ulang
    with open(tmp_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    return segments


def format_transcript_for_prompt(segments: list[dict], max_chars: int = 5000) -> str:
    """Gabungkan segmen-segmen kecil jadi chunk ~15 detik biar hemat token,
    lalu kalau masih kepanjangan, ambil sampel merata dari seluruh video
    (bukan cuma bagian awal) supaya tetap mewakili keseluruhan durasi."""
    chunks = []
    cur_start = None
    cur_end = None
    cur_text = []
    for s in segments:
        if cur_start is None:
            cur_start = s["start"]
        cur_end = s["end"]
        cur_text.append(s["text"])
        if cur_end - cur_start >= 15:
            chunks.append((cur_start, cur_end, " ".join(cur_text).strip()))
            cur_start, cur_end, cur_text = None, None, []
    if cur_text:
        chunks.append((cur_start, cur_end, " ".join(cur_text).strip()))

    lines = [f"[{c[0]:.0f}-{c[1]:.0f}] {c[2]}" for c in chunks]
    full_text = "\n".join(lines)

    if len(full_text) <= max_chars:
        return full_text

    # Terlalu panjang -> ambil sampel merata sepanjang video (bukan cuma awal),
    # supaya AI tetap "melihat" seluruh durasi video, bukan cuma bagian depan.
    avg_line_len = max(1, len(full_text) // max(1, len(lines)))
    target_line_count = max(1, max_chars // avg_line_len)
    keep_ratio = max(1, len(lines) // target_line_count)
    sampled = lines[::keep_ratio]
    sampled_text = "\n".join(sampled)
    if len(sampled_text) > max_chars:
        sampled_text = sampled_text[:max_chars]
    return sampled_text


# ---------------------------------------------------------------------------
# 3. Deteksi momen menarik pakai Claude
# ---------------------------------------------------------------------------

def find_moments(segments: list[dict], num_clips: int, total_duration: float) -> list[Moment]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY belum di-set. Daftar gratis di console.groq.com lalu "
            "buat API key. Lihat README.md bagian instalasi."
        )

    # Video pendek tidak mungkin muat N klip @ minimal 65 detik tanpa
    # tumpang tindih -- turunkan otomatis jumlah klip yang diminta supaya
    # AI tidak "mengarang" timestamp yang tidak masuk akal.
    min_duration = 65.0
    max_feasible = max(1, int(total_duration // (min_duration + 5)))
    effective_clips = min(num_clips, max_feasible)
    if effective_clips < num_clips:
        print(f"    Video terlalu pendek untuk {num_clips} klip @ {min_duration:.0f}d+, "
              f"disesuaikan otomatis jadi {effective_clips} klip.")

    print(f"[3/5] Mencari {effective_clips} momen menarik lewat AI (Groq, gratis) ...")

    import urllib.request
    import urllib.error

    # Alokasikan budget token secara dinamis: makin banyak klip diminta,
    # makin banyak token dibutuhkan buat jawaban -> transkrip yang dikirim
    # dipangkas lebih agresif supaya total (prompt + jawaban) tetap di
    # bawah limit gratis Groq (8000 token/menit).
    completion_budget = min(4000, 350 + effective_clips * 190)
    prompt_char_budget = max(800, int((7200 - completion_budget) / 1.1))
    char_budgets = [prompt_char_budget, prompt_char_budget // 2, prompt_char_budget // 4, 800]
    last_error = None

    for attempt, max_chars in enumerate(char_budgets, start=1):
        transcript_text = format_transcript_for_prompt(segments, max_chars=max_chars)

        prompt = f"""Kamu adalah editor video profesional yang mencari momen paling
menarik dari sebuah transkrip video untuk dijadikan klip pendek (short-form,
untuk TikTok/Reels/YouTube Shorts).

Berikut transkrip lengkap dengan timestamp dalam detik [start-end]:

{transcript_text}

Total durasi video: {total_duration:.0f} detik.

Pilih {effective_clips} momen TERBAIK yang berpotensi viral/menarik ditonton secara
berdiri sendiri (self-contained) -- misalnya: pernyataan mengejutkan, momen
emosional, punchline lucu, insight kuat, cerita singkat yang punya awal-akhir
jelas, atau statement kontroversial.

Aturan penting:
- Setiap klip WAJIB berdurasi minimal 65 detik (syarat monetisasi TikTok
  adalah video minimal 1 menit), idealnya 65-150 detik. JANGAN memilih
  klip pendek walaupun transkrip di atas terlihat terpotong per beberapa
  detik -- itu cuma format tampilan, bukan batasan durasi klip. start dan
  end BOLEH melewati/menggabungkan banyak baris transkrip sekaligus untuk
  membentuk cerita/konteks yang utuh dan cukup panjang.
- start dan end HARUS berada di dalam rentang 0 sampai {total_duration:.0f}
  detik (durasi total video). JANGAN PERNAH melebihi angka ini.
- start dan end WAJIB ditulis sebagai ANGKA DIGIT murni (contoh: 90, bukan
  "ninety" atau "sembilan puluh"). JANGAN PERNAH menulis angka dalam bentuk
  kata, dalam bahasa apapun.
- Klip TIDAK BOLEH saling tumpang tindih satu sama lain.
- Urutkan dari yang paling menarik ke yang kurang menarik.
- Untuk tiap klip, buatkan juga "caption" siap-pakai untuk posting di
  TikTok/Reels/Shorts -- singkat, ada hook menarik di awal, bahasa santai
  sesuai gaya video, maksimal 2-3 kalimat.
- Untuk tiap klip, buatkan juga "hashtags" -- 5 sampai 8 hashtag relevan
  dipisah spasi (contoh: "#fyp #viral #edukasi"), campuran hashtag umum
  (jangkauan luas) dan spesifik sesuai topik klip.

Jawab HANYA dengan JSON array, tanpa teks lain, tanpa markdown code fence,
format persis seperti ini (perhatikan start/end adalah angka, bukan kata):
[
  {{"start": 12.5, "end": 95.0, "title": "Judul singkat menarik",
    "reason": "Kenapa momen ini menarik",
    "caption": "Caption siap posting dengan hook menarik di awal",
    "hashtags": "#fyp #viral #relevan"}}
]
"""

        body = json.dumps({
            "model": GROQ_MODEL,
            "max_tokens": completion_budget,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "curl/8.4.0",
            },
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            break  # berhasil, keluar dari loop retry
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"Groq API menolak request (HTTP {e.code}). Detail: {error_body}"
            )
            if e.code == 413 and attempt < len(char_budgets):
                print(f"    Transkrip masih kepanjangan, memangkas lebih lanjut (percobaan {attempt+1}) ...")
                continue
            raise last_error from e
    else:
        raise last_error

    message = data["choices"][0]["message"]
    text = (message.get("content") or "").strip()
    if not text:
        # Model kadang taruh jawaban di field 'reasoning' kalau token habis
        # sebelum sempat nulis 'content' -- coba ambil dari situ sebagai fallback.
        text = (message.get("reasoning") or "").strip()
    text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()

    if not text:
        raise RuntimeError(
            "AI tidak mengembalikan jawaban (response kosong). "
            f"Response lengkap: {json.dumps(data)[:500]}"
        )

    # Kadang model menambahkan teks penjelasan sebelum/sesudah JSON array-nya,
    # ambil bagian [ ... ] paling luar saja.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    text = fix_word_numbers(text)

    raw_moments = parse_moments_json(text)
    if not raw_moments:
        raise RuntimeError(
            f"AI tidak menghasilkan klip yang valid sama sekali. Isi respons: {text[:500]}"
        )

    moments = [
        Moment(start=float(m["start"]), end=float(m["end"]),
               title=str(m.get("title", "klip")), reason=str(m.get("reason", "")),
               caption=str(m.get("caption", "")), hashtags=str(m.get("hashtags", "")))
        for m in raw_moments
        if _is_valid_raw_moment(m)
    ]

    moments = clamp_and_dedupe_moments(moments, total_duration)
    if not moments:
        raise RuntimeError("Semua klip dari AI tidak valid (timestamp di luar durasi video atau rusak).")

    moments = enforce_min_duration(moments, total_duration, min_duration=min_duration)
    return moments


def _is_valid_raw_moment(m: dict) -> bool:
    try:
        start, end = float(m["start"]), float(m["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return end > start


def parse_moments_json(text: str) -> list[dict]:
    """Parse JSON array hasil AI. Kalau gagal (mis. jawaban kepotong di
    tengah karena kehabisan token), coba selamatkan objek-objek yang masih
    lengkap satu per satu, daripada membuang semua hasil."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: ambil tiap blok {...} yang berdiri sendiri dan valid,
    # lewati yang rusak/terpotong.
    salvaged = []
    for block in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(block.group(0))
            salvaged.append(obj)
        except json.JSONDecodeError:
            continue
    return salvaged


def clamp_and_dedupe_moments(moments: list[Moment], total_duration: float) -> list[Moment]:
    """Pastikan semua klip berada dalam rentang durasi video yang valid,
    dan hilangkan tumpang tindih antar klip (AI kadang tetap overlap
    walau sudah diminta tidak)."""
    cleaned = []
    for m in moments:
        start = max(0.0, min(m.start, total_duration))
        end = max(0.0, min(m.end, total_duration))
        if end - start < 5:  # terlalu pendek/degenerate, buang
            continue
        m.start, m.end = start, end
        cleaned.append(m)

    cleaned.sort(key=lambda m: m.start)
    result = []
    for m in cleaned:
        if result and m.start < result[-1].end:
            m.start = result[-1].end  # geser supaya tidak overlap
            if m.end - m.start < 5:
                continue
        result.append(m)
    return result


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


def fix_word_numbers(text: str) -> str:
    """AI kadang nulis angka dalam bentuk kata Inggris (mis. "start": ninety)
    walau sudah diminta pakai digit. Perbaiki otomatis khusus untuk nilai
    "start"/"end" supaya JSON tetap valid, daripada langsung gagal total."""
    def replace(m):
        key, word = m.group(1), m.group(2).lower().strip()
        if word in _WORD_NUMBERS:
            return f'"{key}": {_WORD_NUMBERS[word]}'
        return m.group(0)

    pattern = r'"(start|end)"\s*:\s*"?([A-Za-z]+)"?'
    return re.sub(pattern, replace, text)


def enforce_min_duration(moments: list[Moment], total_duration: float,
                          min_duration: float = 65.0) -> list[Moment]:
    """AI kadang milih klip yang terlalu pendek (misal ngikutin batas chunk
    transkrip ~15 detik). Perlebar simetris (maju & mundur) sampai minimal
    min_duration, selama tidak nabrak klip lain atau keluar durasi video."""
    moments = sorted(moments, key=lambda m: m.start)
    for i, m in enumerate(moments):
        if m.duration >= min_duration:
            continue
        deficit = min_duration - m.duration

        prev_end = moments[i - 1].end if i > 0 else 0.0
        next_start = moments[i + 1].start if i < len(moments) - 1 else total_duration

        room_before = max(0.0, m.start - prev_end)
        room_after = max(0.0, next_start - m.end)

        extend_before = min(deficit / 2, room_before)
        extend_after = min(deficit - extend_before, room_after)
        # kalau salah satu sisi mentok, coba ambil sisa dari sisi satunya
        remaining = deficit - extend_before - extend_after
        if remaining > 0:
            extend_before = min(extend_before + remaining, room_before)
        remaining = deficit - extend_before - extend_after
        if remaining > 0:
            extend_after = min(extend_after + remaining, room_after)

        m.start = max(0.0, m.start - extend_before)
        m.end = min(total_duration, m.end + extend_after)
    return moments


# ---------------------------------------------------------------------------
# 4. Potong video + crop sesuai aspect ratio (9:16 atau 16:9) + subtitle opsional
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:50]


# Palet warna buat "pembicara" berbeda -- dipilih supaya jelas beda satu
# sama lain dan tetap gampang kebaca di atas outline hitam.
SPEAKER_COLORS = ["FFFFFF", "FFD400", "00E5FF", "FF4FA3"]


def assign_pause_based_speakers(segments: list[dict], gap_threshold: float = 1.0):
    """Deteksi SEDERHANA "ganti pembicara" berdasarkan jeda hening antar
    kalimat (bukan speaker diarization asli). Kalau jeda antara akhir satu
    kalimat dan mulai kalimat berikutnya lebih lama dari gap_threshold detik,
    dianggap gantian orang, warnanya ikut gantian. Ini CUMA HEURISTIK --
    tidak akurat untuk kasus 1 orang bicara panjang dengan jeda alami, atau
    beberapa orang menyahut cepat tanpa jeda."""
    speaker_idx = 0
    prev_end = None
    for seg in segments:
        if prev_end is not None and seg["start"] - prev_end > gap_threshold:
            speaker_idx = (speaker_idx + 1) % len(SPEAKER_COLORS)
        seg["color"] = SPEAKER_COLORS[speaker_idx]
        prev_end = seg["end"]


def build_ass(segments: list[dict], start: float, end: float, out_path: Path,
              font_name: str, font_size: int, primary_colour_ass: str, margin_v: int,
              target_w: int, target_h: int,
              speaker_colors: bool = False, pos_xy: tuple[int, int] | None = None):
    """Bikin file .ass (bukan .srt) -- format ASS native mendukung tag posisi
    (\\pos) dan warna per-baris (\\c) dengan reliable, beda dari .srt yang
    cuma dukung tag terbatas (<font color> doang, tag \\pos tidak dikenali)."""
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {target_w}\n"
        f"PlayResY: {target_h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary_colour_ass},&H000000FF,&H00000000,"
        f"&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        rel_start = max(seg["start"], start) - start
        rel_end = min(seg["end"], end) - start
        text = seg["text"].replace("\n", "\\N")

        override_tags = []
        if pos_xy:
            override_tags.append(f"an5\\pos({pos_xy[0]},{pos_xy[1]})")
        if speaker_colors:
            color = seg.get("color", SPEAKER_COLORS[0])
            bb, gg, rr = color[4:6], color[2:4], color[0:2]
            override_tags.append(f"c&H{bb}{gg}{rr}&")
        override = "{\\" + "\\".join(override_tags) + "}" if override_tags else ""

        lines.append(f"Dialogue: 0,{fmt(rel_start)},{fmt(rel_end)},Default,,0,0,0,,{override}{text}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def detect_face_track(video_path: Path, start: float, end: float,
                       sample_interval: float = 2.0, max_samples: int = 30):
    """Deteksi posisi wajah (horizontal) di beberapa titik waktu sepanjang
    klip, buat dasar "smart crop" yang ikut gerak ke arah wajah. Pakai
    Haar Cascade bawaan OpenCV -- ringan, tidak perlu download model
    tambahan. Kalau OpenCV tidak terpasang atau wajah tidak terdeteksi
    sama sekali, return kosong (caller otomatis fallback ke crop tengah
    statis seperti biasa -- tidak pernah bikin proses gagal total)."""
    try:
        import cv2
    except ImportError:
        return [], 0, 0

    if not hasattr(cv2, "CascadeClassifier"):
        # OpenCV 5.0+ memindahkan CascadeClassifier ke paket "contrib" --
        # kalau yang terinstall cuma opencv-python-headless biasa, class
        # ini gak ada sama sekali. Pesan ini dicetak lewat exception di
        # _try_build_smart_crop_filter (caller), bukan di sini.
        raise RuntimeError(
            "cv2.CascadeClassifier tidak tersedia -- install "
            "opencv-contrib-python-headless (bukan opencv-python-headless biasa)"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0, 0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        return [], 0, 0

    # Dua cascade: wajah depan (paling umum) + wajah profil/miring (buat
    # video interview/podcast di mana orang sering gak lihat lurus ke
    # kamera). Cascade profil defaultnya cuma nangkep wajah nengok ke satu
    # arah, jadi kita cek juga versi frame yang di-flip biar dua arah
    # (kiri & kanan) sama-sama kecover.
    #
    # File XML-nya dibundel sendiri di folder cascades/ (bukan diambil dari
    # cv2.data.haarcascades) -- soalnya packaging OpenCV ternyata bisa
    # berubah antar versi (OpenCV 5.0 berhenti nyertain file ini sama
    # sekali di paketnya, walau class CascadeClassifier-nya sendiri masih
    # ada lewat paket contrib).
    frontal_path = CASCADES_DIR / "haarcascade_frontalface_default.xml"
    profile_path = CASCADES_DIR / "haarcascade_profileface.xml"
    if not frontal_path.is_file() or not profile_path.is_file():
        cap.release()
        raise RuntimeError(f"File cascade tidak ditemukan di {CASCADES_DIR}")
    frontal_cascade = cv2.CascadeClassifier(str(frontal_path))
    profile_cascade = cv2.CascadeClassifier(str(profile_path))
    # minSize proporsional ke ukuran video -- 60px absolut ternyata sering
    # kebesaran/kekecilan tergantung resolusi & seberapa jauh wajah dari
    # kamera, bikin banyak wajah valid gagal kedeteksi.
    min_size = max(30, min(frame_w, frame_h) // 12)

    def _detect(gray_eq):
        faces = list(frontal_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.08, minNeighbors=4, minSize=(min_size, min_size)))
        if faces:
            return faces
        faces = list(profile_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.08, minNeighbors=4, minSize=(min_size, min_size)))
        if faces:
            return faces
        # coba versi di-flip horizontal, buat nangkep wajah profil yang
        # nengok ke arah berlawanan (cascade profil cuma cover 1 arah)
        flipped = cv2.flip(gray_eq, 1)
        faces_flipped = profile_cascade.detectMultiScale(
            flipped, scaleFactor=1.08, minNeighbors=4, minSize=(min_size, min_size))
        # koordinat x dari hasil deteksi di frame yang di-flip perlu
        # dibalik lagi supaya posisinya benar relatif ke frame asli
        w = gray_eq.shape[1]
        return [(w - fx - fw, fy, fw, fh) for (fx, fy, fw, fh) in faces_flipped]

    duration = max(0.1, end - start)
    n_samples = max(2, min(max_samples, int(duration / sample_interval) + 1))
    times = [start + i * duration / (n_samples - 1) for i in range(n_samples)]

    track = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            track.append((t - start, None))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalize histogram -- ratakan kontras, terbukti signifikan
        # meningkatkan akurasi Haar Cascade di video dengan pencahayaan
        # kurang bagus/kurang terang (kondisi umum banget di video hasil
        # download, apalagi yang direkam di dalam ruangan).
        gray = cv2.equalizeHist(gray)
        faces = _detect(gray)
        if len(faces) == 0:
            track.append((t - start, None))
            continue
        # Asumsi wajah terbesar di frame = subjek utama yang lagi ngomong.
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        center_x_norm = (fx + fw / 2) / frame_w
        track.append((t - start, center_x_norm))
    cap.release()
    return track, frame_w, frame_h


def _fill_and_smooth_track(track: list, default: float = 0.5, alpha: float = 0.35):
    """Isi titik yang gagal deteksi wajah (pakai nilai terakhir yang valid),
    lalu haluskan pergerakannya (exponential smoothing) supaya crop tidak
    lompat-lompat kasar antar sampel."""
    if not track:
        return []
    last_valid = next((v for _, v in track if v is not None), default)
    filled = []
    for t, v in track:
        if v is None:
            v = last_valid
        else:
            last_valid = v
        filled.append((t, v))

    smoothed = []
    prev = None
    for t, v in filled:
        s = v if prev is None else prev + alpha * (v - prev)
        smoothed.append((t, s))
        prev = s
    return smoothed


def build_pan_expr(track: list, crop_w: float, frame_w: int) -> str:
    """Bangun ekspresi ffmpeg (dipakai sebagai posisi X filter crop) yang
    geser mengikuti titik-titik posisi wajah dari waktu ke waktu, dengan
    interpolasi linear antar sampel supaya gerakannya halus (bukan patah)."""
    if not track or frame_w <= 0:
        return f"(iw-{crop_w:.0f})/2"

    def clamp_x(center_norm):
        raw = center_norm * frame_w - crop_w / 2
        return max(0.0, min(frame_w - crop_w, raw))

    if len(track) == 1:
        return f"{clamp_x(track[0][1]):.1f}"

    expr = f"{clamp_x(track[-1][1]):.1f}"  # setelah titik terakhir, tahan di posisi itu
    for i in range(len(track) - 1, 0, -1):
        t0, c0 = track[i - 1]
        t1, c1 = track[i]
        x0, x1 = clamp_x(c0), clamp_x(c1)
        if t1 - t0 <= 0:
            seg_expr = f"{x1:.1f}"
        else:
            seg_expr = f"({x0:.1f}+({x1:.1f}-{x0:.1f})*(t-{t0:.2f})/{(t1 - t0):.2f})"
        expr = f"if(lt(t\\,{t1:.2f})\\,{seg_expr}\\,{expr})"
    x_first = clamp_x(track[0][1])
    expr = f"if(lt(t\\,{track[0][0]:.2f})\\,{x_first:.1f}\\,{expr})"
    return expr


def _try_build_smart_crop_filter(video_path: Path, moment: "Moment", square: bool) -> str | None:
    """Coba bangun filter crop yang ikut gerak wajah. Return None kalau
    gagal karena alasan apa pun (OpenCV tidak ada, wajah tidak terdeteksi,
    video rusak, dll) -- caller lalu otomatis fallback ke crop tengah
    statis, jadi smart crop TIDAK PERNAH bikin proses gagal total."""
    try:
        import cv2  # noqa: F401 -- cuma buat cek modulnya beneran ada
    except ImportError:
        print("    [smart crop] OpenCV belum terinstall (pip install -r requirements.txt) -- pakai crop tengah biasa.")
        return None

    try:
        track, frame_w, frame_h = detect_face_track(video_path, moment.start, moment.end)
        if not track or frame_w <= 0 or frame_h <= 0:
            print("    [smart crop] Gagal baca video sumber -- pakai crop tengah biasa.")
            return None
        # Kalau semua sampel gagal deteksi wajah, tidak ada gunanya smart
        # crop (hasilnya bakal sama aja kayak crop tengah) -- skip.
        detected = sum(1 for _, v in track if v is not None)
        if detected == 0:
            print(f"    [smart crop] Wajah tidak terdeteksi di {len(track)} sampel -- pakai crop tengah biasa.")
            return None
        print(f"    [smart crop] Wajah terdeteksi di {detected}/{len(track)} sampel, crop mengikuti posisi wajah.")
        smoothed = _fill_and_smooth_track(track)
        crop_w = float(frame_h) if square else frame_h * 9 / 16
        crop_w = min(crop_w, frame_w)  # jaga-jaga video sudah sempit dari sononya
        x_expr = build_pan_expr(smoothed, crop_w, frame_w)
        if square:
            # PENTING: height WAJIB ikut disamakan dengan crop_w di sini.
            # Kalau cuma width yang di-clamp tapi height dibiarkan "ih" (full
            # tinggi asli), hasilnya BUKAN PERSEGI saat video sumbernya
            # portrait (mis. 1080x1920) -- crop_w ke-clamp jadi 1080 tapi
            # tinggi tetap 1920, lalu dipaksa scale ke 1080x1080 bikin
            # videonya gepeng/melar. y dipusatkan vertikal karena belum ada
            # tracking wajah untuk sumbu Y.
            return f"crop=w={crop_w:.0f}:h={crop_w:.0f}:x={x_expr}:y=(ih-{crop_w:.0f})/2"
        return f"crop=w={crop_w:.0f}:h=ih:x={x_expr}:y=0"
    except Exception as e:
        print(f"    [smart crop] Error tak terduga ({e}) -- pakai crop tengah biasa.")
        return None


def cut_clip(video_path: Path, moment: Moment, segments: list[dict],
             out_path: Path, tmp_dir: Path, with_subtitles: bool, aspect_ratio: str = "9:16",
             font_name: str = "Arial Black", watermark_path: Path | None = None,
             speaker_colors: bool = False, font_color: str | None = None,
             font_size_pct: float | None = None, pos_x_pct: float | None = None,
             pos_y_pct: float | None = None, smart_crop: bool = False,
             wm_size_pct: float | None = None, wm_pos_x_pct: float | None = None,
             wm_pos_y_pct: float | None = None):
    if aspect_ratio == "16:9":
        target_w, target_h = 1920, 1080
        # crop tengah horizontal (potong bagian atas/bawah), lalu scale ke 1080p.
        # min(ih\,iw*9/16) menjaga crop tetap valid walau video sumbernya
        # kebetulan bukan landscape standar (mis. sumber portrait/persegi) --
        # tanpa min() ini, ffmpeg bisa minta crop lebih besar dari video
        # aslinya dan gagal total.
        # (smart crop tidak dipakai di sini -- horizontal biasanya cuma
        # motong sedikit bagian atas/bawah, jarang perlu ikut gerak wajah)
        if smart_crop:
            print("    [smart crop] Tidak dipakai untuk rasio 16:9 (cuma motong dikit atas/bawah).")
        vf_filters = [
            "crop=w=iw:h=min(ih\\,iw*9/16)",
            f"scale={target_w}:{target_h}",
        ]
    elif aspect_ratio == "1:1":
        target_w, target_h = 1080, 1080
        crop_filter = _try_build_smart_crop_filter(video_path, moment, square=True) if smart_crop else None
        # min(iw\,ih) dipakai sebagai sisi persegi -- aman untuk video
        # landscape MAUPUN portrait, bukan cuma asumsi landscape.
        vf_filters = [crop_filter or "crop=w=min(iw\\,ih):h=min(iw\\,ih)", f"scale={target_w}:{target_h}"]
    else:
        target_w, target_h = 1080, 1920
        crop_filter = _try_build_smart_crop_filter(video_path, moment, square=False) if smart_crop else None
        # min(iw\,ih*9/16) menjaga crop tetap valid walau video sumbernya
        # sudah sempit/portrait dari sononya.
        vf_filters = [crop_filter or "crop=w=min(iw\\,ih*9/16):h=ih", f"scale={target_w}:{target_h}"]

    if with_subtitles:
        ass_name = f"{out_path.stem}.ass"
        ass_path = tmp_dir / ass_name
        # Posisi kustom (dari slider X/Y, 0-100%) -> koordinat piksel asli.
        # Kalau tidak diisi (None), pakai posisi default (bottom-center,
        # diatur lewat Alignment=2 + MarginV di style ASS).
        pos_xy = None
        if pos_x_pct is not None and pos_y_pct is not None:
            pos_xy = (
                round(target_w * max(0, min(100, pos_x_pct)) / 100),
                round(target_h * max(0, min(100, pos_y_pct)) / 100),
            )
        # Bersihkan nama font dari karakter yang bisa merusak sintaks filter ffmpeg.
        safe_font = re.sub(r"[,:']", "", font_name).strip() or "Arial"
        # FontSize & MarginV dikalibrasi lewat pengukuran render langsung
        # (bukan tebakan). Nilai default (1/145 dari tinggi video) hasilnya
        # blok teks ringkas dan mepet ke tepi bawah. font_size_pct (opsional,
        # dari slider UI, dalam skala 0-100 relatif terhadap nilai default)
        # membiarkan user memperbesar/mengecilkan dari titik kalibrasi itu.
        # Dikalibrasi ULANG khusus untuk sistem .ass (PlayResX/Y eksplisit).
        # Kalibrasi lama (1/145) itu diam-diam mengandalkan bug auto-upscale
        # dari format .srt tanpa PlayRes yang sekarang sudah tidak ada lagi
        # (makanya kalau dipakai di sini hasilnya kekecilan jauh).
        base_font_size = max(20, target_h // 23)
        font_size = base_font_size if font_size_pct is None else max(6, round(base_font_size * font_size_pct / 100))
        margin_v = max(15, target_h // 75)

        # Warna teks: default putih, atau warna kustom dari color picker (hex
        # RRGGBB tanpa '#') dikonversi ke format ASS (&HBBGGRR&, urutan byte
        # dibalik). Kalau speaker_colors aktif, warna per-baris di build_ass
        # yang menang (override warna default style ini).
        primary_colour = "&H00FFFFFF"
        if font_color:
            hexval = re.sub(r"[^0-9A-Fa-f]", "", font_color)[:6].zfill(6)
            rr, gg, bb = hexval[0:2], hexval[2:4], hexval[4:6]
            primary_colour = f"&H00{bb}{gg}{rr}"

        build_ass(segments, moment.start, moment.end, ass_path, safe_font, font_size,
                  primary_colour, margin_v, target_w, target_h, speaker_colors, pos_xy)

        # fontsdir menunjuk ke salinan lokal folder fonts/ di dalam tmp_dir
        # (bukan path absolut) -- supaya jadi nama relatif pendek tanpa
        # spasi/kutip, aman dari bug parsing ffmpeg versi tertentu di Windows.
        fontsdir_opt = ""
        local_fonts = _ensure_local_fontsdir(tmp_dir)
        if local_fonts:
            fontsdir_opt = f":fontsdir={local_fonts}"

        vf_filters.append(f"subtitles={ass_name}{fontsdir_opt}")

    has_watermark = watermark_path is not None and Path(watermark_path).is_file()

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(moment.start),
        "-to", str(moment.end),
        "-i", str(video_path.resolve()),
    ]

    if has_watermark:
        # Logo watermark: input kedua + filter_complex supaya bisa "overlay"
        # logo di atas video hasil crop/scale/subtitle. Ukuran & posisi bisa
        # dikustomisasi (dari drag/resize di panel preview UI); kalau tidak
        # diisi, pakai default lama: 18% lebar video, pojok kiri atas.
        # Path logo aman dipakai langsung (absolute path) karena ini
        # argumen -i terpisah, bukan bagian dari string filter -- jadi
        # tidak kena masalah escape drive-letter seperti file subtitle.
        wm_width = int(target_w * max(5, min(60, wm_size_pct or 18)) / 100)
        if wm_pos_x_pct is not None and wm_pos_y_pct is not None:
            wm_x = round(target_w * max(0, min(100, wm_pos_x_pct)) / 100)
            wm_y = round(target_h * max(0, min(100, wm_pos_y_pct)) / 100)
            # jaga-jaga logo tidak keluar frame (perkiraan tinggi = lebar,
            # karena rasio asli logo baru diketahui ffmpeg saat runtime)
            wm_x = max(0, min(target_w - wm_width, wm_x))
            wm_y = max(0, min(target_h - wm_width, wm_y))
        else:
            wm_x = wm_y = int(target_w * 0.03)
        chain = ",".join(vf_filters)
        filter_complex = (
            f"[0:v]{chain}[base];"
            f"[1:v]scale={wm_width}:-1,format=rgba[wm];"
            f"[base][wm]overlay={wm_x}:{wm_y}[vout]"
        )
        cmd += ["-i", str(Path(watermark_path).resolve())]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "0:a?",
        ]
    else:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(tmp_dir) if with_subtitles else None,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg gagal: {result.stderr[-800:]}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # Command Prompt Windows kadang pakai encoding lama yang tidak dukung
    # semua karakter unicode (mis. judul video dengan tanda hubung khusus).
    # Coba alihkan ke UTF-8 supaya tidak crash pas print().
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="AI Clipper - potong video panjang jadi klip pendek otomatis")
    parser.add_argument("url", help="Link video (YouTube, dll)")
    parser.add_argument("--clips", type=int, default=8, help="Jumlah klip yang dihasilkan (default: 8)")
    parser.add_argument("--subtitles", action="store_true", help="Bakar subtitle otomatis ke video hasil")
    parser.add_argument("--outdir", default="output", help="Folder induk tempat semua hasil disimpan (tiap video otomatis dapat subfolder baru bernomor urut)")
    parser.add_argument("--aspect", choices=["9:16", "16:9", "1:1"], default="9:16", help="Rasio aspek video hasil (default: 9:16 vertikal)")
    parser.add_argument("--font", default="Arial Black", help="Nama font untuk subtitle (default: Arial Black)")
    parser.add_argument("--watermark", default=None, help="Path ke file PNG logo watermark (opsional)")
    parser.add_argument("--speaker-colors", action="store_true", help="Warna subtitle beda tiap pembicara (deteksi dari jeda bicara, bukan diarization asli)")
    args = parser.parse_args()

    base_dir = Path(args.outdir)

    print("Mengambil info video ...")
    try:
        info = get_video_info(args.url)
    except subprocess.CalledProcessError:
        info = {"title": "video", "uploader": "unknown"}
    out_dir = make_output_folder(base_dir, info["title"], info["uploader"])
    print(f"Hasil akan disimpan di: {out_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ai-clipper-"))
    try:
        video_path = download_video(args.url, tmp_dir)
        segments = transcribe(video_path, tmp_dir)
        total_duration = segments[-1]["end"] if segments else 0
        if args.speaker_colors:
            assign_pause_based_speakers(segments)

        moments = find_moments(segments, args.clips, total_duration)
        print(f"    Ditemukan {len(moments)} momen.")

        print(f"[4/5] Memotong {len(moments)} klip (rasio {args.aspect}) ...")
        results = []
        for i, m in enumerate(moments, start=1):
            fname = f"{i:02d}-{slugify(m.title)}.mp4"
            out_path = out_dir / fname
            try:
                cut_clip(video_path, m, segments, out_path, tmp_dir, args.subtitles, args.aspect, args.font, args.watermark, args.speaker_colors)
                print(f"    [{i}/{len(moments)}] {fname}  ({m.duration:.0f}s) - {m.title}")
                results.append({**asdict(m), "file": str(out_path)})
            except (subprocess.CalledProcessError, RuntimeError) as e:
                print(f"    Gagal memotong klip {i}: {e}")

        with open(out_dir / "moments.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[5/5] Selesai. {len(results)} klip tersimpan di: {out_dir.resolve()}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Video sumber & file temporary sudah dihapus.")


if __name__ == "__main__":
    main()