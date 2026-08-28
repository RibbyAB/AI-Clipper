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
    """Ambil judul & nama creator video tanpa perlu download videonya."""
    cmd = ["yt-dlp", "--skip-download", "--print", "%(title)s", "--print", "%(uploader)s", url]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (result.stdout or "").strip().split("\n")
    title = lines[0] if len(lines) > 0 else "video"
    uploader = lines[1] if len(lines) > 1 else "unknown"
    return {"title": title, "uploader": uploader}


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


def build_srt(segments: list[dict], start: float, end: float, out_path: Path):
    """Bikin file .srt untuk potongan waktu start-end, timestamp direlatifkan ke 0."""
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    idx = 1
    lines = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        rel_start = max(seg["start"], start) - start
        rel_end = min(seg["end"], end) - start
        lines.append(str(idx))
        lines.append(f"{fmt(rel_start)} --> {fmt(rel_end)}")
        lines.append(seg["text"])
        lines.append("")
        idx += 1

    out_path.write_text("\n".join(lines), encoding="utf-8")


def cut_clip(video_path: Path, moment: Moment, segments: list[dict],
             out_path: Path, tmp_dir: Path, with_subtitles: bool, aspect_ratio: str = "9:16"):
    if aspect_ratio == "16:9":
        # crop tengah horizontal (potong bagian atas/bawah), lalu scale ke 1080p
        vf_filters = [
            "crop=iw:iw*9/16",
            "scale=1920:1080",
        ]
    else:
        # crop tengah vertikal (potong kiri/kanan), lalu scale ke ukuran standar
        vf_filters = [
            "crop=ih*9/16:ih",
            "scale=1080:1920",
        ]

    if with_subtitles:
        srt_name = f"{out_path.stem}.srt"
        srt_path = tmp_dir / srt_name
        build_srt(segments, moment.start, moment.end, srt_path)
        # Jalankan ffmpeg dengan cwd=tmp_dir dan rujuk file SRT cukup pakai
        # NAMA FILE saja (bukan path lengkap) -- ini menghindari sepenuhnya
        # masalah klasik ffmpeg-di-Windows yang salah nafsirin drive letter
        # (C:) dan backslash pada path sebagai bagian dari sintaks filter.
        vf_filters.append(
            f"subtitles={srt_name}:force_style='FontName=Arial,FontSize=16,"
            "PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,"
            "Outline=2,Alignment=2,MarginV=60'"
        )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(moment.start),
        "-to", str(moment.end),
        "-i", str(video_path.resolve()),
        "-vf", ",".join(vf_filters),
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
    parser.add_argument("--aspect", choices=["9:16", "16:9"], default="9:16", help="Rasio aspek video hasil (default: 9:16 vertikal)")
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

        moments = find_moments(segments, args.clips, total_duration)
        print(f"    Ditemukan {len(moments)} momen.")

        print(f"[4/5] Memotong {len(moments)} klip (rasio {args.aspect}) ...")
        results = []
        for i, m in enumerate(moments, start=1):
            fname = f"{i:02d}-{slugify(m.title)}.mp4"
            out_path = out_dir / fname
            try:
                cut_clip(video_path, m, segments, out_path, tmp_dir, args.subtitles, args.aspect)
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