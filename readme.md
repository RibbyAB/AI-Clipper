# AI Clipper

Web app lokal yang otomatis mengubah video panjang (dari link YouTube) menjadi
beberapa klip pendek siap upload ke TikTok / Reels / YouTube Shorts —
lengkap dengan caption dan hashtag yang di-generate AI.

**Tech stack:** Python, Flask, faster-whisper (speech-to-text), Groq LLM API,
ffmpeg, yt-dlp, vanilla JS/HTML/CSS.

## Fitur

- 🔗 **Queue banyak link sekaligus** — tempel beberapa link video, diproses berurutan otomatis
- 🤖 **AI cari momen menarik otomatis** dari transkrip video (Groq LLM, gratis)
- ✂️ **Auto-crop** ke rasio 9:16 (vertikal) atau 16:9 (horizontal)
- 📝 **Subtitle otomatis** (opsional, di-burn langsung ke video)
- ✍️ **Caption + hashtag siap-pakai** untuk tiap klip, tinggal copy-paste
- ⚡ **GPU-accelerated** transcription (otomatis fallback ke CPU kalau GPU tidak tersedia)
- 🔔 Notifikasi browser saat proses selesai
- 📦 Download per-klip, ZIP semua sekaligus, atau buka folder hasil langsung

## Alur kerja

```
Link video --> download sementara --> transkrip (faster-whisper)
           --> AI cari momen menarik + caption/hashtag (Groq)
           --> potong + crop + subtitle (ffmpeg) --> klip siap upload
```

Video sumber hanya di-download sementara dan **otomatis dihapus** setelah
proses selesai — tidak menuhin storage jangka panjang.

## Screenshot

<!-- TODO: tempel screenshot/GIF web UI di sini -->

## Instalasi

1. **Python 3.10+** dan **ffmpeg** harus sudah terpasang di komputer.
   Cek ffmpeg: `ffmpeg -version` (kalau belum ada, install lewat
   [ffmpeg.org](https://ffmpeg.org/download.html) atau `brew install ffmpeg` / `apt install ffmpeg`).

2. Install dependency Python:
   ```bash
   pip install -r requirements.txt
   ```

3. Siapkan API key Groq (**gratis**, untuk deteksi momen menarik & generate caption):
   - Daftar/masuk ke [console.groq.com](https://console.groq.com)
   - Buat API key di menu **API Keys**
   - Set sebagai environment variable:
     ```bash
     export GROQ_API_KEY="gsk_xxxxxxxx"
     ```
     (Windows PowerShell: `$env:GROQ_API_KEY="gsk_xxxxxxxx"`)

## Mempercepat pakai GPU NVIDIA (otomatis)

App ini pakai `faster-whisper`, yang otomatis mendeteksi dan memakai GPU
NVIDIA kalau tersedia — tidak perlu instalasi tambahan apa pun. Kalau GPU
gagal dipakai karena alasan apa pun, otomatis fallback ke CPU.

## Cara pakai

**Web UI (disarankan):**
```bash
python app.py
```
Browser otomatis terbuka ke `http://127.0.0.1:5000`. Tempel link video
(satu per baris untuk beberapa video sekaligus), atur jumlah klip & rasio,
klik "Buat klip".

**CLI (command line):**
```bash
python clipper.py "https://youtube.com/watch?v=xxxxxxxx" --clips 8 --aspect 9:16
```

| Opsi | Default | Keterangan |
|---|---|---|
| `--clips N` | 8 | Jumlah klip yang dihasilkan |
| `--subtitles` | mati | Bakar subtitle otomatis ke video |
| `--aspect` | `9:16` | Rasio output: `9:16` (vertikal) atau `16:9` (horizontal) |
| `--outdir folder` | `output` | Folder tempat klip disimpan |

## Output

Tiap video dapat subfolder sendiri di `output/`, berisi:
- `01-judul-klip.mp4`, `02-judul-klip.mp4`, dst — klip video siap upload
- `moments.json` — detail tiap klip (timestamp, judul, caption, hashtag)

## Arsitektur singkat

- `clipper.py` — core pipeline: download (yt-dlp) → transkrip (faster-whisper)
  → cari momen + caption (Groq LLM API) → potong & crop (ffmpeg)
- `app.py` — Flask server, queue management, REST API untuk web UI
- `templates/index.html` — single-page web UI (vanilla JS, tanpa framework)

## Rencana pengembangan selanjutnya

- [ ] Auto-upload ke TikTok (Content Posting API, menunggu approval developer app)
- [ ] Auto-upload ke YouTube Shorts / Instagram Reels
- [ ] Mode "watch folder"