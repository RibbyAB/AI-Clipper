# AI Clipper

Otomatis mengubah video panjang (dari link) menjadi beberapa klip pendek
vertikal (9:16) siap upload ke TikTok / Reels / YouTube Shorts, memakai AI
untuk memilih momen paling menarik.

Video sumber hanya di-download sementara ke folder temp dan **otomatis
dihapus** setelah proses selesai — tidak menuhin storage.

## Alur kerja

```
Link video --> download sementara --> transkrip (Whisper)
           --> AI cari momen menarik (Claude) --> potong + crop vertikal
           --> (opsional) subtitle otomatis --> klip siap upload
```

Auto-upload ke TikTok/Reels/YouTube belum termasuk di versi ini karena
butuh akun developer & OAuth di masing-masing platform — nanti ditambahkan
sebagai tahap berikutnya begitu akunnya siap.

## Instalasi

1. **Python 3.10+** dan **ffmpeg** harus sudah terpasang di komputer.
   Cek ffmpeg: `ffmpeg -version` (kalau belum ada, install lewat
   [ffmpeg.org](https://ffmpeg.org/download.html) atau `brew install ffmpeg` / `apt install ffmpeg`).

2. Install dependency Python:
   ```bash
   pip install -r requirements.txt
   ```

3. Siapkan API key Groq (**gratis**, untuk deteksi momen menarik):
   - Daftar/masuk ke [console.groq.com](https://console.groq.com) (cukup pakai akun Google, tidak perlu kartu kredit)
   - Buat API key di menu **API Keys**
   - Set sebagai environment variable:
     ```bash
     export GROQ_API_KEY="gsk_xxxxxxxx"
     ```
     (Windows PowerShell: `$env:GROQ_API_KEY="gsk_xxxxxxxx"`)
   - Free tier Groq punya batas jumlah request per hari/menit (cukup untuk pemakaian normal). Kalau kena limit, tunggu beberapa saat lalu coba lagi.

## Mempercepat pakai GPU NVIDIA (otomatis)

App ini pakai `faster-whisper`, yang otomatis mendeteksi dan memakai GPU
NVIDIA kalau tersedia -- tidak perlu instalasi tambahan apa pun. Kalau GPU
tidak terdeteksi atau gagal dipakai karena alasan apa pun, otomatis jalan
di CPU sebagai gantinya (lebih lambat, tapi tetap berfungsi).

Cek driver NVIDIA sudah terpasang (opsional, cuma buat memastikan): buka
Command Prompt, ketik `nvidia-smi`. Kalau muncul tabel info GPU, driver
sudah siap. Kalau belum ada, install driver terbaru dari
[nvidia.com/drivers](https://www.nvidia.com/drivers).

Saat pertama kali dijalankan, model Whisper akan didownload otomatis dari
Hugging Face (butuh koneksi internet, cuma sekali di awal lalu tersimpan
di cache lokal untuk pemakaian berikutnya).

## Cara pakai

```bash
python clipper.py "https://youtube.com/watch?v=xxxxxxxx" --clips 8
```

Opsi yang tersedia:

| Opsi | Default | Keterangan |
|---|---|---|
| `--clips N` | 8 | Jumlah klip yang dihasilkan |
| `--subtitles` | mati | Bakar subtitle otomatis ke video |
| `--outdir folder` | `output` | Folder tempat klip disimpan |

Contoh dengan subtitle:
```bash
python clipper.py "https://youtube.com/watch?v=xxxxxxxx" --clips 10 --subtitles
```

## Output

Setelah selesai, folder `output/` berisi:
- `01-judul-klip.mp4`, `02-judul-klip.mp4`, dst — klip video vertikal siap upload
- `moments.json` — detail tiap klip (timestamp asli, judul, alasan AI memilihnya)

## Catatan performa

- Model Whisper default adalah `small` (cukup akurat, tidak terlalu berat).
  Untuk hasil transkrip lebih akurat (tapi lebih lambat & butuh GPU/RAM
  lebih besar), set:
  ```bash
  export CLIPPER_WHISPER_MODEL=medium
  ```
- Video panjang (>1 jam) bisa memakan waktu cukup lama untuk transkrip
  di komputer tanpa GPU. Ini normal.

## Rencana pengembangan selanjutnya

- [ ] Auto-upload ke TikTok (butuh TikTok Developer App + Content Posting API)
- [ ] Auto-upload ke YouTube Shorts (butuh Google Cloud project + YouTube Data API)
- [ ] Auto-upload ke Instagram Reels (butuh Meta App + Instagram Graph API)
- [ ] Mode "watch folder" biar tinggal taruh link di file lalu diproses otomatis