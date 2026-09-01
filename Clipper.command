# Permission dulu kalau baru pertama kali make file ini: chmod +x Clipper.command

cd "$(dirname "$0")"

echo "=================================="
echo "  AI Clipper - starting server..."
echo "=================================="
echo ""

# Cek python3 tersedia
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 tidak ditemukan. Install dulu dari https://www.python.org/downloads/"
    echo "Tekan Enter buat nutup jendela ini."
    read
    exit 1
fi

# Kalau ada virtual environment (venv), aktifkan
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Buka browser otomatis ke halaman clipper setelah 2 detik (kasih waktu server nyala dulu)
(sleep 2 && open "http://127.0.0.1:5000") &

# Jalankan server (biarkan jendela terminal ini tetap terbuka selama dipakai)
python3 web_app.py

echo ""
echo "Server berhenti. Tekan Enter buat nutup jendela ini."
read
