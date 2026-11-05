# Crypto Forecasting: Attention-LSTM

Aplikasi Flask untuk menampilkan prediksi harga penutupan BTC, ETH, dan BNB menggunakan model Attention-LSTM.

## Menjalankan aplikasi

Gunakan Python dengan dependensi yang tercantum di `requirements.txt`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Buka alamat lokal yang ditampilkan Flask di terminal. Aplikasi mengambil data harga terbaru dari Yahoo Finance, sehingga koneksi internet diperlukan.

## Model dan scaler

File model `.keras` di `models/` merupakan model hasil training. Aplikasi tidak memuat file tersebut secara langsung. Skrip `convert_fix_lambda.py` membangun ulang arsitektur Attention-LSTM, memuat bobot dari file `.keras`, lalu mengekspornya sebagai SavedModel ke `models_fixed/`. Aplikasi memuat SavedModel tersebut melalui `keras.layers.TFSMLayer`.

Scaler bukan bagian yang ditanamkan ke model. Statistik `StandardScaler` yang dihasilkan saat training disimpan terpisah sebagai file `.npz` di `scalers_npz/`. Saat prediksi, aplikasi menormalkan input harga dengan scaler tersebut, menjalankan model, lalu mengubah output prediksi kembali ke skala harga asli. File scaler harus berasal dari training yang sama dengan bobot model agar preprocessing dan hasil prediksi konsisten.

Struktur artefak yang digunakan aplikasi:

- `models_fixed/`: SavedModel untuk inferensi.
- `scalers_npz/`: parameter scaler untuk masing-masing aset.
- `models/`: file `.keras` sumber yang dipakai skrip konversi; tidak dimuat langsung oleh aplikasi.

Jika artefak model atau scaler diganti, pastikan mapping aset dan path di `app.py` tetap sesuai.

## Alur prediksi saat ini

Untuk setiap aset yang dipilih, aplikasi mengambil data penutupan dari Yahoo Finance, membentuk input dari 60 harga penutupan terakhir (nilai `lookback` di `config.json`), menormalkannya, lalu meminta model menghasilkan prediksi. Output prediksi kemudian dikembalikan ke satuan harga asli.

**Catatan:** walaupun nama artefak mengandung `H1`, implementasi `app.py` saat ini meminta data dengan interval `1d` dan menggunakan periode `2y`. Jadi prediksi yang ditampilkan mengikuti data harian, bukan data per jam. Nilai `horizon` di `config.json` saat ini belum digunakan untuk mengubah interval atau panjang prediksi.

`app.py` menjalankan Flask dengan `debug=True`, yang ditujukan untuk pengembangan lokal. Jangan gunakan server debug Flask sebagai server deployment publik.
