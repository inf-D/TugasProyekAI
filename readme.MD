# Cara Install dan Pemakaian
1. Clone repository
```bash
git clone -b another_version https://github.com/inf-D/TugasProyekAI
```

2. Masuk ke directory
```bash
cd TugasProyekAI
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run create pickle
```bash
python3 create_pickle.py
```

5. Run face register
```bash
python3 face_register.py
```
Saat menjalankan face register, kamu akan diminta untuk menginput nama, inputlah nama panggilanmu dengan <b>TIDAK LEBIH DARI 1 KATA (TANPA SPASI)</b>
Setelah input, jendela akan terbuka untuk membaca data wajahmu, cukup posisikan wajah di tengah-tengah webcam dan berikan gerakan seperti menengok ke kiri dan ke kanan.

6. Run face train
```bash
python3 face_train.py
```

7. Run face validation
```bash
python3 face_validation.py
```

#### Untuk mendaftarkan wajah baru, cukup lakukan step dari nomor 5.

# PROSES
## 1. PENGAMBILAN DATA
Pengambilan data menggunakan program Python. Program yang dibuat akan meminta kita untuk memasukkan nama, lalu membuka webcam dan mendeteksi wajah menggunakan algoritma haar cascade. Data wajah yang didapatkan akan dilabel ke dalam folder berdasarkan nama yang dimasukkan, kemudian mengalami pemrosesan seperti konversi warna ke grayscale dan <i>cropping</i> 

<figure>
<a href="https://ibb.co/TwmXrDM"><img src="https://i.ibb.co/CwhdWpK/Screenshot-from-2023-06-11-18-53-41.png" alt="Screenshot-from-2023-06-11-18-53-41" border="0"></a>
<figcaption><em>Gambar 2 Kode Pengambilan Data</em></figcaption>
</figure><br>
<figure>
<a href="https://ibb.co/zPtqMck"><img src="https://i.ibb.co/tY1VGfd/Screenshot-from-2023-06-11-19-06-24.png" alt="Screenshot-from-2023-06-11-19-06-24" border="0"></a>
<figcaption><em>Gambar 3 Hasil Pengambilan Data Dengan Algoritma Haar Cascade</em></figcaption>
</figure>

## 2. PEMBUATAN SISTEM
### A. TAHAPAN TRAIN
Dataset yang telah dikumpulkan di-<i>train</i> menggunakan algoritma LBPH yang ada pada module opencv pada bahasa pemrograman Python. Tiap gambar yang telah dilabel dan diproses pada dataset dimuat dan dilatih menggunakan algoritma yang telah disebutkan, lalu disimpan ke dalam file trainer.yml.
<figure>
<a href="https://ibb.co/LpbfhHd"><img src="https://i.ibb.co/12hxJjG/Screenshot-from-2023-06-11-19-15-50.png" alt="Screenshot-from-2023-06-11-19-15-50" border="0"></a>
<figcaption><em>Gambar 4 Kode Training Dataset</em></figcaption>
</figure>

### B. Tahapan Testing
Testing dilakukan dengan mengecek apakah program yang telah dibuat berhasil mengumpulkan data wajah dari webcam, menyimpan hasil latihan dataset ke dalam file trainer.xml, dan melakukan pengenalan wajah berdasarkan dataset yang telah ditrain.
Dari pengujian yang dilakukan, program berhasil membaca data dari webcam, menyimpan hasil latihan dataset, dan melakukan pengenalan wajah melalui webcam.
<figure>
<a href="https://ibb.co/tCrbmmD"><img src="https://i.ibb.co/cTnNggQ/Screenshot-from-2023-06-11-19-23-30.png" alt="Screenshot-from-2023-06-11-19-23-30" border="0"></a>
<figcaption><em>Gambar 5 Pengumpulan Data Wajah dari Webcam</em></figcaption>
</figure><br> 
<figure>
<a href="https://ibb.co/bN3CBYx"><img src="https://i.ibb.co/xzsBjwy/Screenshot-from-2023-06-11-19-26-25.png" alt="Screenshot-from-2023-06-11-19-26-25" border="0"></a>
<figcaption><em>Gambar 6 Hasil Train Dataset</em></figcaption>
</figure><br>
<figure>
<a href="https://ibb.co/8BStMMw"><img src="https://i.ibb.co/GJbN77D/Screenshot-from-2023-06-11-19-28-26.png" alt="Screenshot-from-2023-06-11-19-28-26" border="0"></a>
<figcaption><em>Gambar 7 Pengenalan Wajah melalui Webcam</em></figcaption>
</figure><br>

Referensi:
- [Build Your Own Face Recognition Tool With Python oleh Kyle Stratis](https://realpython.com/face-recognition-with-python/)
- [Face-Recognition-LBPH oleh RitvikDayal](https://github.com/RitvikDayal/Face-Recognition-LBPH)
