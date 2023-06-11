### Cara Install dan Pemakaian

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
Saat menjalankan face register, kamu akan diminta untuk menginput nama, inputlah nama panggilanmu dengan TIDAK LEBIH DARI 1 KATA (TANPA SPASI)
Setelah input, jendela akan terbuka untuk membaca data wajahmu, cukup posisikan wajah di tengah-tengah webcam dan berikan gerakan seperti menengok ke kiri dan ke kanan.

6. Run face train
```bash
python3 face_train.py
```

7. Run face validation
```bash
python3 face_validation.py
```

Untuk mendaftarkan wajah baru, cukup lakukan step dari nomor 5.

