# Laporan Proyek Machine Learning - Fajar

## Project Overview

![image](https://github.com/user-attachments/assets/f907bbbb-fffe-4324-942f-16cf1209502b)


Di era digital saat ini, jumlah informasi dan pilihan yang tersedia bagi konsumen meningkat secara eksponensial. Dalam konteks buku,pembaca seringkali dihadapkan pada jutaan judul yang tersedia di berbagai platform, baik fisik maupun digital. Hal ini dapat menyulitkan pembaca untuk menemukan buku yang sesuai dengan minat dan preferensi mereka. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menavigasi banyaknya pilihan tersebut dengan menyajikan item (dalam hal ini, buku) yang paling relevan.

Proyek ini berfokus pada pengembangan sistem rekomendasi buku menggunakan dataset yang berisi informasi mengenai pengguna, buku, dan peringkat yang diberikan oleh pengguna terhadap buku. Dengan memanfaatkan teknik machine learning, sistem ini diharapkan dapat memberikan rekomendasi buku yang dipersonalisasi kepada pengguna.

**Penyelesaian proyek ini penting karena beberapa alasan:**

1. **Meningkatkan Pengalaman Pengguna:** Dengan memberikan rekomendasi yang relevan, pengguna dapat lebih mudah menemukan buku yang mereka sukai, sehingga meningkatkan kepuasan dan keterlibatan mereka.

2. **Membantu Penemuan Konten:** Sistem rekomendasi dapat membantu pengguna menemukan buku-buku baru yang mungkin tidak akan mereka temukan sendiri, memperluas wawasan dan minat baca mereka.

3. **Potensi Aplikasi Bisnis:** Bagi platform penjualan buku atau perpustakaan digital, sistem rekomendasi yang efektif dapat meningkatkan penjualan, peminjaman, dan loyalitas pengguna.

Penelitian menunjukkan bahwa sistem rekomendasi memainkan peran penting dalam meningkatkan pengalaman pengguna dan nilai bisnis. Menurut Ricci et al. (2015), sistem rekomendasi tidak hanya membantu dalam menyaring informasi tetapi juga meningkatkan keterlibatan pengguna dan kepuasan pelanggan. Selain itu, studi oleh Zhang et al. (2019) menyebutkan bahwa pendekatan hybrid yang menggabungkan content-based dan collaborative filtering terbukti efektif dalam meningkatkan akurasi dan relevansi rekomendasi.

Dataset yang digunakan dalam proyek ini adalah Book Recommendation Dataset yang bersumber dari Kaggle. Dataset ini umum digunakan untuk membangun sistem rekomendasi karena kelengkapan informasinya yang mencakup interaksi pengguna-item (peringkat) serta metadata item (judul buku, penulis, penerbit).

### Referensi:

Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6

Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR), 52(1), 1–38. https://doi.org/10.1145/3285029

## Business Understanding

Proses klarifikasi masalah melibatkan pemahaman mendalam terhadap tantangan yang dihadapi pengguna dalam menemukan buku yang relevan di tengah banyaknya pilihan. Pengguna seringkali menghabiskan banyak waktu untuk mencari buku atau bahkan tidak menemukan buku yang sesuai dengan preferensi mereka.

### Problem Statements
Berikut adalah pernyataan masalah yang ingin dipecahkan melalui proyek ini:

1. Bagaimana cara membantu pengguna menemukan buku yang paling sesuai dengan preferensi baca mereka secara efisien?
2. Bagaimana cara memberikan rekomendasi buku yang dipersonalisasi berdasarkan riwayat baca dan peringkat pengguna lain dengan preferensi serupa?
3. Bagaimana cara merekomendasikan buku berdasarkan kemiripan konten (judul, penulis, penerbit) dengan buku yang pernah disukai pengguna?

### Goals
Tujuan dari proyek ini adalah sebagai berikut:

1. Mengembangkan sistem rekomendasi buku yang mampu memberikan daftar buku yang relevan kepada pengguna.
2. Mengimplementasikan dua pendekatan sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering.
3. Mengevaluasi kinerja kedua model rekomendasi menggunakan metrik yang sesuai untuk menentukan efektivitasnya.

### Solution Statements
Untuk mencapai tujuan di atas, dua pendekatan solusi (algoritma sistem rekomendasi) akan diajukan dan diimplementasikan:

1. Content-Based Filtering:

    - Cara Kerja: Pendekatan ini merekomendasikan buku berdasarkan kemiripan atribut atau konten buku itu sendiri. Fitur-fitur seperti judul, penulis, penerbit, dan tahun terbit akan digunakan untuk membuat profil setiap buku. Sistem kemudian akan merekomendasikan buku yang memiliki profil serupa dengan buku yang pernah disukai atau diberi peringkat tinggi oleh pengguna.

    - Teknik: Menggunakan TF-IDF Vectorizer untuk mengubah fitur teks menjadi representasi numerik dan NearestNeighbors (dengan metrik cosine similarity) untuk menemukan buku-buku yang paling mirip.

2. Collaborative Filtering:

    - Cara Kerja: Pendekatan ini merekomendasikan buku berdasarkan pola perilaku pengguna. Jika pengguna A memiliki preferensi yang mirip dengan pengguna B (misalnya, mereka menyukai buku-buku yang sama), maka buku yang disukai pengguna B tetapi belum dibaca oleh pengguna A akan direkomendasikan kepada pengguna A.

    - Teknik: Menggunakan model berbasis Neural Network dengan embedding layers untuk mempelajari representasi laten dari pengguna dan buku. Model ini akan dilatih untuk memprediksi peringkat yang mungkin diberikan pengguna terhadap buku.

## Data Understanding

Dataset yang digunakan dalam proyek ini terdiri dari tiga file CSV yang diperoleh dari [Book Recommendation Dataset on Kaggle.](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

- Users.csv: Berisi informasi pengguna (278,858 entri).
- Books.csv: Berisi informasi detail buku (271,360 entri).
- Ratings.csv: Berisi informasi peringkat (1,149,780 entri).

**Kondisi Data Awal (Missing Values):**

- Users.csv: Age (110,762 nilai hilang).
- Books.csv: Book-Author (2), Publisher (2), Image-URL-L (3 nilai hilang).
- Ratings.csv: Tidak ada nilai hilang.
Tidak ada data duplikat pada ketiga dataset awal.

**Variabel-variabel pada dataset adalah sebagai berikut:**

1. Users.csv:
    - User-ID (diubah menjadi User_id): ID unik untuk setiap pengguna (tipe: int64).
    - Location: Lokasi pengguna (tipe: object/string).
    - Age: Usia pengguna (tipe: float64).
2. Books.csv:
    - ISBN: Nomor ISBN unik untuk setiap buku (tipe: object/string).
    - Book-Title (diubah menjadi Title): Judul buku (tipe: object/string).
    - Book-Author (diubah menjadi Author): Penulis buku (tipe: object/string).
    - Year-Of-Publication (diubah menjadi Year): Tahun publikasi buku (tipe: object/string).
    - Publisher: Penerbit buku (tipe: object/string).
    - Image-URL-S, Image-URL-M, Image-URL-L: URL gambar sampul buku (tipe: object/string).

3. Ratings.csv:
    - User-ID (diubah menjadi User_id): ID pengguna yang memberikan peringkat (tipe: int64).
    - ISBN: ISBN buku yang diberi peringkat (tipe: object/string).
    - Book-Rating (diubah menjadi Rating): Peringkat yang diberikan (skala 0-10, tipe: int64).

**Analisis Data Eksploratif (EDA):**
- Keunikan Buku: Terdapat 242,135 judul unik dan 271,360 ISBN unik.
- Skala Data Rating: Terdapat 1,149,780 peringkat dari 105,283 pengguna untuk 340,556 buku (setelah diproses untuk collab_df).
- Rentang Rating: Skala 0-10, dengan banyak peringkat 0 yang difilter untuk Collaborative Filtering.

![image](https://github.com/user-attachments/assets/807dfeb0-ba6c-4a4b-90a1-69c51ef541b6)

Distribusi Penerbit dan Penulis Teratas: Berdasarkan dari Visualisasi menunjukkan penerbit seperti Harlequin dan penulis seperti Agatha Christie memiliki banyak entri.

## Data Preparation
Tahapan persiapan data dilakukan untuk memastikan data siap digunakan untuk pemodelan.

**1. Pengubahan Nama Kolom:** Nama kolom diubah untuk konsistensi (misalnya, **Book-Rating** menjadi **Rating**).
   - Alasan: Memudahkan penggabungan dan pemanggilan kolom.
     
**2. Penanganan Nilai Hilang:**
- Entri tanpa Title atau ISBN di books dihapus. Alasan: ISBN dan Title krusial.
- Baris dengan nilai hilang di ratings dihapus (praktik baik).
- Author dan Publisher yang hilang di books diisi 'Unknown'. **Alasan:** Mempertahankan data dan memungkinkan penggunaan fitur.

**3. Penggabungan Data** (Merge): **ratings** dan **books** digabung berdasarkan **ISBN** menjadi **merged**. 
- Alasan: Membuat dataset tunggal dengan informasi peringkat dan detail buku.
     
**4. Pembuatan Fitur Gabungan untuk Content-Based:** Kolom **features** (gabungan **Title**, **Author**, **Publisher**, **Year**) dibuat di **merged**.
- Alasan: Membuat representasi teks tunggal untuk TF-IDF.
     
**5. Filtering Data Peringkat untuk Collaborative:** merged difilter untuk Rating > 0 menjadi filtered_rating.
- Alasan: Fokus pada preferensi eksplisit.
     
**6. Pemisahan Dataset:**
- **content_df** (**ISBN**, **Title**, features unik) untuk Content-Based. Alasan: Data konten bersih.
- collab_df (User_id, ISBN, Rating dari filtered_rating) untuk Collaborative. Alasan: Data interaksi bersih.
**7. Encoding Fitur untuk Collaborative:**

- **ISBN** dipastikan string; ISBN di **collab_df** divalidasi terhadap **content_df**. Alasan: Untuk memastikan konsistensi dan menghindari **ISBN** yang tidak memiliki pasangan data buku.
- **LabelEncoder** mengubah **User_id** menjadi **user** dan **ISBN** menjadi **book**. Alasan: Input numerik untuk embedding.
- Dibuat mapping ID asli ke terenkode dan sebaliknya. **Alasan:** Memudahkan inferensi.
- **collab_final_df** (**user**, **book**, **Rating**) dibuat.

**8. Pembagian Data Latih/Validasi:** collab_final_df dibagi 80% latih, 20% validasi. Alasan: Evaluasi generalisasi model.

**Hasil Data Setelah Persiapan:**

- content_df: 340,556 buku unik.
- collab_df: 433,671 interaksi.

## Modeling
Dua model sistem rekomendasi dikembangkan:
**1. Content-Based Filtering**
Merekomen­dasikan buku berdasarkan kemiripan konten.

**Proses Pembuatan Model:**

1. TF-IDF Vectorization: Fitur features (judul, penulis, penerbit, tahun) dari content_df diubah menjadi matriks numerik (maks 10.000 fitur, stop words Inggris).

2. Nearest Neighbors Model: Menggunakan NearestNeighbors (n_neighbors=11, metric='cosine', algorithm='brute') untuk mencari 10 buku termirip berdasarkan matriks TF-IDF.

3. Pemetaan Judul ke Indeks: Dibuat pandas.Series untuk memetakan judul ke indeks.

Fungsi Rekomendasi (recommend_books_nn):
Menerima judul buku, mencari top_n (default 10) buku termirip menggunakan model NearestNeighbors, lalu mengambil detail lengkap buku rekomendasi dari merged.

Hasil Rekomendasi (Top-N):
Untuk buku 'Death on the Nile':

| No | Title                  | Author          | Publisher                                | Year |
|----|------------------------|-----------------|------------------------------------------|------|
| 1  | At Bertram's Hotel     | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 2  | Death Comes As the End | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 3  | Passenger to Frankfurt | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 4  | Sparkling Cyanide      | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 5  | Sleeping Murder        | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 6  | Mrs. McGinty's Dead    | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 7  | Murder Is Easy         | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 8  | The Clocks             | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1991 |
| 9  | Hickory Dickory Dock   | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |
| 10 | Third Girl             | Agatha Christie | Harper Mass Market Paperbacks (Mm)       | 1992 |

Model berhasil merekomendasikan buku lain karya Agatha Christie.

**Kelebihan:** Tidak butuh data pengguna lain, bisa rekomendasikan item baru, transparan.
**Kekurangan:** Terbatas fitur item, bisa overspecialization, butuh domain knowledge, cold-start untuk pengguna baru.

2. Collaborative Filtering
Merekomen­dasikan buku berdasarkan kemiripan preferensi antar pengguna.

**Proses Pembuatan Model:**

1. Parameter: num_users (pengguna unik terenkode), num_books (buku unik terenkode), embedding_size = 50.

2. Arsitektur Model (Neural Network - Keras):

- Input: user dan book (ID terenkode).

- Embedding Layers: Untuk user dan book (dimensi 50, inisialisasi he_normal, regularisasi L2).

- Flatten & Concatenate: Menggabungkan vektor embedding.

- Dense Layers: Dua lapisan (128, 64 unit, aktivasi ReLU).

- Output Layer: Satu unit (aktivasi linear) untuk prediksi peringkat.

- Kompilasi: Optimizer Adam (lr 0.001), loss mean_squared_# Ringkasan arsitektur model (dari notebook)
  

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
