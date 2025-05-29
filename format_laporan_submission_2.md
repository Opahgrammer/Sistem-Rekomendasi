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
     
**1. Penanganan Nilai Hilang:**
```python
books.dropna(subset=['Title', 'ISBN'], inplace=True)
rating.dropna(inplace=True)

books['Author'] = books['Author'].fillna('Unknown')
books['Publisher'] = books['Publisher'].fillna('Unknown')
```
- Entri tanpa Title atau ISBN di books dihapus. Alasan: ISBN dan Title krusial.
- Baris dengan nilai hilang di ratings dihapus (praktik baik).
- Author dan Publisher yang hilang di books diisi 'Unknown'. **Alasan:** Mempertahankan data dan memungkinkan penggunaan fitur.

**2. Penggabungan Data** (Merge): **ratings** dan **books** digabung berdasarkan **ISBN** menjadi **merged**. 
```python
merged = pd.merge(rating, books[['ISBN', 'Title', 'Author', 'Publisher','Year']], on='ISBN', how='left')
```
- Alasan: Membuat dataset tunggal dengan informasi peringkat dan detail buku.
     
**3. Pembuatan Fitur Gabungan untuk Content-Based:** Kolom **features** (gabungan **Title**, **Author**, **Publisher**, **Year**) dibuat di **merged**.
```python
merged['features'] = (
    merged['Title'].fillna('') + ' ' +
    merged['Author'].fillna('') + ' ' +
    merged['Publisher'].fillna('') + ' ' +
    merged['Year'].fillna('').astype(str)
)
```
- Alasan: Membuat representasi teks tunggal untuk TF-IDF.
     
**4. Filtering Data Peringkat untuk Collaborative:** merged difilter untuk Rating > 0 menjadi filtered_rating.
```python
filtered_rating = merged[merged['Rating'] > 0]
```
- Alasan: Fokus pada preferensi eksplisit.

**5. Pemisahan Dataset:**
```python
content_df = merged[['ISBN', 'Title', 'features']].drop_duplicates().reset_index(drop=True)
collab_df = filtered_rating[['User_id', 'ISBN', 'Rating']]
```
- **content_df** (**ISBN**, **Title**, features unik) untuk Content-Based. Alasan: Data konten bersih.
- collab_df (User_id, ISBN, Rating dari filtered_rating) untuk Collaborative. Alasan: Data interaksi bersih.

 **6. Vektorisasi Teks (TF-IDF) (Untuk *Content-Based*):**
    ```python
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(content_df['features'])
    
- **Proses**: Mengaplikasikan `TfidfVectorizer` pada kolom `features` yang telah dibuat.
- **Hasil**: Dihasilkan `tfidf_matrix`, sebuah matriks numerik yang merepresentasikan setiap buku sebagai vektor.
- **Alasan**: Model *machine learning* tidak bisa memproses teks mentah. **TF-IDF mengubah teks menjadi vektor numerik** yang merepresentasikan pentingnya setiap kata dalam konteks buku dan keseluruhan dataset. Vektor ini memungkinkan model **mengukur kemiripan antar buku** secara matematis.
        
**7. Encoding Fitur untuk Collaborative:**
```python
content_df['ISBN'] = content_df['ISBN'].astype(str)

valid_isbns = set(content_df['ISBN'].unique())
collab_df = collab_df[collab_df['ISBN'].isin(valid_isbns)]
collab_df['ISBN'] = collab_df['ISBN'].astype(str)


user_encoder = LabelEncoder()
isbn_encoder = LabelEncoder()

collab_df['user'] = user_encoder.fit_transform(collab_df['User_id'])
collab_df['book'] = isbn_encoder.fit_transform(collab_df['ISBN'])

user_id_to_encoded = dict(zip(collab_df['User_id'], collab_df['user']))
isbn_to_encoded = dict(zip(collab_df['ISBN'], collab_df['book']))
encoded_to_user_id = {v: k for k, v in user_id_to_encoded.items()}
encoded_to_isbn = {v: k for k, v in isbn_to_encoded.items()}

collab_final_df = collab_df[['user', 'book', 'Rating']]
```
- **ISBN** dipastikan string; ISBN di **collab_df** divalidasi terhadap **content_df**. Alasan: Untuk memastikan konsistensi dan menghindari **ISBN** yang tidak memiliki pasangan data buku.
- **LabelEncoder** mengubah **User_id** menjadi **user** dan **ISBN** menjadi **book**. Alasan: Input numerik untuk embedding.
- Dibuat mapping ID asli ke terenkode dan sebaliknya. **Alasan:** Memudahkan inferensi.
- **collab_final_df** (**user**, **book**, **Rating**) dibuat.

**8. Pembagian Data Latih/Validasi:** collab_final_df dibagi 80% latih, 20% validasi. Alasan: Evaluasi generalisasi model.
- Jumlah data Training : `346936`
- Jumlah data Validasi : `86735`

**Hasil Data Setelah Persiapan:**

- content_df: 340,556 buku unik.
- collab_df: 433,671 interaksi.

## Modeling
Dua model sistem rekomendasi dikembangkan:

**1. Content-Based Filtering**
Merekomen­dasikan buku berdasarkan kemiripan konten.

**Proses Pembuatan Model:**

1. Nearest Neighbors Model: Menggunakan NearestNeighbors (n_neighbors=11, metric='cosine', algorithm='brute') untuk mencari 10 buku termirip berdasarkan matriks TF-IDF.

2. Pemetaan Judul ke Indeks: Dibuat pandas.Series untuk memetakan judul ke indeks.

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

- **Kelebihan:** Tidak butuh data pengguna lain, bisa rekomendasikan item baru, transparan.
- **Kekurangan:** Terbatas fitur item, bisa overspecialization, butuh domain knowledge, cold-start untuk pengguna baru.

**2. Collaborative Filtering**

Merekomen­dasikan buku berdasarkan kemiripan preferensi antar pengguna.

**Proses Pembuatan Model:**

**1. Parameter:** num_users (pengguna unik terenkode), num_books (buku unik terenkode), embedding_size = 50.

**2. Arsitektur Model (Neural Network - Keras):**

- Input: user dan book (ID terenkode).

- Embedding Layers: Untuk user dan book (dimensi 50, inisialisasi he_normal, regularisasi L2).

- Flatten & Concatenate: Menggabungkan vektor embedding.

- Dense Layers: Dua lapisan (128, 64 unit, aktivasi ReLU).

- Output Layer: Satu unit (aktivasi linear) untuk prediksi peringkat.

- Kompilasi: Optimizer Adam (lr 0.001), loss mean_squared_# Ringkasan arsitektur model (dari notebook)
  
**Ringkasan Arsitektur Model Collaborative Filtering**

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_collaborative_model(num_users, num_books, embedding_dim=50):
    user_input = Input(shape=(1,), name='user_input')
    book_input = Input(shape=(1,), name='book_input')

    user_embedding = Embedding(
        input_dim=num_users,
        output_dim=embedding_dim,
        name='user_embedding',
        embeddings_regularizer=l2(1e-6)
    )(user_input)
    book_embedding = Embedding(
        input_dim=num_books,
        output_dim=embedding_dim,
        name='book_embedding',
        embeddings_regularizer=l2(1e-6)
    )(book_input)

    user_vec = Flatten()(user_embedding)
    book_vec = Flatten()(book_embedding)

    dot_product = Dot(axes=1)([user_vec, book_vec])

    output = Dense(1, activation='linear')(dot_product)

    model = Model(inputs=[user_input, book_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

# Ringkasan Arsitektur
collaborative_model = build_collaborative_model(num_users, num_books)
collaborative_model.summary()
```

**3. Pelatihan Model:** Dilatih pada x_train dan y_train selama 10 epoch, batch size 64, dengan data validasi.

Fungsi Rekomendasi (get_collaborative_recommendations_ringkas):
Menerima ID pengguna, memprediksi peringkat untuk buku yang belum berinteraksi, dan mengembalikan k (default 10) buku dengan prediksi peringkat tertinggi.

**Hasil Rekomendasi (Top-N):**
Untuk pengguna ID '276726':

| No. | Judul Buku                                                                 | Predicted Rating |
| --- | -------------------------------------------------------------------------- | ---------------- |
| 1   | The Sneetches and Other Stories                                            | 9.764567         |
| 2   | Lonesome Dove                                                              | 9.445168         |
| 3   | Bury My Heart at Wounded Knee: An Indian History of the American West      | 9.327517         |
| 4   | Mere Christianity: A revised and enlarged edition, with a new introduction | 9.287292         |
| 5   | Dilbert: A Book of Postcards                                               | 9.263111         |
| 6   | ANNE FRANK: DIARY OF A YOUNG GIRL                                          | 9.235662         |
| 7   | My Sister's Keeper : A Novel (Picoult, Jodi)                               | 9.234827         |
| 8   | The Biggest Pumpkin Ever                                                   | 9.209048         |
| 9   | Love, Greg & Lauren                                                        | 9.191188         |
| 10  | It Came From The Far Side                                                  | 9.079331         |

Model merekomendasikan buku berdasarkan preferensi pengguna serupa.

- Kelebihan: Tidak butuh fitur item, bisa menemukan item tak terduga (serendipity), efektif dengan banyak data interaksi.
- Kekurangan: Cold-start problem (pengguna/item baru), data sparsity, popularity bias.

## Evaluation

Metrik evaluasi berbeda untuk kedua pendekatan.

**1. Content-Based Filtering**

Metrik Evaluasi: Precision (berdasarkan kesamaan penulis).

Cara Kerja Metrik:

![image](https://github.com/user-attachments/assets/e46fca87-0452-4794-9196-cd79d9013953)

**Hasil Proyek Berdasarkan Metrik:**

Untuk 'Death on the Nile': **Precision (Author):** 1.00. (Semua 10 rekomendasi dari Agatha Christie).

**2. Collaborative Filtering**
Metrik Evaluasi: Root Mean Squared Error (RMSE).

Cara Kerja Metrik (Formula):

![image](https://github.com/user-attachments/assets/4c4eae4c-610e-4351-9d85-66ceeb011fae)

**Dimana:**
- `N` adalah jumlah total item dalam dataset evaluasi.
- `y_i` adalah nilai peringkat aktual untuk item ke-i.
- `ŷ_i` adalah nilai peringkat yang diprediksi oleh model untuk item ke-i.

RMSE mengukur rata-rata besarnya kesalahan prediksi model. **Semakin rendah nilai RMSE, semakin baik kinerja model dalam memprediksi peringkat.**

**Hasil Evaluasi**

- **Dataset evaluasi:** `collab_final_df`  
  (berisi semua interaksi pengguna-buku yang telah diberi peringkat dan di-encode)
- **Skala peringkat:** 1–10 (peringkat 0 telah difilter sebelumnya)
  
**Hasil:**

- **RMSE Model Collaborative Filtering: `1.0130`**

Nilai RMSE sekitar **1.01** menunjukkan bahwa rata-rata kesalahan prediksi peringkat oleh model adalah sekitar **1.01 poin**. Ini merupakan hasil yang cukup baik dan menandakan bahwa model memiliki kemampuan prediksi yang **cukup akurat** terhadap data yang tersedia.
