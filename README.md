# Nutriplan

## Deskripsi

**Nutrplan** adalah aplikasi AI yang berfungsi sebagai asisten nutrisi pribadi. Aplikasi ini mengintegrasikan tiga komponen utama:

- **Chatbot Cerdas** berbasis dokumen WHO untuk menjawab pertanyaan seputar kesehatan.
- **Model Klasifikasi Nutrisi** untuk mengidentifikasi kandungan gizi dari gambar makanan.
- **Generator Jadwal Makan** untuk menyusun rencana diet mingguan yang dipersonalisasi.

---

## Fitur Utama

### 💬 Chatbot Cerdas (RAG)
- Menggunakan arsitektur **Retrieval-Augmented Generation (RAG)**.
- Sumber informasi berasal dari dokumen resmi **WHO** berformat PDF.
- Didukung oleh model bahasa **Mistral** yang telah di-*fine-tune* menggunakan **LoRA (Low-Rank Adaptation)**.
- Mampu menjawab pertanyaan kompleks tentang nutrisi dan kesehatan berdasarkan konteks dokumen.

### 🥗 Klasifikasi Nutrisi Makanan
- Model *Deep Learning* berbasis **TensorFlow** dan **Keras**.
- Menggunakan arsitektur **Random Forest** untuk klasifikasi gambar makanan.
- Mengestimasi informasi nutrisi dari citra makanan secara efisien.

### 📅 Generator Jadwal Menu Makanan
- Sistem cerdas untuk menyusun menu makan selama 7 hari.
- Menyediakan menu sarapan, makan siang, dan makan malam.
- Menggunakan informasi nutrisi dan preferensi pengguna.

---

## Arsitektur & Teknologi

| Komponen              | Teknologi/Library       | Deskripsi                                                                 |
|-----------------------|-------------------------|--------------------------------------------------------------------------|
| **Chatbot Engine**    | `Transformers`, `Langchain`, `PyTorch` | Pipeline RAG dari pemuatan dokumen hingga inferensi.                   |
| **Model Bahasa (LLM)**| `Mistral`, `PEFT (LoRA)`| Model teks untuk pemahaman dan generasi respons.                         |
| **Klasifikasi Nutrisi**| `TensorFlow`, `Keras` | Framework Deep Learning untuk klasifikasi citra.                         |
| **Ekstraksi Dokumen** | `PyPDF`               | Mengekstrak teks dari file PDF (dokumen WHO).                            |
| **Bahasa Pemrograman**| `Python 3.x`            | Bahasa utama pengembangan aplikasi.                                      |
| **Deployment**        | `FastAPI`               | Digunakan untuk membangun REST API aplikasi.                             |

---

## Struktur Proyek

```
├── Chatbot/
│   ├── app.py                   # Deploy model API
│   ├── agent.py                 # Pipeline RAG & logika chatbot
│   ├── requirements.txt         # Dependensi Python
│   ├── WHO_doc/                 # Folder dokumen PDF WHO
│   ├── mistral-lora-adapter/    # Adapter LoRA untuk model Mistral yang sudah di-finetune
│   ├── test.py                  # Finetune model Mistral
│   ├── mistral_data.jsonl       # Data untuk finetune model Mistral
│   └── faiss_who_index/         # Index FAISS untuk dokumen WHO
│
├── Klasifikasi_Nutrisi/
│   └── Klasifikasi_Nutrisi.ipynb  # Notebook model klasifikasi makanan
│
├── Penjadwalan/
│   ├── generator.py            # Logika jadwal menu makan mingguan
│   ├── main.py                 # Skrip untuk menjalankan generator
│   └── model.py                # Model penjadwalan menu makan
│
└── README.md
```

---

## Instalasi

1. **Clone Repository**
   ```bash
   git clone [URL_REPOSITORY_ANDA]
   cd [NAMA_FOLDER_REPOSITORY]
   ```

2. **Buat Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```

3. **Instal Dependensi**
   ```bash
   pip install -r Chatbot/requirements.txt
   pip install -r Penjadwalan/requirements.txt
   pip install -r Klasifikasi_Nutrisi/requirements.txt
   ```

---

## Kontribusi

### Fitur apa yang ditawarkan NutriPlan?
NutriPlan menyediakan perencanaan menu, penjadwalan latihan, dan program pribadi yang didukung oleh kecerdasan buatan (AI) yang disesuaikan dengan tujuan Anda.

---
### Bisakah saya menyesuaikan rencana diet saya?
Ya, Anda dapat menyesuaikan rencana Anda atau memilih dari berbagai program yang paling sesuai dengan kebutuhan Anda.

---

## Kontak

@rafinas2133 as Front-End Developer

@daniadrian as Back-End Developer

@FRfans as Machiner Learning Engineer

@Firmnm as Machiner Learning Engineer

@luckedenn as Machiner Learning Engineer