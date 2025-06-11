import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms.base import LLM
from pydantic import PrivateAttr

def extract_text_from_pdfs(folder_path):
    text = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(folder_path, file))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                print(f"Gagal memproses {file}: {e}")
    return text

def build_faiss_index(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not text.strip():
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_texts(chunks, embeddings)

def load_faiss_retriever(path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not os.path.exists(path):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return index.as_retriever(search_type="similarity", search_kwargs={"k": 3})

class CustomLLM(LLM):
    _model: any = PrivateAttr()
    _tokenizer: any = PrivateAttr()
    _device: str = PrivateAttr(default="cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, tokenizer, device=None, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _llm_type(self):
        return "custom-mistral-lora"

    def _call(self, prompt: str, stop=None) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.9,       # tingkatkan supaya output lebih variatif
            top_p=0.95,            # sampling yang lebih luas
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id
        )
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Hapus prompt dari hasil output agar hanya jawaban yang keluar
        return response[len(prompt):].strip()

def combined_retriever(query, retrievers):
    docs = []
    for retriever in retrievers:
        if retriever:
            docs += retriever.get_relevant_documents(query)
    return docs

def classify_food_or_drink(query: str) -> str:
    """Klasifikasi sederhana untuk membedakan makanan atau minuman."""
    minuman_keywords = ["susu", "teh", "kopi", "air", "jus", "sirup", "soda", "minuman", "milk", "drink"]
    makanan_keywords = ["nasi", "roti", "daging", "telur", "sayur", "makanan", "makan", "ayam", "ikan", "mie", "rice", "food", "bread"]

    query_lower = query.lower()
    if any(word in query_lower for word in minuman_keywords):
        return "minuman"
    elif any(word in query_lower for word in makanan_keywords):
        return "makanan"
    else:
        return "tidak diketahui"

def answer_query(query, llm, retrievers):
    jenis = classify_food_or_drink(query)

    docs = combined_retriever(query, retrievers)

    if docs:
        context = "\n\n---\n\n".join(
            [f"Sumber: {d.metadata.get('source', 'unknown')}\nKonten: {d.page_content}" for d in docs]
        )
        prompt = f"""
Anda adalah asisten nutrisi yang sangat membantu dan informatif.

Jenis bahan dalam pertanyaan ini adalah: {jenis}.

Jawab pertanyaan berdasarkan konteks berikut:

{context}

Pertanyaan: {query}

Jawaban:
"""
    else:
        prompt = f"""
Anda adalah asisten nutrisi yang sangat membantu dan informatif.

Jenis bahan dalam pertanyaan ini adalah: {jenis}.

Jawab pertanyaan berikut berdasarkan pengetahuan umum Anda.

Pertanyaan: {query}

Jawaban:
"""

    print("Prompt yang dikirim ke model:\n", prompt)  # Debug
    answer = llm._call(prompt)
    return answer, docs

# ===== Konfigurasi Path dan Model =====
pdf_folder = "WHO_doc"  # Folder berisi PDF WHO/FAO
index_path = "faiss_who_index"
adapter_path = "./mistral-lora-adapter"  # Path ke adapter LoRA
base_model_name = "mistralai/Mistral-7B-v0.1"

# ===== Cek dan Bangun FAISS Index jika belum ada =====
if not os.path.exists(index_path):
    print("Membangun indeks FAISS dari dokumen PDF...")
    raw_text = extract_text_from_pdfs(pdf_folder)
    index = build_faiss_index(raw_text)
    if index:
        index.save_local(index_path)
        print("Indeks FAISS berhasil dibuat dan disimpan.")
    else:
        print("Gagal membangun indeks FAISS.")
else:
    print("Menggunakan indeks FAISS lokal yang sudah ada.")

# ===== Load Tokenizer dan Model =====
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16
)
model.eval()

llm = CustomLLM(model, tokenizer)

# ===== Load FAISS Retriever =====
retriever = load_faiss_retriever(index_path)
retrievers = [retriever]