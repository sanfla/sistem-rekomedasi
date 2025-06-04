import streamlit as st
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_url = "https://huggingface.co/datasets/sanfla/Book-Embedding/resolve/main/book_embeddings.npy"
csv_url = "https://huggingface.co/datasets/sanfla/Book-Embedding/resolve/main/BooksDataset_Index.csv"

with open("book_embeddings.npy", "wb") as f:
    f.write(requests.get(embedding_url).content)

with open("BooksDataset_Index.csv", "wb") as f:
    f.write(requests.get(csv_url).content)

df = pd.read_csv("BooksDataset_Index.csv")
embeddings = np.load("book_embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Sistem Rekomendasi Buku", layout="centered")
st.title("📚 Sistem Rekomendasi Buku")
st.write("Masukkan topik atau preferensi Anda, dan sistem akan merekomendasikan buku paling relevan berdasarkan deskripsi.")

query = st.text_input("🔍 Contoh: Harry Potter, Adventure Kids, Comedy")

if query:
    with st.spinner("🔍 Mencari rekomendasi terbaik..."):
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

        hasil = df.copy()
        hasil["Similarity"] = similarity_scores
        hasil = hasil.sort_values(by="Similarity", ascending=False).head(5)

    st.subheader("📖 Rekomendasi Buku:")
    for _, row in hasil.iterrows():
        st.markdown(f"### {row['Title']}")
        st.caption(f"_oleh {row['Authors']}_")
        st.markdown(f"**Kategori:** {row['Category']}")
        st.markdown(row['Description'])
        st.markdown("---")
