import streamlit as st

# Configurações iniciais da app
st.set_page_config(page_title="Classificador de Projetos SIFIDE", layout="wide")
st.title("📊 Classificador de Projetos para Domínios ENEI")

# Sidebar principal
modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🔍 Classificação", "📈 Métricas e Visualizações"]
)

if modo_app == "🔍 Classificação":
    tipo_classificador = st.sidebar.selectbox(
        "Escolhe o tipo de classificador:",
        ["Classificação por Palavras-chave", "Classificação com LLM"]
    )

    if tipo_classificador == "Classificação por Palavras-chave":
        st.subheader("🔎 Modo por Palavras-chave")
        with open("Classifier.py", "r", encoding="utf-8") as f:
            exec(f.read())

    elif tipo_classificador == "Classificação com LLM":
        st.subheader("🤖 Modo com Modelo de Linguagem (LLM)")
        with open("appclas.py", "r", encoding="utf-8") as f:
            exec(f.read())

elif modo_app == "📈 Métricas e Visualizações":
    st.subheader("📈 Avaliação das Classificações (LLM vs Manual)")
    with open("metrics.py", "r", encoding="utf-8") as f:
        exec(f.read())
