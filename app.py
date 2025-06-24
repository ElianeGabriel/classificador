import streamlit as st
import importlib.util
import sys

# Configurações iniciais da app
st.set_page_config(page_title="Classificador de Projetos SIFIDE", layout="wide")
st.title("📊 Classificador de Projetos para Domínios ENEI")

# Sidebar principal
modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🔍 Classificação", "📈 Métricas e Visualizações"]
)

# Função para carregar e executar um módulo externo
def carregar_modulo(nome_ficheiro, nome_modulo):
    spec = importlib.util.spec_from_file_location(nome_modulo, nome_ficheiro)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo

# Classificação
if modo_app == "🔍 Classificação":
    tipo_classificador = st.sidebar.selectbox(
        "Escolhe o tipo de classificador:",
        ["Classificação por Palavras-chave", "Classificação com LLM"]
    )

    if tipo_classificador == "Classificação por Palavras-chave":
        st.subheader("🔎 Modo por Palavras-chave")
        carregar_modulo("Classifier.py", "classifier")

    elif tipo_classificador == "Classificação com LLM":
        st.subheader("🤖 Modo com Modelo de Linguagem (LLM)")
        carregar_modulo("appclas.py", "appclas")

# Avaliação
elif modo_app == "📈 Métricas e Visualizações":
    st.subheader("📈 Avaliação das Classificações (LLM vs Manual)")
    carregar_modulo("metrics.py", "metrics")
