import streamlit as st
import importlib.util
import sys

# Configurações iniciais da app
st.set_page_config(page_title="Classificador de Projetos ENEI", layout="wide")
st.title("🤖 Classificador Inteligente de Projetos ENEI")

# Sidebar principal
modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🏠 Página Inicial", "🧠 Classificação com LLM", "📈 Métricas e Visualizações"]
)

# Função para carregar e executar um módulo externo
def carregar_modulo(nome_ficheiro, nome_modulo):
    spec = importlib.util.spec_from_file_location(nome_modulo, nome_ficheiro)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo

# Página Inicial
if modo_app == "🏠 Página Inicial":
    st.markdown("""
    ## 👋 Bem-vindo ao Classificador de Projetos ENEI
    Esta aplicação permite classificar automaticamente projetos de I&D nos domínios da Estratégia Nacional de Especialização Inteligente (ENEI 2020 e 2030), usando um modelo de linguagem avançado (LLM).

    ### Funcionalidades:
    - Classificação automática com LLM (GPT)
    - Comparação com classificações manuais
    - Visualização de métricas de desempenho

    **Escolhe uma opção no menu lateral para começar.**
    """)

# Classificação com LLM
elif modo_app == "🧠 Classificação com LLM":
    st.subheader("🧠 Classificação Automática com Modelo de Linguagem (LLM)")
    carregar_modulo("Classifier.py", "classifier")

# Avaliação
elif modo_app == "📈 Métricas e Visualizações":
    st.subheader("📈 Avaliação das Classificações (LLM vs Manual)")
    carregar_modulo("metrics.py", "metrics")
