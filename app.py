import streamlit as st
import importlib.util
import sys
import os

st.set_page_config(page_title="Classificador de Projetos ENEI", layout="wide")
st.title("🤖 Classificador Inteligente de Projetos ENEI")

# Sidebar principal
modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🏠 Página Inicial", "🧠 Classificação com LLM", "📈 Métricas e Visualizações"]
)

# Verificação rápida do ambiente (opcional)
with st.sidebar.expander("⚙️ Ambiente"):
    def ok(x): return "✅" if os.getenv(x) else "⚠️"
    st.write(f"{ok('AZURE_OPENAI_KEY')} AZURE_OPENAI_KEY")
    st.write(f"{ok('AZURE_OPENAI_ENDPOINT')} AZURE_OPENAI_ENDPOINT")
    st.write(f"{ok('AZURE_OPENAI_API_VERSION')} AZURE_OPENAI_API_VERSION")
    st.write(f"{ok('AZURE_OPENAI_DEPLOYMENT')} AZURE_OPENAI_DEPLOYMENT (chat)")
    st.write(f"{ok('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT')} AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (embeddings)")

# Função para carregar e executar um módulo externo (com run())
def carregar_modulo(nome_ficheiro, nome_modulo):
    spec = importlib.util.spec_from_file_location(nome_modulo, nome_ficheiro)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo

# Página Inicial
def pagina_inicial():
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
def pagina_classificacao():
    st.subheader("🧠 Classificação Automática com Modelo de Linguagem (LLM)")
    modulo = carregar_modulo("Classifier.py", "classifier")
    if hasattr(modulo, "run"):
        modulo.run()

# Métricas e Visualizações
def pagina_metricas():
    st.subheader("📈 Avaliação das Classificações (LLM vs Manual)")
    modulo = carregar_modulo("metrics.py", "metrics")
    if hasattr(modulo, "run"):
        modulo.run()

# Router simples
if modo_app == "🏠 Página Inicial":
    pagina_inicial()
elif modo_app == "🧠 Classificação com LLM":
    pagina_classificacao()
elif modo_app == "📈 Métricas e Visualizações":
    pagina_metricas()
