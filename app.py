import streamlit as st
import importlib.util
import sys
import os

st.set_page_config(page_title="Classificador de Projetos ENEI", layout="wide")
st.title("🤖 Classificador Inteligente de Projetos ENEI")

# --------- Verificação rápida do ambiente (visível na sidebar)
with st.sidebar.expander("⚙️ Ambiente"):
    def ok(val): return "✅" if val else "⚠️"
    st.write(f"{ok(os.getenv('AZURE_OPENAI_KEY'))} AZURE_OPENAI_KEY")
    st.write(f"{ok(os.getenv('AZURE_OPENAI_ENDPOINT'))} AZURE_OPENAI_ENDPOINT")
    st.write(f"{ok(os.getenv('AZURE_OPENAI_API_VERSION'))} AZURE_OPENAI_API_VERSION")
    st.write(f"{ok(os.getenv('AZURE_OPENAI_DEPLOYMENT'))} AZURE_OPENAI_DEPLOYMENT (chat)")
    st.write(f"{ok(os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT'))} AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (embeddings)")

modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🏠 Página Inicial", "🧠 Classificação com LLM", "📈 Métricas e Visualizações"],
    index=1
)

def carregar_modulo(nome_ficheiro, nome_modulo):
    try:
        spec = importlib.util.spec_from_file_location(nome_modulo, nome_ficheiro)
        modulo = importlib.util.module_from_spec(spec)
        sys.modules[nome_modulo] = modulo
        spec.loader.exec_module(modulo)
        return modulo
    except FileNotFoundError:
        st.error(f"Ficheiro não encontrado: `{nome_ficheiro}`.")
    except Exception as e:
        st.exception(e)
    return None

def pagina_inicial():
    st.markdown("""
    ## 👋 Bem-vindo ao Classificador de Projetos ENEI
    Esta aplicação classifica projetos de I&D nos domínios da ENEI **2020** ou **2030**, usando LLM.
    
    ### O que podes fazer
    - **🧠 Classificar** projetos (LLM, com opção de percentagens por similaridade)
    - **📈 Avaliar** a qualidade (comparação com classificação manual, métricas e gráficos)
    
    Usa o menu lateral para escolher a secção.
    """)

def pagina_classificacao():
    modulo = carregar_modulo("Classifier.py", "classifier")
    if modulo and hasattr(modulo, "run"):
        modulo.run()  # se mais tarde quiseres encapsular no próprio ficheiro
    # Nota: o teu Classifier.py atual executa no import, por isso mesmo sem run() funciona.

def pagina_metricas():
    modulo = carregar_modulo("metrics.py", "metrics")
    if modulo and hasattr(modulo, "run"):
        modulo.run()  # idem

# app.py (apenas a parte do menu + handlers)
def pagina_alocacao():
    modulo = carregar_modulo("ExpertsAllocator.py", "experts_allocator")
    if modulo and hasattr(modulo, "run"):
        modulo.run()

modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🏠 Página Inicial", "🧠 Classificação com LLM", "👥 Alocação de Peritos", "📈 Métricas e Visualizações"],
    index=1
)


if modo_app == "🏠 Página Inicial":
    pagina_inicial()
elif modo_app == "🧠 Classificação com LLM":
    pagina_classificacao()
elif modo_app == "👥 Alocação de Peritos":
    pagina_alocacao()
elif modo_app == "📈 Métricas e Visualizações":
    pagina_metricas()
