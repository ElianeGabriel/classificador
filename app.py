import streamlit as st
import importlib.util
import sys
import os
from pathlib import Path
import base64

# -----------------------------------------
# Configuração geral
# -----------------------------------------
st.set_page_config(page_title="Classificador de Projetos ENEI / EREI", layout="wide")
st.title("🤖 Classificador Automático de Projetos — ENEI / EREI")

# -----------------------------------------
# Função para fundo (opcional)
# -----------------------------------------
def set_background(image_filename: str, opacity: float = 0.10):
    img_path = Path(__file__).parent / image_filename
    if not img_path.exists():
        return
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image:
          linear-gradient(rgba(255,255,255,{opacity}), rgba(255,255,255,{opacity})),
          url("data:image/png;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Aplica fundo (se quiseres)
# set_background("ANI_Screen Call_02.png", opacity=0.10)

# -----------------------------------------
# Menu lateral
# -----------------------------------------
modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    [
        "🏠 Página Inicial",
        "🧠 Classificação com LLM (ENEI / EREI)",
        "👥 Alocação de Peritos",
        "📈 Métricas e Visualizações"
    ]
)

# Diagnóstico do ambiente
with st.sidebar.expander("⚙️ Ambiente"):
    def ok(x): return "✅" if os.getenv(x) else "⚠️"
    st.write(f"{ok('AZURE_OPENAI_KEY')} AZURE_OPENAI_KEY")
    st.write(f"{ok('AZURE_OPENAI_ENDPOINT')} AZURE_OPENAI_ENDPOINT")
    st.write(f"{ok('AZURE_OPENAI_API_VERSION')} AZURE_OPENAI_API_VERSION")
    st.write(f"{ok('AZURE_OPENAI_DEPLOYMENT')} AZURE_OPENAI_DEPLOYMENT (chat)")
    st.write(f"{ok('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT')} AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (embeddings)")

# -----------------------------------------
# Função utilitária para carregar módulos
# -----------------------------------------
def carregar_modulo(nome_ficheiro, nome_modulo):
    spec = importlib.util.spec_from_file_location(nome_modulo, nome_ficheiro)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo

# -----------------------------------------
# Páginas
# -----------------------------------------
def pagina_inicial():
    st.markdown("""
    ## 👋 Bem-vindo ao Classificador de Projetos  
    Esta aplicação permite classificar automaticamente projetos de I&D nos domínios da **ENEI 2020 / ENEI 2030** ou das **EREI (Estratégias Regionais de Especialização Inteligente)**,  
    e também realizar **alocação automática de peritos** a projetos.
    """)

def pagina_classificacao():
    st.subheader("🧠 Classificação Automática com Modelo de Linguagem (LLM) — ENEI / EREI")
    modulo = carregar_modulo("Classifier.py", "classifier")
    if hasattr(modulo, "run"):
        modulo.run()

def pagina_peritos():
    st.subheader("👥 Alocação de Peritos a Projetos")
    modulo = carregar_modulo("ExpertsAllocator.py", "experts_allocator")
    if hasattr(modulo, "run"):
        modulo.run()

def pagina_metricas():
    st.subheader("📈 Avaliação das Classificações (LLM vs Manual)")
    modulo = carregar_modulo("metrics.py", "metrics")
    if hasattr(modulo, "run"):
        modulo.run()

# -----------------------------------------
# Router
# -----------------------------------------
if modo_app == "🏠 Página Inicial":
    pagina_inicial()
elif modo_app == "🧠 Classificação com LLM (ENEI / EREI)":
    pagina_classificacao()
elif modo_app == "👥 Alocação de Peritos":
    pagina_peritos()
elif modo_app == "📈 Métricas e Visualizações":
    pagina_metricas()
