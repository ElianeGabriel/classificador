import streamlit as st
import importlib.util
import sys
import os
from pathlib import Path
import base64  


def set_background(image_filename: str, opacity: float = 0.10):
    """
    Define imagem de fundo via CSS com base64.
    opacity -> 0 a 1 (um véu branco para melhorar a legibilidade)
    """
    img_path = Path(__file__).parent / image_filename
    if not img_path.exists():
        st.warning(f"Imagem de fundo não encontrada: {img_path.name}")
        return

    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    # Vários seletores para diferentes versões do Streamlit
    css = f"""
    <style>
    /* Fundo do corpo da app */
    html, body, .stApp {{
        height: 100%;
        background: none !important;
    }}

    /* Conteúdo principal */
    [data-testid="stAppViewContainer"] > .main {{
        background-image:
          linear-gradient(rgba(255,255,255,{opacity}), rgba(255,255,255,{opacity})),
          url("data:image/png;base64,{b64}");
        background-position: center center;
        background-repeat: no-repeat;
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Cabeçalho transparente */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* Sidebar (opcional): comenta se não quiseres */
    [data-testid="stSidebar"] > div:first-child {{
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(2px);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# chama a função com o nome da tua imagem
set_background("ANI_Screen Call_02.png", opacity=0.10)

st.markdown(page_bg_css, unsafe_allow_html=True)
st.set_page_config(page_title="Classificador de Projetos", layout="wide")
st.title("🤖 Classificador Automático")

# Sidebar principal
modo_app = st.sidebar.radio(
    "Seleciona o modo:",
    ["🏠 Página Inicial", "🧠 Classificação com LLM", "👥 Alocação de Peritos", "📈 Métricas e Visualizações"]
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
    ## 👋 Bem-vindo ao Classificador de Projetos 
    Esta aplicação permite classificar automaticamente projetos de I&D nos domínios da Estratégia Nacional de Especialização Inteligente (ENEI 2020 e 2030), e alocar peritos a projetos.
    """)

# Classificação com LLM
def pagina_classificacao():
    st.subheader("🧠 Classificação Automática com LLM")
    modulo = carregar_modulo("Classifier.py", "classifier")
    if hasattr(modulo, "run"):
        modulo.run()

# Alocação de Peritos
def pagina_peritos():
    st.subheader("👥 Alocação de Peritos a Projetos")
    modulo = carregar_modulo("ExpertsAllocator.py", "experts_allocator")
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
elif modo_app == "👥 Alocação de Peritos":
    pagina_peritos()
elif modo_app == "📈 Métricas e Visualizações":
    pagina_metricas()
