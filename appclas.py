import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from io import BytesIO

# ------------------------------
# Inicializar modelo com cache
# ------------------------------
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

modelo = carregar_modelo()

# ------------------------------
# Interface de seleção do ficheiro de domínios
# ------------------------------
opcao_enei = st.sidebar.radio(
    "Seleciona a versão da ENEI para classificar:",
    ["ENEI 2030", "ENEI 2020"]
)

# Mapeamento das opções para ficheiros e sheets
config_enei = {
    "ENEI 2030": {"ficheiro": "descricao2030.xlsx", "sheet": "Dominios"},
    "ENEI 2020": {"ficheiro": "descricao2020.xlsx", "sheet": "Eixos"}
}

# ------------------------------
# Função para ENEI 2030 (mantida como está)
# ------------------------------
@st.cache_data
def carregar_dominios_2030(ficheiro, sheet):
    dominios_df = pd.read_excel(ficheiro, sheet_name=sheet)
    dominios_desc = {}
    for _, row in dominios_df.iterrows():
        nome = str(row['Dominios']).strip()
        area = str(row.get('Principal área de atuação (Opções de Resposta)', ''))
        desc = str(row.get('Descrição', ''))
        texto_completo = f"{nome}. {area}. {desc}"
        dominios_desc[nome] = texto_completo
    return dominios_desc

# ------------------------------
# Nova função para ENEI 2020
# ------------------------------
@st.cache_data
def carregar_dominios_2020(ficheiro, sheet):
    dominios_df = pd.read_excel(ficheiro, sheet_name=sheet)
    dominios_df.dropna(how="all", inplace=True)

    # Normaliza os nomes das colunas
    colunas_originais = dominios_df.columns.tolist()
    colunas_normalizadas = [c.strip().lower() for c in colunas_originais]
    col_map = dict(zip(colunas_normalizadas, colunas_originais))

    nome_col = col_map.get("dominios")
    desc_col = col_map.get("descrição")
    area_col = col_map.get("principal área de atuação (opções de resposta)")

    if not nome_col or not desc_col:
        raise ValueError(
            f"❌ Colunas obrigatórias não encontradas.\n"
            f"Esperadas: 'Dominios' e 'Descrição'.\n"
            f"Colunas disponíveis: {colunas_originais}"
        )

    dominios_desc = {}
    for _, row in dominios_df.iterrows():
        nome = str(row.get(nome_col, '')).strip()
        if not nome:
            continue

        desc = str(row.get(desc_col, '')).strip()
        area = str(row.get(area_col, '')).strip() if area_col else ''
        texto_completo = f"{nome}. {desc}. {area}".strip()
        dominios_desc[nome] = texto_completo

    return dominios_desc

# ------------------------------
# Carregar domínios conforme versão selecionada
# ------------------------------
if opcao_enei == "ENEI 2030":
    dominios_desc = carregar_dominios_2030(
        ficheiro=config_enei["ENEI 2030"]["ficheiro"],
        sheet=config_enei["ENEI 2030"]["sheet"]
    )
else:
    dominios_desc = carregar_dominios_2020(
        ficheiro=config_enei["ENEI 2020"]["ficheiro"],
        sheet=config_enei["ENEI 2020"]["sheet"]
    )

dominios_lista = list(dominios_desc.keys())
dominios_embs = modelo.encode(list(dominios_desc.values()), convert_to_tensor=True)

# ------------------------------
# Função para classificar projetos
# ------------------------------
def classificar_projeto(texto):
    texto_emb = modelo.encode(texto, convert_to_tensor=True)
    similaridades = util.cos_sim(texto_emb, dominios_embs)[0]
    pontuacoes = [(dominios_lista[i], float(similaridades[i])) for i in range(len(dominios_lista))]
    pontuacoes.sort(key=lambda x: x[1], reverse=True)
    top_k = [p for p in pontuacoes if p[1] > 0.3][:3]
    soma = sum(p[1] for p in top_k)
    if soma == 0:
        return []
    return [(p[0], round(100 * p[1]/soma, 2)) for p in top_k]

# ------------------------------
# Interface principal
# ------------------------------
st.markdown(f"### Classificação baseada na versão **{opcao_enei}**")
uploaded_file = st.file_uploader("Faz upload do ficheiro Excel com a sheet 'Projetos':", type=["xlsx"])

# ------------------------------
# Processar ficheiro carregado com seleção de sheet e colunas
# ------------------------------
if uploaded_file:
    try:
        # Carrega todas as sheets do Excel
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("📄 Escolhe a sheet do ficheiro", xls.sheet_names)
        projetos_df = pd.read_excel(xls, sheet_name=sheet_name)

        st.markdown("### 🧩 Seleciona as colunas a utilizar para classificação")

        colunas_disponiveis = projetos_df.columns.tolist()
        col_titulo = st.selectbox("📝 Coluna com o título do projeto", colunas_disponiveis)
        col_resumo = st.selectbox("📋 Coluna com o resumo/descrição", colunas_disponiveis)

        # Quantos projetos classificar
        limite_opcao = st.radio(
            "Quantos projetos queres classificar?",
            ["Todos", "10", "20", "50", "100"]
        )

        if limite_opcao != "Todos":
            limite = int(limite_opcao)
            projetos_df = projetos_df.head(limite)

        if st.button("🚀 Classificar projetos"):
            resultados = []
            for _, row in projetos_df.iterrows():
                titulo = str(row.get(col_titulo, ""))
                resumo = str(row.get(col_resumo, ""))
                texto = f"{titulo}. {resumo}"
                dominios_previstos = classificar_projeto(texto)

                linha = {
                    "Projeto": titulo,
                    "Resumo": resumo
                }
                for i, (dom, score) in enumerate(dominios_previstos):
                    linha[f"Domínio {i+1}"] = dom
                    linha[f"% {i+1}"] = score

                resultados.append(linha)

            final_df = pd.DataFrame(resultados)
            st.success("Classificação concluída!")
            st.dataframe(final_df)

            # Exportar para Excel
            buffer = BytesIO()
            final_df.to_excel(buffer, index=False)
            st.download_button(
                label="📄 Download dos resultados (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"classificacao_{opcao_enei.replace(' ', '').lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Erro ao processar o ficheiro: {e}")
