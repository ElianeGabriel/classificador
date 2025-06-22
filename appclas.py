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
# Carregar ficheiro de domínios selecionado
# ------------------------------
@st.cache_data
def carregar_dominios(ficheiro, sheet):
    dominios_df = pd.read_excel(ficheiro, sheet_name=sheet)

    # Limpar linhas totalmente em branco
    dominios_df.dropna(how="all", inplace=True)

    # Identificar colunas esperadas (em qualquer ordem)
    colunas = dominios_df.columns.str.strip().str.lower()
    col_map = {c.lower(): c for c in dominios_df.columns}

    nome_col = col_map.get("dominios", None)
    desc_col = col_map.get("descrição", None)
    area_col = col_map.get("principal área de atuação (opções de resposta)", None)

    if not nome_col or not desc_col:
        raise ValueError("Colunas obrigatórias ('Dominios' e 'Descrição') não encontradas no ficheiro.")

    dominios_desc = {}
    for _, row in dominios_df.iterrows():
        nome = str(row.get(nome_col, '')).strip()
        if not nome:
            continue  # Ignora linhas sem nome de domínio

        desc = str(row.get(desc_col, '')).strip()
        area = str(row.get(area_col, '')).strip() if area_col else ''

        # Combina tudo numa única string
        texto_completo = f"{nome}. {desc}. {area}"
        dominios_desc[nome] = texto_completo

    return dominios_desc

dominios_desc = carregar_dominios(
    ficheiro=config_enei[opcao_enei]["ficheiro"],
    sheet=config_enei[opcao_enei]["sheet"]
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
# Processar ficheiro carregado
# ------------------------------
if uploaded_file:
    try:
        projetos_df = pd.read_excel(uploaded_file, sheet_name="Projetos")

        resultados = []
        for _, row in projetos_df.iterrows():
            titulo = str(row.get("Designacao Projecto", ""))
            resumo = str(row.get("Sumario Executivo", ""))
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
        st.error(f"Erro ao processar o ficheiro: {e}")
