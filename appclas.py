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
# Carregar ficheiro de domínios
# ------------------------------
@st.cache_data
def carregar_dominios():
    dominios_df = pd.read_excel("descricao2030.xlsx", sheet_name="Dominios")
    dominios_desc = {}
    for _, row in dominios_df.iterrows():
        nome = str(row['Dominios']).strip()
        area = str(row.get('Principal área de atuação (Opções de Resposta)', ''))
        desc = str(row.get('Descrição', ''))
        texto_completo = f"{nome}. {area}. {desc}"
        dominios_desc[nome] = texto_completo
    return dominios_desc

dominios_desc = carregar_dominios()
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
# Interface do utilizador
# ------------------------------
st.title("Classificador de Projetos SIFIDE para Domínios ENEI")
st.markdown("Faça upload de um ficheiro Excel com a sheet 'Projetos'.")

uploaded_file = st.file_uploader("Ficheiro Excel (.xlsx)", type=["xlsx"])

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

        # Exporta para download
        buffer = BytesIO()
        final_df.to_excel(buffer, index=False)
        st.download_button(
            label="📄 Download dos resultados (.xlsx)",
            data=buffer.getvalue(),
            file_name="classificacao_dominios_llm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Erro ao processar o ficheiro: {e}")
