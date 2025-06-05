import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Inicializar modelo
@st.cache_resource

def carregar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

modelo = carregar_modelo()

st.title("Classificador de Projetos SIFIDE para Domínios ENEI")
st.markdown("Faça upload de um ficheiro Excel com as sheets 'Projetos' e 'Dominios'.")

uploaded_file = st.file_uploader("Ficheiro Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        projetos_df = xls.parse("Projetos")
        dominios_df = xls.parse("Dominios")

        dominios_desc = {}
        for _, row in dominios_df.iterrows():
            nome = str(row['Dominios']).strip()
            area = str(row.get('Principal área de atuação (Opções de Resposta)', ''))
            desc = str(row.get('Descrição', ''))
            texto_completo = f"{nome}. {area}. {desc}"
            dominios_desc[nome] = texto_completo

        dominios_lista = list(dominios_desc.keys())
        dominios_embs = modelo.encode(list(dominios_desc.values()), convert_to_tensor=True)

        def classificar_projeto(texto):
            texto_emb = modelo.encode(texto, convert_to_tensor=True)
            similaridades = util.cos_sim(texto_emb, dominios_embs)[0]
            pontuacoes = [(dominios_lista[i], float(similaridades[i])) for i in range(len(dominios_lista))]
            pontuacoes.sort(key=lambda x: x[1], reverse=True)
            top_k = [p for p in pontuacoes if p[1] > 0.4][:3]
            soma = sum(p[1] for p in top_k)
            if soma == 0:
                return []
            return [(p[0], round(100 * p[1]/soma, 2)) for p in top_k]

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
        from io import BytesIO
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
