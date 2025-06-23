import streamlit as st
import pandas as pd
import openai
import os
from io import BytesIO

# ------------------------------
# Inicializar cliente da OpenAI com a API Key
# ------------------------------
client = openai.OpenAI()  # a API Key já é lida automaticamente da variável OPENAI_API_KEY no Secrets do Streamlit

# ------------------------------
# Função: preparar prompt para o modelo
# ------------------------------
def preparar_prompt(titulo, resumo, dominios):
    prompt = f"""
Classifica o projeto abaixo num dos seguintes domínios prioritários da Estratégia Nacional de Especialização Inteligente (ENEI 2020):

{chr(10).join([f"- {d}" for d in dominios])}

Projeto:
Título: {titulo}
Descrição: {resumo}

Responde apenas com o nome exato do domínio mais adequado, sem explicações. Se não conseguires decidir com certeza, responde com "Indefinido".
""".strip()
    return prompt

# ------------------------------
# Carregar lista de domínios da ENEI 2020
# ------------------------------
def carregar_dominios_2020():
    df = pd.read_excel("descricao2020.xlsx", sheet_name=0)
    df.dropna(subset=["Dominios"], inplace=True)
    return df["Dominios"].unique().tolist()

# ------------------------------
# Função: Classificar projeto com OpenAI GPT
# ------------------------------
def classificar_llm(texto_prompt):
    try:
        chat_response = client.chat.completions.create(
            model="gpt-4o",  # ou "gpt-3.5-turbo" se preferires
            messages=[{"role": "user", "content": texto_prompt}],
            temperature=0
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {e}"

# ------------------------------
# Interface Streamlit
# ------------------------------
st.markdown("### 🤖 Classificação com LLM (GPT via OpenAI)")
uploaded_file = st.file_uploader("📁 Faz upload do ficheiro com os projetos reais:", type=["xlsx"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("📄 Escolhe a sheet com os projetos:", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)

        colunas = df.columns.tolist()
        col_titulo = st.selectbox("📝 Coluna do título do projeto:", colunas, index=colunas.index("Designacao Projecto") if "Designacao Projecto" in colunas else 0)
        col_resumo = st.selectbox("📋 Coluna da descrição:", colunas, index=colunas.index("Sumario Executivo") if "Sumario Executivo" in colunas else 0)

        limite_opcao = st.radio("Quantos projetos queres classificar?", ["Todos", "10", "20", "50", "100"])
        if limite_opcao != "Todos":
            limite = int(limite_opcao)
            df = df.head(limite)

        dominios_enei = carregar_dominios_2020()

        if st.button("🚀 Classificar com LLM"):
            resultados = []
            with st.spinner("A classificar projetos com o LLM da OpenAI..."):
                for _, row in df.iterrows():
                    titulo = str(row.get(col_titulo, ""))
                    resumo = str(row.get(col_resumo, ""))
                    prompt = preparar_prompt(titulo, resumo, dominios_enei)
                    classificacao = classificar_llm(prompt)

                    resultados.append({
                        "NIPC": row.get("NIPC", ""),
                        "Projeto": titulo,
                        "Resumo": resumo,
                        "Domínio LLM": classificacao
                    })

            final_df = pd.DataFrame(resultados)
            st.success("✅ Classificação concluída com sucesso!")
            st.dataframe(final_df)

            buffer = BytesIO()
            final_df.to_excel(buffer, index=False)
            st.download_button(
                label="📥 Download dos resultados (.xlsx)",
                data=buffer.getvalue(),
                file_name="classificacao_llm_enei2020.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Erro ao processar: {e}")
