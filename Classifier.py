import streamlit as st
import pandas as pd
import openai
import os
from io import BytesIO

# ------------------------------
# API KEY (por variável de ambiente segura)
# ------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------
# Preparar o prompt para o LLM
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
# Carregar os domínios da ENEI 2020
# ------------------------------
def carregar_dominios_2020():
    df = pd.read_excel("descricao2020.xlsx", sheet_name=0)
    df.dropna(subset=['Dominios'], inplace=True)
    return df['Dominios'].unique().tolist()

# ------------------------------
# Função para classificar com OpenAI LLM
# ------------------------------
def classificar_llm(prompt_texto):
    try:
        resposta = openai.chat.completions.create(
            model="gpt-4o",  # Ou "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt_texto}],
            temperature=0
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {e}"

# ------------------------------
# INTERFACE
# ------------------------------
st.markdown("### 🧠 Classificação com LLM (OpenAI API)")
uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos reais (.xlsx):", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("📄 Escolhe a folha (sheet):", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)

    colunas = df.columns.tolist()
    col_titulo = st.selectbox("📝 Coluna do título:", colunas, index=colunas.index("Designacao Projecto") if "Designacao Projecto" in colunas else 0)
    col_resumo = st.selectbox("📋 Coluna da descrição/resumo:", colunas, index=colunas.index("Sumario Executivo") if "Sumario Executivo" in colunas else 0)

    dominios_enei = carregar_dominios_2020()

    st.markdown("### ⚙️ Quantos projetos queres classificar?")
    opcao_modo = st.radio("Modo:", ["Teste (1 projeto)", "5", "10", "20", "50", "Todos"])
    
    if opcao_modo == "Teste (1 projeto)":
        df = df.head(1)
    elif opcao_modo != "Todos":
        df = df.head(int(opcao_modo))

    # Estimativa de tokens
    n_proj = len(df)
    tokens_por_proj = 610  # aproximado
    total_tokens = n_proj * tokens_por_proj
    st.info(f"🧮 Estimativa: {total_tokens} tokens (aprox.) para {n_proj} projetos")

    if st.button("🚀 Classificar com LLM"):
        resultados = []
        with st.spinner("A classificar projetos..."):
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
            label="📥 Download (.xlsx)",
            data=buffer.getvalue(),
            file_name="classificacao_llm_enei2020.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
