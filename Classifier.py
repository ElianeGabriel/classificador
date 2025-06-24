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
Classifica o projeto abaixo num dos seguintes domínios prioritários da Estratégia Nacional de Especialização Inteligente ({st.session_state.get('versao_enei', 'ENEI')}):

{chr(10).join([f"- {d}" for d in dominios])}

Projeto:
Título: {titulo}
Descrição: {resumo}

Responde apenas com o nome exato do domínio mais adequado, sem explicações. Se não conseguires decidir com certeza, responde com \"Indefinido\".
""".strip()
    return prompt

# ------------------------------
# Carregar domínios da ENEI
# ------------------------------
def carregar_dominios(ficheiro, sheet):
    df = pd.read_excel(ficheiro, sheet_name=sheet)
    df.dropna(subset=['Dominios'], inplace=True)
    return df['Dominios'].unique().tolist()

# ------------------------------
# Função para classificar com OpenAI LLM
# ------------------------------
def classificar_llm(prompt_texto):
    try:
        resposta = openai.chat.completions.create(
            model="gpt-4o",
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

# Escolher versão ENEI
versao_enei = st.sidebar.radio("Seleciona a versão da ENEI:", ["ENEI 2020", "ENEI 2030"])
st.session_state["versao_enei"] = versao_enei

config_enei = {
    "ENEI 2020": {"ficheiro": "descricao2020.xlsx", "sheet": "Eixos"},
    "ENEI 2030": {"ficheiro": "descricao2030.xlsx", "sheet": "Dominios"}
}

# Upload do ficheiro com projetos reais
uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos reais (.xlsx):", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("📄 Escolhe a folha (sheet):", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)

    colunas = df.columns.tolist()
    col_titulo = st.selectbox("📝 Coluna do título:", colunas, index=colunas.index("Designacao Projecto") if "Designacao Projecto" in colunas else 0)
    col_resumo = st.selectbox("📋 Coluna da descrição/resumo:", colunas, index=colunas.index("Sumario Executivo") if "Sumario Executivo" in colunas else 0)

    # Classificações manuais (opcional)
    col_manual1 = st.selectbox("✅ Classificação manual principal (opcional):", ["Nenhuma"] + colunas, index=colunas.index("Dominio ENEI") + 1 if "Dominio ENEI" in colunas else 0)
    col_manual2 = st.selectbox("📘 Classificação manual alternativa (opcional):", ["Nenhuma"] + colunas, index=colunas.index("Dominio ENEI Projecto") + 1 if "Dominio ENEI Projecto" in colunas else 0)

    dominios = carregar_dominios(config_enei[versao_enei]["ficheiro"], config_enei[versao_enei]["sheet"])

    st.markdown("### ⚙️ Quantos projetos queres classificar?")
    opcao_modo = st.radio("Modo:", ["Teste (1 projeto)", "5", "10", "20", "50", "Todos"])

    if opcao_modo == "Teste (1 projeto)":
        df = df.head(1)
    elif opcao_modo != "Todos":
        df = df.head(int(opcao_modo))

    # Estimativa de tokens
    n_proj = len(df)
    tokens_por_proj = 610
    total_tokens = n_proj * tokens_por_proj
    st.info(f"🧮 Estimativa: {total_tokens} tokens (aprox.) para {n_proj} projetos")

    if st.button("🚀 Classificar com LLM"):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df.iterrows():
                titulo = str(row.get(col_titulo, ""))
                resumo = str(row.get(col_resumo, ""))
                prompt = preparar_prompt(titulo, resumo, dominios)
                classificacao = classificar_llm(prompt)

                linha = {
                    "NIPC": row.get("NIPC", ""),
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Domínio LLM": classificacao
                }

                if col_manual1 != "Nenhuma":
                    linha["Classificação Manual 1"] = row.get(col_manual1, "")
                if col_manual2 != "Nenhuma":
                    linha["Classificação Manual 2"] = row.get(col_manual2, "")

                resultados.append(linha)

        final_df = pd.DataFrame(resultados)
        final_df.index += 1

        st.success("✅ Classificação concluída com sucesso!")
        st.markdown("### 🔎 Resultados")
        st.dataframe(final_df, use_container_width=True)
        
        # Guardar no session_state (permanece visível após o download)
        st.session_state["classificacoes_llm"] = final_df.copy()
        
        # Exportar para Excel
        buffer = BytesIO()
        final_df.to_excel(buffer, index=False)
        st.download_button(
            label="📥 Download (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"classificacao_llm_{versao_enei.replace(' ', '').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
