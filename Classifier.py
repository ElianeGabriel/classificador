import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from openai import AzureOpenAI

# ------------------------------
# Cliente Azure OpenAI
# ------------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# ------------------------------
# Preparar o prompt para o LLM
# ------------------------------
def preparar_prompt(titulo, resumo, dominios):
    prompt = f"""
Classifica o projeto abaixo num ou dois dos seguintes domínios prioritários da Estratégia Nacional de Especialização Inteligente ({st.session_state.get('versao_enei', 'ENEI')}):\n
{chr(10).join([f"- {d}" for d in dominios])}

Projeto:
Título: {titulo}
Descrição: {resumo}

Responde com os dois domínios mais adequados por ordem de relevância, seguidos da percentagem estimada (ex: 1. Saúde (60%), 2. Energia (40%)). Se não conseguires decidir com certeza, responde apenas com "Indefinido".
""".strip()
    return prompt

# ------------------------------
# Carregar domínios com descrição e área
# ------------------------------
def carregar_dominios(ficheiro, sheet):
    df = pd.read_excel(ficheiro, sheet_name=sheet)
    df.dropna(subset=['Dominios'], inplace=True)

    dominios = []
    for _, row in df.iterrows():
        nome = str(row['Dominios']).strip()
        descricao = str(row.get('Descrição', '')).strip()
        area = str(row.get('Principal área de atuação (Opções de Resposta)', '')).strip()

        texto_completo = f"{nome}. {descricao}"
        if area:
            texto_completo += f" ({area})"
        dominios.append(texto_completo)
    return dominios

# ------------------------------
# Classificar com Azure OpenAI
# ------------------------------
def classificar_llm(prompt_texto):
    try:
        resposta = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt_texto}],
            temperature=0
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {e}"

# ------------------------------
# Extrair domínios e percentagens
# ------------------------------
def extrair_dominios_e_percentagens(resposta):
    if resposta.lower().strip() == "indefinido":
        return ("Indefinido", "", "", "")
    padrao = r"\d+\.\s*(.*?)\s*\((\d+)%\)"
    correspondencias = re.findall(padrao, resposta)
    if len(correspondencias) >= 2:
        return correspondencias[0][0], correspondencias[0][1], correspondencias[1][0], correspondencias[1][1]
    elif len(correspondencias) == 1:
        return correspondencias[0][0], correspondencias[0][1], "", ""
    else:
        return resposta, "", "", ""

# ------------------------------
# INTERFACE
# ------------------------------
st.markdown("### 🤖 Classificador Automático com LLM (Azure OpenAI)")

versao_enei = st.sidebar.radio("Seleciona a versão da ENEI:", ["ENEI 2020", "ENEI 2030"])
st.session_state["versao_enei"] = versao_enei

config_enei = {
    "ENEI 2020": {"ficheiro": "descricao2020.xlsx", "sheet": "Eixos"},
    "ENEI 2030": {"ficheiro": "descricao2030.xlsx", "sheet": "Dominios"}
}

uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos reais (.xlsx):", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_dados = st.selectbox("📄 Sheet com o título/resumo:", xls.sheet_names)
    sheet_class = st.selectbox("📑 Sheet com classificações manuais (ou candidaturas duplicadas):", xls.sheet_names)

    df_dados = pd.read_excel(xls, sheet_name=sheet_dados)
    df_class = pd.read_excel(xls, sheet_name=sheet_class)

    if 'cand' not in df_dados.columns or 'cand' not in df_class.columns:
        st.error("Ambas as sheets devem conter a coluna 'cand' para cruzamento.")
        st.stop()

    col_titulo = st.selectbox("📝 Coluna do título:", df_dados.columns)
    col_resumo = st.selectbox("📋 Coluna do resumo:", df_dados.columns)
    col_manual1 = st.selectbox("✅ Classificação manual 1 (opcional):", ["Nenhuma"] + df_class.columns.tolist())
    col_manual2 = st.selectbox("📘 Classificação manual 2 (opcional):", ["Nenhuma"] + df_class.columns.tolist())

    # Merge
    df_merged = df_class.merge(df_dados[['cand', col_titulo, col_resumo]], on='cand', how='left')

    if df_merged[col_titulo].isna().all():
        st.error("Não foi possível cruzar os dados. Verifica se os valores de 'cand' coincidem entre as sheets.")
        st.stop()

    dominios = carregar_dominios(config_enei[versao_enei]["ficheiro"], config_enei[versao_enei]["sheet"])

    quantidade = st.radio("Quantos projetos queres classificar?", ["1", "5", "10", "20", "50", "Todos"])
    df_filtrado = df_merged if quantidade == "Todos" else df_merged.head(int(quantidade))

    st.info(f"🧮 Estimativa: {len(df_filtrado) * 610} tokens (aprox.)")

    if st.button("🚀 Classificar com LLM"):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df_filtrado.iterrows():
                titulo = str(row.get(col_titulo, ""))
                resumo = str(row.get(col_resumo, ""))
                prompt = preparar_prompt(titulo, resumo, dominios)
                resposta = classificar_llm(prompt)
                d1, p1, d2, p2 = extrair_dominios_e_percentagens(resposta)

                linha = {
                    "cand": row.get("cand", ""),
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Domínio LLM 1": d1.replace("*", ""),
                    "% 1": p1,
                    "Domínio LLM 2": d2.replace("*", ""),
                    "% 2": p2
                }

                if col_manual1 != "Nenhuma":
                    linha["Classificação Manual 1"] = row.get(col_manual1, "")
                if col_manual2 != "Nenhuma":
                    linha["Classificação Manual 2"] = row.get(col_manual2, "")

                resultados.append(linha)

        final_df = pd.DataFrame(resultados)
        final_df.index += 1
        st.session_state["classificacoes_llm"] = final_df

if "classificacoes_llm" in st.session_state:
    st.success("✅ Classificação concluída com sucesso!")
    st.markdown("### 🔎 Resultados")
    st.dataframe(st.session_state["classificacoes_llm"], use_container_width=True)

    buffer = BytesIO()
    st.session_state["classificacoes_llm"].to_excel(buffer, index=False)
    st.download_button(
        label="📥 Download (.xlsx)",
        data=buffer.getvalue(),
        file_name=f"classificacao_llm_{versao_enei.replace(' ', '').lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
