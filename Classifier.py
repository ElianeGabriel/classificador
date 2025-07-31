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
# Extrair domínios formatados
# ------------------------------
def extrair_resposta_formatada(resposta):
    resposta = resposta.strip()
    if resposta.lower() == "indefinido":
        return "Indefinido"
    resposta = resposta.replace("*", "").replace("\n", " ").strip()
    return resposta

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
    sheet_class = st.selectbox("📑 Sheet com classificações manuais (múltiplas linhas por candidatura):", xls.sheet_names)

    df_dados = pd.read_excel(xls, sheet_name=sheet_dados)
    df_class = pd.read_excel(xls, sheet_name=sheet_class)

    if 'cand' not in df_dados.columns or 'cand' not in df_class.columns:
        st.error("Ambas as sheets devem conter a coluna 'cand'.")
        st.stop()

    col_titulo = st.selectbox("📝 Coluna do título:", df_dados.columns)
    col_resumo = st.selectbox("📋 Coluna do resumo:", df_dados.columns)
    col_manual = st.selectbox("✅ Coluna das classificações manuais:", df_class.columns)

    # Agrupar todas as classificações manuais por cand
    classificacoes_agrupadas = df_class.groupby('cand').agg({
        col_manual: lambda x: "; ".join(sorted(set(str(v).strip() for v in x if pd.notna(v))))
    }).rename(columns={col_manual: "Classificação Manual"}).reset_index()

    # Escolher para cada cand a primeira linha com título e resumo preenchidos
    df_dados_validos = df_dados.dropna(subset=[col_titulo, col_resumo])
    dados_unicos = df_dados_validos.groupby('cand').first().reset_index()

    # Juntar os dois
    df_final = dados_unicos.merge(classificacoes_agrupadas, on='cand', how='left')

    # Quantidade a classificar
    quantidade = st.radio("Quantas candidaturas queres classificar?", ["1", "5", "10", "20", "50", "Todas"])
    df_filtrado = df_final if quantidade == "Todas" else df_final.head(int(quantidade))

    dominios = carregar_dominios(config_enei[versao_enei]["ficheiro"], config_enei[versao_enei]["sheet"])
    st.info(f"🧮 Estimativa: {len(df_filtrado) * 610} tokens (aprox.)")

    if st.button("🚀 Classificar com LLM"):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df_filtrado.iterrows():
                titulo = str(row.get(col_titulo, "")).strip()
                resumo = str(row.get(col_resumo, "")).strip()
                prompt = preparar_prompt(titulo, resumo, dominios)
                resposta = classificar_llm(prompt)
                dominio_llm = extrair_resposta_formatada(resposta)

                linha = {
                    "cand": row["cand"],
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Classificação Manual": row.get("Classificação Manual", ""),
                    "Domínios LLM": dominio_llm
                }
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
