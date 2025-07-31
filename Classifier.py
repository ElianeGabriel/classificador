import streamlit as st
import pandas as pd
import openai
import os
from io import BytesIO
import re

# ------------------------------
# API KEY (por variável de ambiente segura)
# ------------------------------
#openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------
# Preparar o prompt para o LLM
# ------------------------------
def preparar_prompt(titulo, resumo, dominios):
    prompt = f"""
Classifica o projeto abaixo num ou dois dos seguintes domínios prioritários da Estratégia Nacional de Especialização Inteligente ({st.session_state.get('versao_enei', 'ENEI')}):

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
        
    # Mostrar os domínios carregados no Streamlit
    #st.write("🧾 Domínios carregados:", dominios)
    #st.write(f"Total de domínios carregados: {len(dominios)}")    
    return dominios

# ------------------------------
# Função para classificar com OpenAI LLM
# ------------------------------
#def classificar_llm(prompt_texto):
#    try:
#        resposta = openai.chat.completions.create(
#            model="gpt-4o",
#            messages=[{"role": "user", "content": prompt_texto}],
#            temperature=0
#        )
#        return resposta.choices[0].message.content.strip()
#    except Exception as e:
#        return f"Erro: {e}"

def classificar_llm(prompt_texto):
    try:
        resposta = openai.ChatCompletion.create(
            engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_type="azure",
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            messages=[{"role": "user", "content": prompt_texto}],
            temperature=0
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {e}"

# ------------------------------
# Extrair domínios e percentagens da resposta
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
st.markdown("### 🤖 Classificador Automático com LLM (OpenAI)")

versao_enei = st.sidebar.radio("Seleciona a versão da ENEI:", ["ENEI 2020", "ENEI 2030"])
st.session_state["versao_enei"] = versao_enei

config_enei = {
    "ENEI 2020": {"ficheiro": "descricao2020.xlsx", "sheet": "Eixos"},
    "ENEI 2030": {"ficheiro": "descricao2030.xlsx", "sheet": "Dominios"}
}

uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos reais (.xlsx):", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("📄 Escolhe a folha (sheet):", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)

    colunas = df.columns.tolist()
    col_titulo = st.selectbox("📝 Coluna do título:", colunas)
    col_resumo = st.selectbox("📋 Coluna do resumo:", colunas)
    col_manual1 = st.selectbox("✅ Classificação manual principal (opcional):", ["Nenhuma"] + colunas)
    col_manual2 = st.selectbox("📘 Classificação manual alternativa (opcional):", ["Nenhuma"] + colunas)

    dominios = carregar_dominios(config_enei[versao_enei]["ficheiro"], config_enei[versao_enei]["sheet"])

    modo_classificacao = st.radio("Modo de classificação:", ["Classificação normal", "Classificação por grupo (ex: por domínio manual)"])

    if modo_classificacao == "Classificação normal":
        quantidade = st.radio("Quantos projetos queres classificar?", ["1", "5", "10", "20", "50", "Todos"])
        df_filtrado = df if quantidade == "Todos" else df.head(int(quantidade))
    else:
        coluna_grupo = st.selectbox("📂 Coluna para agrupar (ex: domínio manual):", colunas)
        n_grupo = st.selectbox("📌 Nº de projetos por grupo:", [1, 2, 3, 5, 10])
        df_filtrado = df.groupby(df[coluna_grupo], group_keys=False).apply(lambda x: x.sample(n=min(n_grupo, len(x)), random_state=42))

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
                    "NIPC": row.get("NIPC", ""),
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Domínio LLM 1": d1,
                    "% 1": p1,
                    "Domínio LLM 2": d2,
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
