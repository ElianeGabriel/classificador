import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import numpy as np
from openai import AzureOpenAI

# -------------------------------------------------
# Cliente Azure OpenAI
# -------------------------------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# -------------------------------------------------
# Helpers de Prompt / Parsing
# -------------------------------------------------
def preparar_prompt(titulo, resumo, dominios):
    """Força o LLM a devolver apenas 'DOMÍNIO_1; DOMÍNIO_2' (ou um só, ou Indefinido)."""
    nomes = [d["nome"] for d in dominios]
    prompt = f"""
Classifica o projeto em até dois domínios da {st.session_state.get('versao_enei', 'ENEI')}.

Lista de domínios possíveis:
{chr(10).join([f"- {d}" for d in nomes])}

Projeto:
Título: {titulo}
Descrição: {resumo}

Responde EXCLUSIVAMENTE numa ÚNICA linha, no formato:
DOMÍNIO_1; DOMÍNIO_2

Regras:
- Sem texto extra, sem explicações, sem percentagens.
- Se só houver um domínio claro, devolve apenas "DOMÍNIO_1".
- Se não conseguires decidir, devolve "Indefinido".
""".strip()
    return prompt

def extrair_resposta_formatada(resposta):
    """Normaliza para 'A, B' (ou 'A'), aceita ';' e ',' como separadores."""
    r = (resposta or "").strip().replace("*", " ")
    # tudo numa linha
    r = re.sub(r"\s+", " ", r)
    if r.lower() == "indefinido":
        return "Indefinido"
    partes = [p.strip() for p in re.split(r"[;,]", r) if p.strip()]
    if not partes:
        return "Indefinido"
    return ", ".join(partes[:2])

# -------------------------------------------------
# Leitura dos Domínios
# -------------------------------------------------
def carregar_dominios(ficheiro, sheet):
    """
    Devolve uma lista de dicts: {"nome": <nome>, "texto": <nome + descrição (+ área)>}
    """
    df = pd.read_excel(ficheiro, sheet_name=sheet)
    df.dropna(subset=['Dominios'], inplace=True)

    dominios = []
    for _, row in df.iterrows():
        nome = str(row['Dominios']).strip()
        descricao = str(row.get('Descrição', '')).strip()
        area = str(row.get('Principal área de atuação (Opções de Resposta)', '')).strip()
        texto_completo = f"{nome}. {descricao}" + (f" ({area})" if area else "")
        dominios.append({"nome": nome, "texto": texto_completo})
    return dominios

# -------------------------------------------------
# LLM Chat
# -------------------------------------------------
def classificar_llm(prompt_texto):
    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt_texto}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {e}"

# -------------------------------------------------
# Embeddings + Similaridade
# -------------------------------------------------
def obter_embedding(texto: str):
    try:
        resp = client.embeddings.create(
            model=EMB_DEPLOYMENT,
            input=texto
        )
        return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        st.warning(f"Falha ao obter embeddings: {e}")
        return None

@st.cache_data(show_spinner=False)
def embeddings_dos_dominios_cache(dominios, versao_enei: str):
    """Cacheia embeddings de cada domínio por versão ENEI e texto do domínio."""
    emb_map = {}
    for d in dominios:
        emb = obter_embedding(d["texto"])
        if emb is not None:
            emb_map[d["nome"]] = emb
    return emb_map

def percentagens_por_similaridade(titulo, resumo, dominios, emb_dom_map):
    """Calcula similaridade coseno projeto -> domínios, devolve {nome: score}."""
    texto_proj = f"{titulo}\n\n{resumo}".strip()
    emb_proj = obter_embedding(texto_proj)
    if emb_proj is None:
        return {}

    sims = {}
    norm_proj = np.linalg.norm(emb_proj) + 1e-12
    for nome, emb_dom in emb_dom_map.items():
        sim = float(np.dot(emb_proj, emb_dom) / (norm_proj * (np.linalg.norm(emb_dom) + 1e-12)))
        sims[nome] = max(sim, 0.0)  # corta negativos
    return sims

def formatar_com_percentagens(dominios_llm_str, sims_dict):
    """Recebe 'A, B' e dicionário {nome: sim}, devolve 'A (xx%), B (yy%)' normalizado a 100."""
    if dominios_llm_str.lower() == "indefinido":
        return "Indefinido"
    nomes = [p.strip() for p in dominios_llm_str.split(",") if p.strip()]
    if not nomes:
        return "Indefinido"
    valores = [(n, sims_dict.get(n, 0.0)) for n in nomes]
    total = sum(v for _, v in valores) or 1e-12
    percent = {n: round(100 * v / total) for n, v in valores}
    # ajusta para somar exatamente 100
    soma = sum(percent.values())
    if soma != 100:
        primeiro = nomes[0]
        percent[primeiro] = percent[primeiro] + (100 - soma)
    return ", ".join([f"{n} ({percent[n]}%)" for n in nomes])

# -------------------------------------------------
# UI
# -------------------------------------------------
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

    # 1) Limpar linhas inválidas (sem título/resumo)
    df_dados_validos = df_dados.dropna(subset=[col_titulo, col_resumo])

    # 2) Interseção de cands com dados + classificações manuais
    cands_validos = set(df_dados_validos['cand']).intersection(set(df_class['cand']))

    # 3) Filtrar
    df_dados_validos = df_dados_validos[df_dados_validos['cand'].isin(cands_validos)]
    df_class_filtrado = df_class[df_class['cand'].isin(cands_validos)]

    # 4) Agregar classificações manuais por cand
    classificacoes_agrupadas = df_class_filtrado.groupby('cand').agg({
        col_manual: lambda x: "; ".join(sorted(set(str(v).strip() for v in x if pd.notna(v))))
    }).rename(columns={col_manual: "Classificação Manual"}).reset_index()

    # 5) Uma linha por cand (primeira com título/resumo)
    dados_unicos = df_dados_validos.groupby('cand').first().reset_index()

    # 6) Merge
    df_final = dados_unicos.merge(classificacoes_agrupadas, on='cand', how='inner')

    # Quantidade a classificar
    quantidade = st.radio("Quantas candidaturas queres classificar?", ["1", "5", "10", "20", "50", "Todas"], horizontal=True)
    df_filtrado = df_final if quantidade == "Todas" else df_final.head(int(quantidade))

    # Carregar domínios (nome + texto)
    ficheiro_desc = config_enei[versao_enei]["ficheiro"]
    sheet_desc = config_enei[versao_enei]["sheet"]
    dominios = carregar_dominios(ficheiro_desc, sheet_desc)

    # Opcional: percentagens por similaridade
    mostrar_percentagens = st.checkbox("Adicionar percentagens baseadas em similaridade (embeddings)", value=False,
                                       help="Se marcado, as percentagens são calculadas por similaridade coseno entre o texto do projeto e as descrições dos domínios. Se não fizer sentido, deixa desligado.")

    # Pré-computar embeddings dos domínios (cache)
    emb_dom_map = {}
    if mostrar_percentagens:
        if not EMB_DEPLOYMENT:
            st.warning("Defina AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT para usar percentagens por similaridade.")
        else:
            emb_dom_map = embeddings_dos_dominios_cache(dominios, versao_enei)

    st.info(f"🧮 Estimativa rápida: ~{len(df_filtrado) * 600} tokens (aprox.)")

    if st.button("🚀 Classificar com LLM", use_container_width=True):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df_filtrado.iterrows():
                titulo = str(row.get(col_titulo, "")).strip()
                resumo = str(row.get(col_resumo, "")).strip()

                prompt = preparar_prompt(titulo, resumo, dominios)
                resposta = classificar_llm(prompt)
                dominios_llm = extrair_resposta_formatada(resposta)  # "A, B" ou "A" ou "Indefinido"

                saida = dominios_llm
                if mostrar_percentagens and dominios_llm.lower() != "indefinido" and emb_dom_map:
                    sims = percentagens_por_similaridade(titulo, resumo, dominios, emb_dom_map)
                    saida = formatar_com_percentagens(dominios_llm, sims)

                linha = {
                    "cand": row["cand"],
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Classificação Manual": row.get("Classificação Manual", ""),
                    "Domínios LLM": saida
                }
                resultados.append(linha)

        final_df = pd.DataFrame(resultados)
        final_df.index += 1
        st.session_state["classificacoes_llm"] = final_df

# -------------------------------------------------
# Resultados + Download
# -------------------------------------------------
if "classificacoes_llm" in st.session_state:
    st.success("✅ Classificação concluída com sucesso!")
    st.markdown("### 🔎 Resultados")
    st.dataframe(st.session_state["classificacoes_llm"], use_container_width=True)

    buffer = BytesIO()
    st.session_state["classificacoes_llm"].to_excel(buffer, index=False)
    st.download_button(
        label="📥 Download (.xlsx)",
        data=buffer.getvalue(),
        file_name=f"classificacao_llm_{st.session_state.get('versao_enei','enei').replace(' ', '').lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
