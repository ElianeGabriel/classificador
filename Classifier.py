import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import numpy as np
from openai import AzureOpenAI

# -------------------------------------------------
# Azure OpenAI
# -------------------------------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMB_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# -------------------------------------------------
# Helpers gerais
# -------------------------------------------------
def _strip(s):
    return ("" if pd.isna(s) else str(s)).strip()

def coalesce_row(row, cols):
    for c in cols:
        if c in row:
            v = _strip(row[c])
            if v:
                return v
    return ""

def guess_column(columns, keywords):
    cols_lower = {c: c.lower() for c in columns}
    for kw in keywords:
        for c, lc in cols_lower.items():
            if kw in lc:
                return c
    return None

# -------------------------------------------------
# Prompts / Parsing
# -------------------------------------------------
def preparar_prompt(titulo, resumo, dominios, etiqueta_taxonomia):
    nomes = [d["nome"] for d in dominios]
    prompt = f"""
Classifica o projeto em até dois domínios da {etiqueta_taxonomia}.

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
    r = (resposta or "").strip().replace("*", " ")
    r = re.sub(r"\s+", " ", r)
    if r.lower() == "indefinido":
        return "Indefinido"
    partes = [p.strip() for p in re.split(r"[;,]", r) if p.strip()]
    if not partes:
        return "Indefinido"
    return ", ".join(partes[:2])

# -------------------------------------------------
# Ficheiro de domínios (ENEI / EREI)
# -------------------------------------------------
def carregar_dominios(ficheiro, sheet):
    try:
        df = pd.read_excel(ficheiro, sheet_name=sheet)
    except FileNotFoundError:
        st.error(f"Ficheiro de domínios não encontrado: **{ficheiro}**.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao ler ficheiro de domínios: {e}")
        st.stop()

    if "Dominios" not in df.columns:
        st.error("A sheet de domínios tem de ter a coluna **'Dominios'**.")
        st.stop()

    df = df.dropna(subset=['Dominios']).copy()
    dominios = []
    for _, row in df.iterrows():
        nome = _strip(row['Dominios'])
        descricao = _strip(row.get('Descrição', ''))
        area = _strip(row.get('Principal área de atuação (Opções de Resposta)', ''))
        texto_completo = f"{nome}. {descricao}" + (f" ({area})" if area else "")
        if nome:
            dominios.append({"nome": nome, "texto": texto_completo})

    if not dominios:
        st.error("Lista de domínios ficou vazia. Verifica as colunas/linhas do ficheiro.")
        st.stop()
    return dominios

# -------------------------------------------------
# Azure Chat + Embeddings
# -------------------------------------------------
def classificar_llm(prompt_texto):
    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt_texto}],
            temperature=0
        )
        ch = resp.choices[0]
        content = (ch.message.content or "").strip()
        return content
    except Exception as e:
        st.error(f"Erro no Azure OpenAI (chat): {e}")
        return ""

def obter_embedding(texto: str):
    try:
        resp = client.embeddings.create(model=EMB_DEPLOYMENT, input=texto)
        return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        st.warning(f"Falha ao obter embeddings: {e}")
        return None

@st.cache_data(show_spinner=False)
def embeddings_dos_dominios_cache(dominios, etiqueta_taxonomia: str):
    emb_map = {}
    for d in dominios:
        emb = obter_embedding(d["texto"])
        if emb is not None:
            emb_map[d["nome"]] = emb
    return emb_map

def percentagens_por_similaridade(titulo, resumo, dominios, emb_dom_map):
    texto_proj = f"{titulo}\n\n{resumo}".strip()
    emb_proj = obter_embedding(texto_proj)
    if emb_proj is None:
        return {}
    sims = {}
    norm_proj = np.linalg.norm(emb_proj) + 1e-12
    for nome, emb_dom in emb_dom_map.items():
        sim = float(np.dot(emb_proj, emb_dom) / (norm_proj * (np.linalg.norm(emb_dom) + 1e-12)))
        sims[nome] = max(sim, 0.0)
    return sims

def formatar_com_percentagens(dominios_llm_str, sims_dict):
    if dominios_llm_str.lower() == "indefinido":
        return "Indefinido"
    nomes = [p.strip() for p in dominios_llm_str.split(",") if p.strip()]
    if not nomes:
        return "Indefinido"
    valores = [(n, sims_dict.get(n, 0.0)) for n in nomes]
    total = sum(v for _, v in valores) or 1e-12
    percent = {n: round(100 * v / total) for n, v in valores}
    soma = sum(percent.values())
    if soma != 100:
        primeiro = nomes[0]
        percent[primeiro] = percent[primeiro] + (100 - soma)
    return ", ".join([f"{n} ({percent[n]}%)" for n in nomes])

# -------------------------------------------------
# UI
# -------------------------------------------------
def run():
    st.subheader("🧠 Classificação Automática (ENEI / EREI)")

    if not CHAT_DEPLOYMENT:
        st.error("Variável **AZURE_OPENAI_DEPLOYMENT** não definida.")
        st.stop()

    # =========================
    # Escolha da TAXONOMIA
    # =========================
    taxonomia = st.sidebar.radio("Taxonomia:", ["ENEI", "EREI"], horizontal=True)

    ficheiro_desc = None
    sheet_desc = None

    if taxonomia == "ENEI":
        versao_enei = st.sidebar.radio("Versão ENEI:", ["ENEI 2020", "ENEI 2030"])
        st.session_state["versao_enei"] = versao_enei
        config_enei = {
            "ENEI 2020": {"ficheiro": "descricao2020.xlsx", "sheet": "Eixos"},
            "ENEI 2030": {"ficheiro": "descricao2030.xlsx", "sheet": "Dominios"}
        }
        ficheiro_desc = config_enei[versao_enei]["ficheiro"]
        sheet_desc = config_enei[versao_enei]["sheet"]
        etiqueta_taxonomia = versao_enei
    else:
        # EREI: ficheiro único com várias sheets (uma por região)
        ficheiro_erei = "DominiosEREI_10112025.xlsx"
        try:
            xls_erei = pd.ExcelFile(ficheiro_erei)
            sheet_erei = st.sidebar.selectbox("Região (sheet EREI):", xls_erei.sheet_names)
        except Exception as e:
            st.error(f"Não consegui abrir o ficheiro das EREI ({ficheiro_erei}): {e}")
            st.stop()
        ficheiro_desc = ficheiro_erei
        sheet_desc = sheet_erei
        etiqueta_taxonomia = f"EREI – {sheet_erei}"

    st.session_state["versao_enei"] = etiqueta_taxonomia

    # =========================
    # Carregar ficheiro de projetos
    # =========================
    uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos (.xlsx):", type=["xlsx"])
    if not uploaded_file:
        st.info("Carrega um ficheiro Excel para começar.")
        return

    xls = pd.ExcelFile(uploaded_file)

    # Escolha de sheets
    sheet_titulo = st.selectbox("📄 Sheet do TÍTULO (opcional):", ["(Nenhuma)"] + xls.sheet_names)
    sheet_resumo = st.selectbox("📄 Sheet do RESUMO/Descrição (obrigatória):", xls.sheet_names)
    sheet_class  = st.selectbox("📑 Sheet com classificações manuais (opcional):", ["(Nenhuma)"] + xls.sheet_names)

    # Ler dataframes
    df_resumo = pd.read_excel(xls, sheet_name=sheet_resumo)
    id_col_resumo = st.selectbox("🆔 Coluna identificadora (sheet RESUMO):", df_resumo.columns)

    if sheet_titulo != "(Nenhuma)":
        df_titulo = pd.read_excel(xls, sheet_name=sheet_titulo)
        id_col_titulo = st.selectbox("🆔 Coluna identificadora (sheet TÍTULO):", df_titulo.columns)
    else:
        df_titulo = pd.DataFrame()
        id_col_titulo = None

    if sheet_class != "(Nenhuma)":
        df_class = pd.read_excel(xls, sheet_name=sheet_class)
        id_col_class = st.selectbox("🆔 Coluna identificadora (sheet MANUAIS):", df_class.columns)
    else:
        df_class = pd.DataFrame()
        id_col_class = None

    # Normalizar ID
    df_resumo[id_col_resumo] = df_resumo[id_col_resumo].apply(_strip)
    if not df_titulo.empty: df_titulo[id_col_titulo] = df_titulo[id_col_titulo].apply(_strip)
    if not df_class.empty:  df_class[id_col_class]  = df_class[id_col_class].apply(_strip)

    # Detectar colunas
    tit_kw  = ["título", "titulo", "designação", "designacao", "nome do projeto"]
    res_kw  = ["resumo", "sumário", "sumario", "descrição", "descricao", "objetivo", "objectivo"]

    guess_resumo = guess_column(df_resumo.columns, res_kw) or df_resumo.columns[0]
    col_resumo = st.selectbox("📋 Coluna principal do RESUMO/Descrição:", df_resumo.columns,
                              index=df_resumo.columns.get_loc(guess_resumo))
    alt_resumo_cols = st.multiselect("Fallback para RESUMO:", [c for c in df_resumo.columns if c != col_resumo])

    if not df_titulo.empty:
        guess_titulo = guess_column(df_titulo.columns, tit_kw) or df_titulo.columns[0]
        col_titulo = st.selectbox("📝 Coluna principal do TÍTULO:", df_titulo.columns,
                                  index=df_titulo.columns.get_loc(guess_titulo))
        alt_titulo_cols = st.multiselect("Fallback para TÍTULO:", [c for c in df_titulo.columns if c != col_titulo])
    else:
        col_titulo = None
        alt_titulo_cols = []

    # Combinar e limpar
    df_resumo["__RESUMO__"] = df_resumo.apply(lambda r: _strip(r.get(col_resumo)) or coalesce_row(r, alt_resumo_cols), axis=1)
    if not df_titulo.empty:
        df_titulo["__TITULO__"] = df_titulo.apply(lambda r: _strip(r.get(col_titulo)) or coalesce_row(r, alt_titulo_cols), axis=1)
        df_base = df_resumo.merge(df_titulo[[id_col_titulo, "__TITULO__"]], left_on=id_col_resumo, right_on=id_col_titulo, how="left")
    else:
        df_base = df_resumo.copy()
        df_base["__TITULO__"] = ""

    df_validos = df_base[df_base["__RESUMO__"].astype(str).str.strip().ne("")].copy()
    if df_validos.empty:
        st.error("🚫 Nenhum resumo válido encontrado.")
        st.stop()

    # Classificações manuais (opcional)
    if not df_class.empty:
        possiveis_man = [c for c in df_class.columns if c != id_col_class]
        col_manual = st.selectbox("✅ Coluna das classificações manuais:", possiveis_man or ["(Nenhuma)"])
        if possiveis_man and col_manual != "(Nenhuma)":
            df_class = df_class.groupby(id_col_class).agg({
                col_manual: lambda x: "; ".join(sorted(set(_strip(v) for v in x if _strip(v))))
            }).rename(columns={col_manual: "Classificação Manual"}).reset_index()
        else:
            df_class = pd.DataFrame(columns=[id_col_class, "Classificação Manual"])
    else:
        df_class = pd.DataFrame(columns=["ID", "Classificação Manual"])

    # Merge final
    if not df_class.empty:
        df_final = df_validos.merge(df_class, left_on=id_col_resumo, right_on=id_col_class, how="inner")
        if df_final.empty:
            st.info("Sem correspondência com classificações manuais. Prossegue sem.")
            df_final = df_validos.copy()
            df_final["Classificação Manual"] = ""
    else:
        df_final = df_validos.copy()
        df_final["Classificação Manual"] = ""

    st.info(f"Linhas válidas: {len(df_final)}")

    quantidade = st.radio("Quantas candidaturas queres classificar?", ["1", "5", "10", "20", "50", "Todas"], horizontal=True)
    df_filtrado = df_final if quantidade == "Todas" else df_final.head(int(quantidade))

    # Carregar domínios ENEI/EREI
    dominios = carregar_dominios(ficheiro_desc, sheet_desc)

    # Similaridade opcional
    mostrar_percentagens = st.checkbox("Adicionar percentagens por embeddings", value=False)
    emb_dom_map = {}
    if mostrar_percentagens and EMB_DEPLOYMENT:
        emb_dom_map = embeddings_dos_dominios_cache(dominios, etiqueta_taxonomia)

    # ------------------ CLASSIFICAÇÃO ------------------
    if st.button("🚀 Classificar com LLM", use_container_width=True):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df_filtrado.iterrows():
                titulo = _strip(row["__TITULO__"])
                resumo = _strip(row["__RESUMO__"])
                id_val = row[id_col_resumo]

                prompt = preparar_prompt(titulo, resumo, dominios, etiqueta_taxonomia)
                resposta = classificar_llm(prompt)
                dominios_llm = extrair_resposta_formatada(resposta)

                saida = dominios_llm
                if mostrar_percentagens and dominios_llm.lower() != "indefinido" and emb_dom_map:
                    sims = percentagens_por_similaridade(titulo, resumo, dominios, emb_dom_map)
                    saida = formatar_com_percentagens(dominios_llm, sims)

                resultados.append({
                    "ID": id_val,
                    "Título": titulo,
                    "Resumo/Descrição": resumo,
                    "Classificação Manual": row.get("Classificação Manual", ""),
                    "Domínios LLM": saida
                })

        if resultados:
            final_df = pd.DataFrame(resultados)
            final_df.index += 1
            st.session_state["classificacoes_llm"] = final_df
            st.success("✅ Classificação concluída!")
        else:
            st.error("🚫 Nenhum projeto classificado.")

    # ------------------ RESULTADOS ------------------
    if "classificacoes_llm" in st.session_state:
        st.markdown("### 🔎 Resultados")
        st.dataframe(st.session_state["classificacoes_llm"], use_container_width=True)

        buffer = BytesIO()
        st.session_state["classificacoes_llm"].to_excel(buffer, index=False)
        file_label = etiqueta_taxonomia.replace(" ", "_").lower()
        st.download_button(
            label="📥 Download (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"classificacao_llm_{file_label}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    run()
