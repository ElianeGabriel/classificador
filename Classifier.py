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
def _strip(x):
    return ("" if pd.isna(x) else str(x)).strip()

def preparar_prompt(titulo, resumo, dominios):
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
    r = (resposta or "").strip().replace("*", " ")
    r = re.sub(r"\s+", " ", r)
    if r.lower() == "indefinido":
        return "Indefinido"
    partes = [p.strip() for p in re.split(r"[;,]", r) if p.strip()]
    if not partes:
        return "Indefinido"
    return ", ".join(partes[:2])

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
        dominios.append({"nome": nome, "texto": texto_completo})
    if not dominios:
        st.error("Lista de domínios ficou vazia. Verifica as colunas/linhas do ficheiro.")
        st.stop()
    return dominios

def classificar_llm(prompt_texto):
    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt_texto}],
            temperature=0
        )
        ch = resp.choices[0]
        content = (ch.message.content or "").strip()
        finish = getattr(ch, "finish_reason", None)
        if finish and finish != "stop":
            st.warning(f"LLM terminou com finish_reason='{finish}'.")
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
def embeddings_dos_dominios_cache(dominios, versao_enei: str):
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

# ---------------- coalesce multi-sheet ----------------
def build_sources_options(sheets_map):
    """
    Devolve lista ['Sheet::Coluna'] para todas as sheets com coluna 'cand'.
    Ignora a coluna 'cand' na lista de opções.
    """
    options = []
    for sname, df in sheets_map.items():
        if 'cand' not in df.columns:
            continue
        for col in df.columns:
            if col == 'cand':
                continue
            options.append(f"{sname}::{col}")
    return sorted(options)

def series_from_source(sheets_map, source):
    """Devolve DataFrame com ['cand', source_alias] a partir de 'Sheet::Col'."""
    sname, col = source.split("::", 1)
    if sname not in sheets_map:
        return pd.DataFrame(columns=["cand", source])
    df = sheets_map[sname]
    if 'cand' not in df.columns or col not in df.columns:
        return pd.DataFrame(columns=["cand", source])
    tmp = df[['cand', col]].copy()
    tmp['cand'] = tmp['cand'].apply(_strip)
    tmp[source] = tmp[col].apply(_strip)
    tmp = tmp.drop(columns=[col])
    return tmp[tmp[source] != ""]

def coalesce_from_sources(sheets_map, sources, alias):
    """
    Faz outer-join por 'cand' de todas as fontes e coalesce por ordem.
    Retorna DataFrame ['cand', alias]
    """
    if not sources:
        return pd.DataFrame(columns=["cand", alias])
    acc = None
    for src in sources:
        s = series_from_source(sheets_map, src)
        if s.empty:
            continue
        if acc is None:
            acc = s
        else:
            acc = acc.merge(s, on="cand", how="outer")
    if acc is None or acc.empty:
        return pd.DataFrame(columns=["cand", alias])
    # coalesce por ordem das fontes
    vals_cols = [src for src in sources if src in acc.columns]
    acc[alias] = ""
    for c in vals_cols:
        acc[alias] = acc[alias].mask(acc[alias] == "", acc[c])
    acc = acc[['cand', alias]].copy()
    acc[alias] = acc[alias].apply(_strip)
    acc = acc[acc[alias] != ""]
    return acc

# -------------------------------------------------
# UI
# -------------------------------------------------
def run():
    st.markdown("### 🤖 Classificador Automático com LLM (Azure OpenAI)")

    # Diagnóstico Azure
    with st.expander("⚙️ Diagnóstico Azure/OpenAI"):
        colA, colB, colC = st.columns(3)
        colA.write(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT') or '—'}")
        colB.write(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION') or '—'}")
        colC.write(f"Chat Deployment: {CHAT_DEPLOYMENT or '—'}")
        if st.button("▶️ Testar Azure Chat"):
            try:
                r = client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[{"role": "user", "content": "pong?"}],
                    temperature=0
                )
                st.success(f"OK: {r.choices[0].message.content!r}")
            except Exception as e:
                st.error(f"Falha: {e}")

    if not CHAT_DEPLOYMENT:
        st.error("**AZURE_OPENAI_DEPLOYMENT** não definido.")
        st.stop()

    versao_enei = st.sidebar.radio("Seleciona a versão da ENEI:", ["ENEI 2020", "ENEI 2030"])
    st.session_state["versao_enei"] = versao_enei

    config_enei = {
        "ENEI 2020": {"ficheiro": "descricao2020.xlsx", "sheet": "Eixos"},
        "ENEI 2030": {"ficheiro": "descricao2030.xlsx", "sheet": "Dominios"}
    }

    uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos (.xlsx):", type=["xlsx"])
    if not uploaded_file:
        st.info("Carrega um ficheiro .xlsx para começar.")
        return

    # Lê todas as sheets para permitir fontes multi-sheet
    xls = pd.ExcelFile(uploaded_file)
    sheets_map = {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names}
    # normalizar 'cand' para string em todas as sheets que a possuam
    for s in sheets_map.values():
        if 'cand' in s.columns:
            s['cand'] = s['cand'].apply(_strip)

    # Opções de fontes "sheet::coluna"
    all_sources = build_sources_options(sheets_map)

    st.markdown("#### 🔎 Seleção de Fontes (podes misturar sheets diferentes)")
    st.caption("Escolhe **uma ou mais** fontes para cada campo. A primeira não vazia por cand é usada. O **Resumo é obrigatório**; o **Título é opcional**.")

    # Sugestões de defaults por keywords
    def guess_sources(keywords, limit=3):
        out = []
        for src in all_sources:
            _, col = src.split("::", 1)
            lc = col.lower()
            if any(k in lc for k in keywords):
                out.append(src)
        return out[:limit]

    default_title = guess_sources(["título", "titulo", "designa", "nome"], 3)
    default_summary = guess_sources(["resumo", "sumár", "sumar", "abstract", "descri", "objetiv", "objectiv"], 3)

    titulo_sources = st.multiselect("📝 Fontes para **TÍTULO** (opcional)", options=all_sources, default=default_title)
    resumo_sources = st.multiselect("📋 Fontes para **RESUMO** (obrigatório)", options=all_sources, default=default_summary)

    if not resumo_sources:
        st.error("Precisas de escolher pelo menos **uma** fonte para o RESUMO.")
        st.stop()

    # Construir Título/Resumo coalescidos por cand
    df_title = coalesce_from_sources(sheets_map, titulo_sources, "__TITULO__") if titulo_sources else pd.DataFrame(columns=["cand", "__TITULO__"])
    df_sum   = coalesce_from_sources(sheets_map, resumo_sources, "__RESUMO__")

    if df_sum.empty:
        st.error("As fontes de **RESUMO** não produziram valores. Ajusta a seleção acima.")
        st.stop()

    # Juntar título (opcional) com resumo (obrigatório)
    df_base = df_sum
    if not df_title.empty:
        df_base = df_base.merge(df_title, on="cand", how="left")
    else:
        df_base["__TITULO__"] = ""  # título opcional

    # Sheet/coluna de classificações manuais (opcional)
    st.markdown("#### ✅ Classificações manuais (opcional)")
    manual_sheet = st.selectbox("Sheet de classificações manuais:", ["(Nenhuma)"] + xls.sheet_names)
    if manual_sheet != "(Nenhuma)":
        df_class = sheets_map[manual_sheet].copy()
        if 'cand' not in df_class.columns:
            st.error("A sheet de manuais selecionada não tem coluna **'cand'**.")
            st.stop()
        df_class['cand'] = df_class['cand'].apply(_strip)
        col_manual = st.selectbox("Coluna das classificações manuais:", [c for c in df_class.columns if c != "cand"])
        df_class = df_class.groupby("cand").agg({
            col_manual: lambda x: "; ".join(sorted(set(_strip(v) for v in x if _strip(v))))
        }).rename(columns={col_manual: "Classificação Manual"}).reset_index()
        df_final = df_base.merge(df_class, on="cand", how="inner")
        inter = len(set(df_base['cand']).intersection(set(df_class['cand'])))
    else:
        df_final = df_base.copy()
        df_final["Classificação Manual"] = ""
        inter = "N/A"

    # Diagnóstico
    st.info(
        "🧾 Contagens | "
        f"cands com RESUMO: {df_sum['cand'].nunique()} | "
        f"cands com TÍTULO: {df_title['cand'].nunique() if not df_title.empty else 0} | "
        f"Interseção com manuais: {inter} | "
        f"Linhas após merge: {len(df_final)}"
    )

    if manual_sheet != "(Nenhuma)" and df_final.empty:
        st.warning("Não há interseção de 'cand' entre RESUMO (e TÍTULO) e os manuais. Vou **prosseguir sem manuais**.")
        df_final = df_base.copy()
        df_final["Classificação Manual"] = ""

    # Escolha de quantidade
    quantidade = st.radio("Quantas candidaturas queres classificar?", ["1", "5", "10", "20", "50", "Todas"], horizontal=True)
    df_filtrado = df_final if quantidade == "Todas" else df_final.head(int(quantidade))

    # Carregar domínios
    ficheiro_desc = config_enei[versao_enei]["ficheiro"]
    sheet_desc = config_enei[versao_enei]["sheet"]
    dominios = carregar_dominios(ficheiro_desc, sheet_desc)

    # Percentagens por similaridade (opcional)
    mostrar_percentagens = st.checkbox(
        "Adicionar percentagens baseadas em similaridade (embeddings)",
        value=False,
        help="Calculadas por similaridade coseno entre o texto do projeto e as descrições dos domínios."
    )
    emb_dom_map = {}
    if mostrar_percentagens:
        if not EMB_DEPLOYMENT:
            st.warning("Defina **AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT** para usar percentagens por similaridade.")
        else:
            emb_dom_map = embeddings_dos_dominios_cache(dominios, versao_enei)

    st.info(f"🧮 Estimativa rápida: ~{len(df_filtrado) * 600} tokens (aprox.)")

    modo_debug = st.checkbox("🛠️ Modo debug (mostrar prompt e resposta crua por linha)", value=False)

    if st.button("🚀 Classificar com LLM", use_container_width=True):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df_filtrado.iterrows():
                titulo = _strip(row["__TITULO__"])  # pode ser ""
                resumo = _strip(row["__RESUMO__"])  # obrigatório

                prompt = preparar_prompt(titulo, resumo, dominios)
                resposta = classificar_llm(prompt)

                if not resposta:
                    st.error(f"❌ LLM devolveu vazio. cand={row['cand']} | Título='{titulo[:80]}'")
                    if modo_debug:
                        with st.expander(f"Debug cand={row['cand']}"):
                            st.code(prompt, language="markdown")
                            st.write("**Resposta crua do LLM:** (string vazia)")
                    dominios_llm = "Indefinido"
                else:
                    if modo_debug:
                        with st.expander(f"Debug cand={row['cand']}"):
                            st.code(prompt, language="markdown")
                            st.write("**Resposta crua do LLM:**")
                            st.text(resposta)
                    dominios_llm = extrair_resposta_formatada(resposta)

                saida = dominios_llm
                if mostrar_percentagens and dominios_llm.lower() != "indefinido" and emb_dom_map:
                    sims = percentagens_por_similaridade(titulo, resumo, dominios, emb_dom_map)
                    saida = formatar_com_percentagens(dominios_llm, sims)

                resultados.append({
                    "cand": row["cand"],
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Classificação Manual": row.get("Classificação Manual", ""),
                    "Domínios LLM": saida
                })

        if not resultados:
            st.error("🚫 Nada classificado. Ajusta as fontes (sobretudo de RESUMO) e testa o Azure no painel de diagnóstico.")
        else:
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
            file_name=f"classificacao_llm_{st.session_state.get('versao_enei','enei').replace(' ', '').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    run()
