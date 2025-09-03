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
# Utils
# -------------------------------------------------
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
        nome = str(row['Dominios']).strip()
        descricao = str(row.get('Descrição', '')).strip()
        area = str(row.get('Principal área de atuação (Opções de Resposta)', '')).strip()
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

# ---------------- helpers de limpeza/coalesce ----------------
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
    # devolve a primeira coluna cujo nome contiver qualquer keyword
    cols_lower = {c: c.lower() for c in columns}
    for kw in keywords:
        for c, lc in cols_lower.items():
            if kw in lc:
                return c
    return None

# -------------------------------------------------
# UI
# -------------------------------------------------
def run():
    st.markdown("### 🤖 Classificador Automático com LLM (Azure OpenAI)")

    # Diagnóstico rápido Azure
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

    uploaded_file = st.file_uploader("📁 Upload do ficheiro de projetos reais (.xlsx):", type=["xlsx"])
    if not uploaded_file:
        st.info("Carrega um ficheiro .xlsx para começar.")
        return

    xls = pd.ExcelFile(uploaded_file)
    sheet_dados = st.selectbox("📄 Sheet com os dados dos projetos:", xls.sheet_names)
    sheet_class = st.selectbox("📑 Sheet com classificações manuais (opcional):", ["(Nenhuma)"] + xls.sheet_names)

    df_dados = pd.read_excel(xls, sheet_name=sheet_dados)
    df_class = pd.read_excel(xls, sheet_name=sheet_class) if sheet_class != "(Nenhuma)" else pd.DataFrame(columns=["cand"])

    if 'cand' not in df_dados.columns:
        st.error("A sheet de dados tem de conter a coluna **'cand'**.")
        st.stop()
    if sheet_class != "(Nenhuma)" and 'cand' not in df_class.columns:
        st.error("A sheet de manuais tem de conter a coluna **'cand'**.")
        st.stop()

    # ---------- detetar colunas prováveis ----------
    texto_title_keywords  = ["título", "titulo", "designação", "designacao", "nome do projeto", "nome do projecto", "nome do projeto", "nome"]
    texto_resumo_keywords = ["resumo", "sumário", "sumario", "abstract", "descrição", "descricao", "objetivo", "objectivo"]

    guess_titulo = guess_column(df_dados.columns, texto_title_keywords) or df_dados.columns[0]
    guess_resumo = guess_column(df_dados.columns, texto_resumo_keywords) or (df_dados.columns[1] if len(df_dados.columns)>1 else df_dados.columns[0])

    col_titulo = st.selectbox("📝 Coluna principal do título/designação:", df_dados.columns, index=df_dados.columns.get_loc(guess_titulo))
    col_resumo = st.selectbox("📋 Coluna principal do resumo/sumário:", df_dados.columns, index=df_dados.columns.get_loc(guess_resumo))

    # colunas alternativas para coalesce
    alt_titulo_cols = st.multiselect(
        "Opções de fallback para TÍTULO (usadas quando a principal vier vazia nessa linha):",
        df_dados.columns,
        default=[c for c in df_dados.columns if c != col_titulo and guess_column([c], texto_title_keywords)]
    )
    alt_resumo_cols = st.multiselect(
        "Opções de fallback para RESUMO (usadas quando a principal vier vazia nessa linha):",
        df_dados.columns,
        default=[c for c in df_dados.columns if c != col_resumo and guess_column([c], texto_resumo_keywords)]
    )

    # ------------- limpeza/coalesce linha-a-linha -------------
    df_dados = df_dados.copy()
    # normalizar cand para string limpa (para merges robustos)
    df_dados["cand"] = df_dados["cand"].apply(lambda x: _strip(x))

    # construir colunas coalescidas
    def build_title(row):
        return _strip(row.get(col_titulo)) or coalesce_row(row, alt_titulo_cols)

    def build_summary(row):
        return _strip(row.get(col_resumo)) or coalesce_row(row, alt_resumo_cols)

    df_dados["__TITULO__"] = df_dados.apply(build_title, axis=1)
    df_dados["__RESUMO__"] = df_dados.apply(build_summary, axis=1)

    # válidas se houver pelo menos título OU resumo
    mask_validos = (df_dados["__TITULO__"] != "") | (df_dados["__RESUMO__"] != "")
    df_dados_validos = df_dados[mask_validos].copy()

    if df_dados_validos.empty:
        st.error(
            "🚫 Após coalescer colunas, todas as linhas continuam sem Título e/ou Resumo.\n"
            "➡️ Ajusta as colunas principais e as de fallback acima."
        )
        st.stop()

    # preparar manuais (opcional)
    if not df_class.empty:
        df_class = df_class.copy()
        df_class["cand"] = df_class["cand"].apply(lambda x: _strip(x))
        col_manual = st.selectbox("✅ Coluna das classificações manuais:", [c for c in df_class.columns if c != "cand"] or ["(Nenhuma)"])
        if col_manual == "(Nenhuma)":
            df_class = pd.DataFrame(columns=["cand", "Classificação Manual"])
        else:
            df_class = df_class.groupby("cand").agg({
                col_manual: lambda x: "; ".join(sorted(set(_strip(v) for v in x if _strip(v))))
            }).rename(columns={col_manual: "Classificação Manual"}).reset_index()
    else:
        col_manual = "(Nenhuma)"

    # merge (se houver manuais)
    if not df_class.empty:
        df_final = df_dados_validos.merge(df_class, on="cand", how="inner")
        tem_intersecao = not df_final.empty
    else:
        df_final = df_dados_validos.copy()
        df_final["Classificação Manual"] = ""
        tem_intersecao = True

    # ---- diagnóstico
    st.info(
        "🧾 Contagens | "
        f"Linhas sheet dados: {len(df_dados)} | "
        f"Com título+resumo (após coalesce): {len(df_dados_validos)} | "
        f"Linhas sheet manuais: {len(df_class) if col_manual != '(Nenhuma)' else 0} | "
        f"Interseção cands: {'N/A' if col_manual=='(Nenhuma)' else len(set(df_dados_validos['cand']).intersection(set(df_class['cand'])))} | "
        f"Linhas após merge: {len(df_final)}"
    )

    if not tem_intersecao:
        st.warning(
            "Não há interseção de 'cand' entre dados e manuais. "
            "Vou **prosseguir sem classificações manuais** para não bloquear o processo."
        )
        df_final = df_dados_validos.copy()
        df_final["Classificação Manual"] = ""

    # Quantidade
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
        help="Se ligado, as percentagens são calculadas por similaridade coseno entre o texto do projeto e as descrições dos domínios."
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
                titulo = _strip(row["__TITULO__"])
                resumo = _strip(row["__RESUMO__"])

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
            st.error(
                "🚫 Nada classificado.\n"
                "• Verifica as colunas principais e de fallback para Título/Resumo.\n"
                "• Testa o Azure no painel de diagnóstico."
            )
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
