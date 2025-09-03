# ExpertsAllocator.py
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from openai import AzureOpenAI

# =============================
# Azure OpenAI
# =============================
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMB_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# =============================
# Helpers
# =============================
def _clean(s):
    if not isinstance(s, str): s = "" if pd.isna(s) else str(s)
    return re.sub(r"\s+", " ", s.strip())

def _cos(a, b):
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

@st.cache_data(show_spinner=False)
def embed_text_cached(text: str):
    if not EMB_DEPLOYMENT:
        return None
    try:
        r = client.embeddings.create(model=EMB_DEPLOYMENT, input=text)
        return np.array(r.data[0].embedding, dtype=float)
    except Exception:
        return None

def embed_many(texts, dim_fallback=1536):
    vecs = []
    for t in texts:
        v = embed_text_cached(t)
        vecs.append(v if v is not None else np.zeros(dim_fallback))
    return vecs

def has_conflict(project_row, expert_row):
    """Conflito se nome/organização do perito apareceu em Nome ou Resumo do projeto."""
    perito_nome = _clean(expert_row.get("Nome", ""))
    perito_org  = _clean(expert_row.get("Organização", ""))
    blob = (_clean(project_row.get("Nome", "")) + " " + _clean(project_row.get("Resumo", ""))).lower()
    if perito_nome and perito_nome.lower() in blob: return True
    if perito_org  and perito_org.lower()  in blob: return True
    return False

def llm_rerank(project_name, project_summary, candidates, k_min=3, k_max=5, debug=False):
    """candidates = [{'Nome','Interesses','Organização'}] -> lista ordenada de nomes (3..5)."""
    if not CHAT_DEPLOYMENT:
        return [c["Nome"] for c in candidates[:k_min]]

    blocos = []
    for i, c in enumerate(candidates, 1):
        org = f" | Org: {c.get('Organização','')}" if c.get("Organização") else ""
        blocos.append(f"{i}. {c['Nome']} | Interesses: {c['Interesses']}{org}")
    candidatos_txt = "\n".join(blocos)

    prompt = f"""
És um avaliador. Recebes a descrição de um projeto e uma lista de peritos com os respetivos interesses.
Seleciona entre {k_min} e {k_max} peritos (idealmente {min(max(k_min,3),k_max)}) por ORDEM de melhor correspondência ao projeto.

Projeto:
- Nome (título): {project_name or "(sem nome)"}
- Resumo (usa principalmente este campo): {project_summary}

Peritos (nome e interesses):
{candidatos_txt}

Responde **apenas** com os nomes escolhidos, um por linha, sem numeração ou explicações.
""".strip()

    try:
        r = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = (r.choices[0].message.content or "").strip()
        linhas = [l.strip().strip("-•1234567890. ") for l in raw.splitlines() if l.strip()]
        nomes_validos = {c["Nome"]: True for c in candidates}
        out = []
        for l in linhas:
            if l in nomes_validos:
                out.append(l)
            else:
                # match case-insensitive
                for nome in nomes_validos.keys():
                    if l.lower() == nome.lower():
                        out.append(nome)
                        break
        if len(out) < k_min:
            out = [c["Nome"] for c in candidates[:k_min]]
        if len(out) > k_max:
            out = out[:k_max]

        if debug:
            with st.expander("🛠️ Debug LLM (prompt + resposta)"):
                st.code(prompt, language="markdown")
                st.text(raw)

        return out
    except Exception as e:
        if debug:
            st.warning(f"Falha LLM (re-ranking): {e}")
        return [c["Nome"] for c in candidates[:k_min]]

# =============================
# UI
# =============================
def run():
    st.markdown("### 👥 Alocação de Peritos a Projetos (Resumo → Interesses)")

    with st.expander("⚙️ Azure/OpenAI"):
        st.write(f"Chat deployment: {CHAT_DEPLOYMENT or '—'} | Embeddings: {EMB_DEPLOYMENT or '—'}")

    c1, c2 = st.columns(2)
    with c1:
        f_proj = st.file_uploader("📂 Lista de Projetos (.xlsx)", type=["xlsx"], key="proj_up")
    with c2:
        f_exp  = st.file_uploader("📂 Listagem de Peritos (.xlsx)", type=["xlsx"], key="exp_up")

    if not f_proj or not f_exp:
        st.info("Carrega ambos os ficheiros.")
        return

    try:
        dfp = pd.read_excel(f_proj)   # projetos
        dfe = pd.read_excel(f_exp)    # peritos
    except Exception as e:
        st.error(f"Erro a ler ficheiros: {e}")
        return

    # Mapeamento (defaults para os teus nomes)
    cols_p = dfp.columns.tolist()
    cols_e = dfe.columns.tolist()

    col_idproj = st.selectbox("Coluna Nº Projecto", cols_p, index=cols_p.index("Nº Projecto") if "Nº Projecto" in cols_p else 0)
    col_nome   = st.selectbox("Coluna Nome (entidade / título do projeto)", cols_p, index=cols_p.index("Nome") if "Nome" in cols_p else min(1, len(cols_p)-1))
    col_resumo = st.selectbox("Coluna Resumo", cols_p, index=cols_p.index("Resumo") if "Resumo" in cols_p else min(2, len(cols_p)-1))

    col_nomeexp   = st.selectbox("Coluna Nome do Perito", cols_e, index=cols_e.index("Nome") if "Nome" in cols_e else 0)
    col_orgexp    = st.selectbox("Coluna Organização (perito)", cols_e, index=cols_e.index("Organização") if "Organização" in cols_e else (cols_e.index("Organizacao") if "Organizacao" in cols_e else min(1, len(cols_e)-1)))
    col_interesse = st.selectbox("Coluna Interesses de pesquisa (obrigatória)", cols_e, index=cols_e.index("Interesses de pesquisa") if "Interesses de pesquisa" in cols_e else 0)

    st.divider()
    st.subheader("Âmbito")
    modo = st.radio("Projetos a alocar", ["Todos", "Primeiros N", "Escolher por Nº Projecto"], horizontal=True)
    if modo == "Primeiros N":
        n_top = st.number_input("Nº de projetos (topo do ficheiro)", 1, 10000, 20, 1)
    else:
        n_top = None
    if modo == "Escolher por Nº Projecto":
        lista_ids = dfp[col_idproj].astype(str).tolist()
        ids_escolhidos = st.multiselect("Seleciona Nº Projecto", options=lista_ids, default=lista_ids[: min(20, len(lista_ids))])
    else:
        ids_escolhidos = None

    st.subheader("Parâmetros")
    k_min = st.number_input("Nº mínimo de peritos", 3, 5, 3, 1)
    k_max = st.number_input("Nº máximo de peritos", 3, 5, 5, 1)
    topN = st.slider("Pré-selecionar top-N por embeddings antes do LLM", 5, 30, 12, 1)
    debug = st.checkbox("🛠️ Mostrar prompt/resposta do LLM", value=False)

    if st.button("🚀 Alocar", use_container_width=True):
        with st.spinner("A calcular correspondências e alocar..."):
            # -------- projetos
            dfp_use = dfp.copy()
            dfp_use["Nº Projecto"] = dfp_use[col_idproj].apply(_clean)
            dfp_use["Nome"]        = dfp_use[col_nome].apply(_clean)
            dfp_use["Resumo"]      = dfp_use[col_resumo].apply(_clean)

            if modo == "Primeiros N":
                dfp_use = dfp_use.head(int(n_top))
            elif modo == "Escolher por Nº Projecto":
                dfp_use = dfp_use[dfp_use["Nº Projecto"].astype(str).isin(set(ids_escolhidos or []))]

            # texto a comparar (Resumo principalmente; Nome só complementa)
            dfp_use["texto_proj"] = (dfp_use["Resumo"].replace("", np.nan).fillna("") + " || " +
                                     dfp_use["Nome"].replace("", np.nan).fillna("")).str.strip()

            # -------- peritos (somente com interesses)
            dfe_use = dfe.copy()
            dfe_use["Nome"]         = dfe_use[col_nomeexp].apply(_clean)
            dfe_use["Organização"]  = dfe_use[col_orgexp].apply(_clean) if col_orgexp in dfe_use else ""
            dfe_use["Interesses"]   = dfe_use[col_interesse].apply(_clean)

            antes = len(dfe_use)
            dfe_use = dfe_use[dfe_use["Interesses"].str.len() > 0].reset_index(drop=True)
            if len(dfe_use) == 0:
                st.error("Não há peritos com 'Interesses de pesquisa' preenchidos.")
                return
            remov = antes - len(dfe_use)
            if remov > 0:
                st.info(f"🔎 {remov} peritos removidos por não terem 'Interesses de pesquisa'.")

            # -------- embeddings (pré-ranking rápido)
            proj_vecs = embed_many(dfp_use["texto_proj"].tolist())
            exp_vecs  = embed_many((dfe_use["Interesses"] + ("; " + dfe_use["Organização"]).where(dfe_use["Organização"]!="", "")).tolist())

            P, E = len(proj_vecs), len(exp_vecs)
            if P == 0 or E == 0:
                st.error("Sem projetos ou peritos após filtragem.")
                return

            S = np.zeros((P, E), dtype=float)
            for i in range(P):
                for j in range(E):
                    if proj_vecs[i] is None or exp_vecs[j] is None:
                        S[i, j] = -1e9
                    else:
                        S[i, j] = _cos(proj_vecs[i], exp_vecs[j])

            # remover conflitos
            for i in range(P):
                prow = dfp_use.iloc[i]
                for j in range(E):
                    erow = dfe_use.iloc[j]
                    if has_conflict({"Nome": prow["Nome"], "Resumo": prow["Resumo"]},
                                    {"Nome": erow["Nome"], "Organização": erow["Organização"]}):
                        S[i, j] = -1e9

            # -------- por projeto: topN por embeddings -> LLM re-ranking -> 3..5
            linhas = []
            for i in range(P):
                # candidatos por score desc e válidos
                idx_sorted = np.argsort(-S[i])
                idx_sorted = [ix for ix in idx_sorted if S[i, ix] > -1e8]
                if not idx_sorted:
                    linhas.append({
                        "Nº Projecto": dfp_use.iloc[i]["Nº Projecto"],
                        "Nome do Projeto": dfp_use.iloc[i]["Nome"],
                        "Atribuição": "—"
                    })
                    continue

                idx_top = idx_sorted[:topN]
                candidatos = [{
                    "Nome": dfe_use.iloc[j]["Nome"],
                    "Interesses": dfe_use.iloc[j]["Interesses"],
                    "Organização": dfe_use.iloc[j]["Organização"]
                } for j in idx_top]

                nomes_final = llm_rerank(
                    project_name=dfp_use.iloc[i]["Nome"],
                    project_summary=dfp_use.iloc[i]["Resumo"],
                    candidates=candidatos,
                    k_min=int(k_min),
                    k_max=int(k_max),
                    debug=debug
                )

                linhas.append({
                    "Nº Projecto": dfp_use.iloc[i]["Nº Projecto"],
                    "Nome do Projeto": dfp_use.iloc[i]["Nome"],
                    "Atribuição": ", ".join(nomes_final) if nomes_final else "—"
                })

            out = pd.DataFrame(linhas)
            st.success("✅ Alocação concluída!")
            st.dataframe(out, use_container_width=True)

            buf = BytesIO()
            out.to_excel(buf, index=False)
            st.download_button(
                "📥 Download (.xlsx)",
                data=buf.getvalue(),
                file_name="alocacao_peritos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    run()
