# ExpertsAllocator.py
import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from openai import AzureOpenAI

# -----------------------------
# Azure OpenAI
# -----------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMB_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# -----------------------------
# Helpers
# -----------------------------
def _clean(s):
    if not isinstance(s, str): s = "" if pd.isna(s) else str(s)
    return re.sub(r"\s+", " ", s.strip())

def _cos(a, b):
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def _email_domain(email):
    email = (email or "").strip().lower()
    if "@" in email:
        return email.split("@", 1)[1]
    return ""

# -----------------------------
# Embeddings (cache)
# -----------------------------
@st.cache_data(show_spinner=False)
def embed_text_cached(text: str):
    try:
        resp = client.embeddings.create(model=EMB_DEPLOYMENT, input=text)
        return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        st.warning(f"Falha embeddings: {e}")
        return None

def embed_many(texts, dim_fallback=1536):
    vecs = []
    for t in texts:
        v = embed_text_cached(t)
        vecs.append(v if v is not None else np.zeros(dim_fallback))
    return vecs

# -----------------------------
# Enriquecimento minimal (quando faltar interesses)
# -----------------------------
def enrich_openalex(name, org=None, max_concepts=10):
    try:
        url = "https://api.openalex.org/authors"
        params = {"search": name, "per_page": 5}
        if org:
            params["filter"] = f"last_known_institution.display_name.search:{org}"
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: 
            return ""
        results = r.json().get("results", [])
        if not results:
            return ""
        aid = results[0].get("id")
        if not aid:
            return ""
        w = requests.get("https://api.openalex.org/works",
                         params={"filter": f"authorships.author.id:{aid}", "per_page": 25},
                         timeout=8)
        if w.status_code != 200:
            return ""
        freq = {}
        for it in w.json().get("results", []):
            for c in (it.get("concepts") or []):
                kw = c.get("display_name")
                if kw:
                    freq[kw] = freq.get(kw, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
        return "; ".join([k for k, _ in top])
    except Exception:
        return ""

def build_expert_profile(row, use_web=False):
    nome = _clean(row.get("Nome", ""))
    org  = _clean(row.get("Organização", "") or row.get("Organizacao", ""))
    interesses = _clean(row.get("Interesses de pesquisa", "") or row.get("Interesses", ""))
    extra = ""
    if use_web and not interesses:
        extra = enrich_openalex(nome, org)
    perfil = "; ".join([p for p in [interesses, extra] if p]) or org or nome
    return perfil

def compress_keywords_with_llm(texto):
    try:
        prompt = f"Extrai 10-15 keywords/tópicos do texto (separa por ';' e sem frases longas):\n\n{texto}"
        r = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        out = r.choices[0].message.content.strip()
        out = re.sub(r"\s+", " ", out).strip(" -•")
        return out
    except Exception:
        return texto

# -----------------------------
# Conflito de interesses
# -----------------------------
def has_conflict(project_row, expert_row, proj_org_col="Nome do Beneficiário/Entidade", proj_text_cols=("Nome do Projeto","Resumo")):
    """
    Regra conservadora:
    - Se o Nome/Organização do perito aparecem no 'Nome' do projeto (empresa/entidade) ou no Resumo, é conflito.
    - Também bloqueia se o domínio do email do perito coincide com um domínio listado no projeto (se houver coluna).
    Ajusta os nomes de colunas conforme os teus ficheiros.
    """
    perito_nome = _clean(expert_row.get("Nome", ""))
    perito_org  = _clean(expert_row.get("Organização", ""))
    perito_dom  = _email_domain(expert_row.get("Email", ""))

    campos_proj = []
    # Nome da entidade do projeto
    if proj_org_col in project_row:
        campos_proj.append(_clean(project_row.get(proj_org_col, "")))
    # Nome do projeto + Resumo (ou equivalentes)
    for c in proj_text_cols:
        if c in project_row:
            campos_proj.append(_clean(project_row.get(c, "")))
    blob = " ".join(campos_proj).lower()

    if perito_nome and perito_nome.lower() in blob:
        return True
    if perito_org and perito_org.lower() in blob:
        return True

    # se houver coluna 'Domínio' do projeto (não tens — deixa neutro)
    proj_domain = ""
    if proj_domain and perito_dom and perito_dom == proj_domain.lower():
        return True
    return False

# -----------------------------
# Alocação: top-K por projeto + capacidade por perito
# -----------------------------
def allocate_topk(score_matrix, max_per_expert, K, conflicts_mask=None):
    """
    score_matrix: PxE (P projetos, E peritos)
    max_per_expert: capacidade inteira por perito (ou array Ex1)
    K: nº de peritos por projeto (top-K válidos), respeitando capacidades
    conflicts_mask: PxE boolean (True = CONFLITO → impossível)
    Retorna: lista de listas, cada projeto com até K peritos [indices]
    """
    P, E = score_matrix.shape
    if isinstance(max_per_expert, int):
        cap = np.full(E, max_per_expert, dtype=int)
    else:
        cap = np.array(max_per_expert, dtype=int)

    # aplica conflito: zera score
    S = score_matrix.copy()
    if conflicts_mask is not None:
        S[conflicts_mask] = -1e9  # inviável

    # resultado
    assigned = [[] for _ in range(P)]

    # vamos iterar enquanto houver projetos com vagas e peritos disponíveis
    # greedy: escolhe sempre o maior score disponível globalmente (p, e)
    while True:
        # computar score máximo restante
        best_score = -1e9
        best_p, best_e = -1, -1
        for p in range(P):
            if len(assigned[p]) >= K:
                continue
            # melhores e para p (respeitando cap>0)
            for e in range(E):
                if cap[e] <= 0: 
                    continue
                if S[p, e] > best_score:
                    best_score, best_p, best_e = S[p, e], p, e

        if best_score <= -1e8:  # sem pares válidos
            break

        # atribui
        assigned[best_p].append(best_e)
        cap[best_e] -= 1

        # para não voltar a escolher o mesmo par, marca a célula como -inf
        S[best_p, best_e] = -1e9

        # se o projeto já tem K, passamos; o ciclo continua até não haver mais pares válidos
    return assigned

# -----------------------------
# UI
# -----------------------------
def run():
    st.markdown("### 👥 Alocação de Peritos a Projetos")

    st.write("Carrega os dois ficheiros necessários:")
    c1, c2 = st.columns(2)
    with c1:
        f_proj = st.file_uploader("📂 Lista de Projetos (.xlsx)", type=["xlsx"], key="proj_up")
    with c2:
        f_exp  = st.file_uploader("📂 Listagem de Peritos (.xlsx)", type=["xlsx"], key="exp_up")

    if not f_proj or not f_exp:
        st.info("Falta carregar os dois ficheiros.")
        return

    # Ler
    dfp = pd.read_excel(f_proj)   # projetos
    dfe = pd.read_excel(f_exp)    # peritos

    # Mapeamento de colunas
    st.subheader("Mapeamento de colunas")
    cols_p = dfp.columns.tolist()
    cols_e = dfe.columns.tolist()

    # Projetos (ajusta índices conforme o teu ficheiro)
    col_idproj   = st.selectbox("Coluna Nº Projeto", cols_p, index=0)
    col_org_proj = st.selectbox("Coluna 'Nome' (entidade/beneficiário do projeto)", cols_p, index=1 if len(cols_p)>1 else 0)
    col_nomeproj = st.selectbox("Coluna Nome do Projeto (se existir)", ["(Nenhuma)"] + cols_p, index=0)
    col_resumo   = st.selectbox("Coluna Resumo do Projeto", cols_p, index=min(3, len(cols_p)-1))

    # Peritos
    col_nomeexp   = st.selectbox("Coluna Nome do Perito", cols_e, index=0)
    col_orgexp    = st.selectbox("Coluna Organização do Perito", cols_e, index=1 if len(cols_e)>1 else 0)
    col_emailexp  = st.selectbox("Coluna Email do Perito", cols_e, index=2 if len(cols_e)>2 else 0)
    col_interesse = st.selectbox("Coluna Interesses de pesquisa (opcional)", ["Nenhuma"] + cols_e,
                                 index=(cols_e.index("Interesses de pesquisa")+1) if "Interesses de pesquisa" in cols_e else 0)
    col_capacidade= st.selectbox("Coluna Capacidade por perito (opcional)", ["Nenhuma"] + cols_e, index=0)

    st.divider()
    st.subheader("Âmbito da alocação")
    modo_proj = st.radio("Quais projetos queres alocar?", ["Todos", "Primeiros N", "Escolher por Nº Projeto"], horizontal=True)
    if modo_proj == "Primeiros N":
        n_top = st.number_input("Nº de projetos (topo do ficheiro)", 1, 10000, 20, 1)
    else:
        n_top = None

    if modo_proj == "Escolher por Nº Projeto":
        # mostra lista e permite multi-seleção
        todos_ids = dfp[col_idproj].astype(str).tolist()
        pre_sel = todos_ids[: min(20, len(todos_ids))]
        ids_escolhidos = st.multiselect("Seleciona Nº Projeto", options=todos_ids, default=pre_sel)
    else:
        ids_escolhidos = None

    st.subheader("Regras de alocação")
    K = st.number_input("Top-N peritos por projeto", min_value=1, max_value=10, value=1, step=1)
    cap_default = st.number_input("Capacidade por perito (se não houver coluna específica)", min_value=1, max_value=100, value=3, step=1)
    usar_enriquecimento = st.checkbox("Enriquecer peritos via web quando faltam interesses (OpenAlex)", value=True)
    comprimir_llm = st.checkbox("Sintetizar perfil do perito com LLM (keywords curtas)", value=True)

    if st.button("🚀 Alocar", use_container_width=True):
        with st.spinner("A preparar dados, calcular embeddings e alocar..."):
            # ------- preparar projetos
            dfp_use = dfp.copy()
            dfp_use["Nº Projeto"] = dfp_use[col_idproj].apply(_clean)
            dfp_use["Entidade"]   = dfp_use[col_org_proj].apply(_clean)

            if col_nomeproj != "(Nenhuma)":
                dfp_use["Nome do Projeto"] = dfp_use[col_nomeproj].apply(_clean)
            else:
                dfp_use["Nome do Projeto"] = ""

            dfp_use["Resumo"] = dfp_use[col_resumo].apply(_clean)

            # subset conforme modo
            if modo_proj == "Primeiros N":
                dfp_use = dfp_use.head(int(n_top))
            elif modo_proj == "Escolher por Nº Projeto":
                ids_set = set(ids_escolhidos or [])
                dfp_use = dfp_use[dfp_use["Nº Projeto"].astype(str).isin(ids_set)]

            # texto do projeto (usar nome do projeto se existir, senão entidade)
            dfp_use["texto_proj"] = (
                (dfp_use["Nome do Projeto"].replace("", np.nan).fillna(dfp_use["Entidade"])) + ". " +
                dfp_use["Resumo"].fillna("")
            ).str.strip()

            # ------- preparar peritos
            dfe_use = dfe.copy()
            dfe_use["Nome"]         = dfe_use[col_nomeexp].apply(_clean)
            dfe_use["Organização"]  = dfe_use[col_orgexp].apply(_clean) if col_orgexp in dfe_use else ""
            dfe_use["Email"]        = dfe_use[col_emailexp].apply(_clean) if col_emailexp in dfe_use else ""
            dfe_use["Interesses de pesquisa"] = dfe_use[col_interesse].apply(_clean) if col_interesse!="Nenhuma" else ""

            # capacidade
            if col_capacidade != "Nenhuma":
                caps = pd.to_numeric(dfe_use[col_capacidade], errors="coerce").fillna(cap_default).astype(int).clip(lower=1)
            else:
                caps = pd.Series([cap_default]*len(dfe_use), index=dfe_use.index, dtype=int)

            # perfis (com enriquecimento opcional)
            perfis = []
            for _, row in dfe_use.iterrows():
                perfil = build_expert_profile(row, use_web=usar_enriquecimento)
                if comprimir_llm and perfil:
                    perfil = compress_keywords_with_llm(perfil)
                perfis.append(perfil)

            # ------- embeddings
            proj_vecs = embed_many(dfp_use["texto_proj"].tolist())
            exp_vecs  = embed_many(perfis)

            P, E = len(proj_vecs), len(exp_vecs)
            S = np.zeros((P, E), dtype=float)
            for i in range(P):
                for j in range(E):
                    if proj_vecs[i] is None or exp_vecs[j] is None:
                        S[i, j] = -1e9
                    else:
                        S[i, j] = _cos(proj_vecs[i], exp_vecs[j])

            # ------- conflitos
            conflicts = np.zeros((P, E), dtype=bool)
            for i in range(P):
                prow = dfp_use.iloc[i]
                for j in range(E):
                    erow = dfe_use.iloc[j]
                    # conflito se perito estiver envolvido: nome/org no nome/resumo/entidade
                    # (ajusta proj_org_col e proj_text_cols se mudares nomes)
                    # Aqui passamos explicitamente os campos que já normalizámos.
                    if has_conflict(
                        {"Nome do Beneficiário/Entidade": prow["Entidade"],
                         "Nome do Projeto": prow["Nome do Projeto"],
                         "Resumo": prow["Resumo"]},
                        {"Nome": erow["Nome"], "Organização": erow["Organização"], "Email": erow["Email"]}
                    ):
                        conflicts[i, j] = True
                        S[i, j] = -1e9  # inviável

            # ------- alocação (top-K por projeto + capacidade por perito)
            assigned_idx = allocate_topk(S, max_per_expert=caps.values, K=int(K), conflicts_mask=conflicts)

            # ------- saída minimal
            rows = []
            for i, experts in enumerate(assigned_idx):
                peritos = [dfe_use.iloc[e]["Nome"] for e in experts] if experts else []
                rows.append({
                    "Nº Projeto": dfp_use.iloc[i]["Nº Projeto"],
                    "Nome do Projeto": dfp_use.iloc[i]["Nome do Projeto"] or dfp_use.iloc[i]["Entidade"],
                    "Peritos": ", ".join(peritos) if peritos else "—"
                })
            out = pd.DataFrame(rows)
            st.success("✅ Alocação concluída!")
            st.dataframe(out, use_container_width=True)

            # download
            buf = BytesIO()
            out.to_excel(buf, index=False)
            st.download_button(
                "📥 Download (.xlsx)",
                data=buf.getvalue(),
                file_name="alocacao_peritos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Execução direta opcional
if __name__ == "__main__":
    run()
