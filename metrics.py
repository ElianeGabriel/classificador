import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from io import BytesIO
import re

# ------------------- Helpers -------------------
def _possible_llm_cols(cols):
    prefer = ["Domínios LLM", "Domínio LLM", "Dominios LLM", "Dominios", "LLM"]
    for p in prefer:
        if p in cols:
            return p
    # fallback: tenta algo que contenha 'llm'
    for c in cols:
        if "llm" in c.lower():
            return c
    return cols[0] if cols else None

def _parse_llm_output(val: str):
    """
    Converte saída do LLM em lista ordenada de domínios.
    Exemplos aceites:
      "A, B"
      "A; B"
      "1. A (60%), 2. B (40%)"
      "A (62%), B (38%)"
      "Indefinido"
    """
    if not isinstance(val, str):
        return []
    s = val.strip()
    if s.lower() == "indefinido":
        return []
    # remove bullets/numeração
    s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)
    # substitui separadores por vírgula
    s = s.replace(";", ",")
    # remove percentagens
    s = re.sub(r"\(\s*\d+%?\s*\)", "", s)
    # quebra e limpa
    parts = [p.strip(" -•:") for p in s.split(",") if p.strip(" -•:")]
    # remove vazios e dupes mantendo ordem
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    # máximo 2
    return out[:2]

def _normalize_labels(series: pd.Series):
    """Trima espaços e uniformiza pequenos detalhes."""
    return series.astype(str).str.strip()

def _choose_labels(df):
    cols = df.columns.tolist()
    # heurística para manual
    manual_main = "Dominio ENEI" if "Dominio ENEI" in cols else cols[0]
    manual_alt = "Dominio ENEI Projecto" if "Dominio ENEI Projecto" in cols else None
    llm_col = _possible_llm_cols(cols)
    return manual_main, manual_alt, llm_col

# ------------------- Página -------------------
def run():
    st.markdown("### 📈 Avaliação das Classificações (LLM vs Manual)")

    # usar resultados em sessão se existirem
    if "classificacoes_llm" in st.session_state:
        df = st.session_state.classificacoes_llm.copy()
        st.success("A usar os resultados da classificação recente.")
        usar_upload = st.radio("Queres carregar outro ficheiro?", ["Não", "Sim"], horizontal=True, index=0)
    else:
        usar_upload = "Sim"

    if usar_upload == "Sim":
        teste_file = st.file_uploader("📁 Upload do ficheiro com classificação LLM (.xlsx)", type=["xlsx"], key="avaliacao")
        if teste_file:
            df = pd.read_excel(teste_file)

    if 'df' not in locals():
        st.info("Carrega um ficheiro ou volta a executar a classificação para ver métricas.")
        return

    cols = df.columns.tolist()
    manual_main_guess, manual_alt_guess, llm_guess = _choose_labels(df)

    col_manual_1 = st.selectbox("✅ Coluna com classificação manual principal:", cols, index=cols.index(manual_main_guess) if manual_main_guess in cols else 0)
    col_manual_2 = st.selectbox("✅ Coluna com classificação manual alternativa (opcional):", ["Nenhuma"] + cols, index=(["Nenhuma"] + cols).index(manual_alt_guess) if manual_alt_guess in cols else 0)
    col_llm = st.selectbox("🤖 Coluna com a classificação LLM:", cols, index=cols.index(llm_guess) if llm_guess in cols else 0)

    modo_avaliacao = st.radio("Como avaliar o LLM?", ["Top‑1 (apenas 1º domínio)", "Top‑2 (acerta se qualquer dos 2 bater)"], horizontal=True)

    # Normalizações básicas
    y_true = _normalize_labels(df[col_manual_1])
    y_true_alt = _normalize_labels(df[col_manual_2]) if col_manual_2 != "Nenhuma" else pd.Series(["__none__"] * len(y_true))
    llm_raw = df[col_llm].astype(str).fillna("")

    # Extrair lista de domínios do LLM
    llm_list = llm_raw.apply(_parse_llm_output)

    # Escolha top‑1 ou top‑2 como predição final (texto)
    if modo_avaliacao.startswith("Top‑1"):
        y_pred = llm_list.apply(lambda xs: xs[0] if xs else "Indefinido")
    else:
        # Para métricas “clássicas” precisamos de 1 rótulo. Usamos o 1º para relatório/confusão,
        # mas calculamos também acerto “top‑2” à parte.
        y_pred = llm_list.apply(lambda xs: xs[0] if xs else "Indefinido")

    # Filtro: remover indefinidos
    filtro_validos = (y_pred != "Indefinido") & (y_true != "Indefinido")
    y_true = y_true[filtro_validos].reset_index(drop=True)
    y_true_alt = y_true_alt[filtro_validos].reset_index(drop=True)
    y_pred = y_pred[filtro_validos].reset_index(drop=True)
    llm_list = llm_list[filtro_validos].reset_index(drop=True)

    if len(y_true) == 0:
        st.warning("Sem linhas válidas para avaliação (tudo ‘Indefinido’ ou sem correspondência).")
        return

    # Acertos
    match_principal = y_true == y_pred
    match_alternativo = (y_true_alt == y_pred) & (y_true_alt != "__none__")

    if modo_avaliacao.startswith("Top‑2"):
        # Acerto se manual_principal ∈ top2 do LLM OU manual_alt ∈ top2
        def in_top2(man, xs): return man in xs
        top2_true = [in_top2(t, xs) for t, xs in zip(y_true.tolist(), llm_list.tolist())]
        if (y_true_alt != "__none__").any():
            top2_alt = [in_top2(t, xs) if t != "__none__" else False for t, xs in zip(y_true_alt.tolist(), llm_list.tolist())]
            acerto_total = np.array(top2_true) | np.array(top2_alt)
        else:
            acerto_total = np.array(top2_true)
        acc_label = "Acurácia Top‑2 (qualquer dos dois)"
    else:
        acerto_total = (match_principal | match_alternativo).values
        acc_label = "Acurácia geral (principal ou alternativa)"

    # Métricas principais
    st.markdown("#### 📋 Métricas")
    acc_total = acerto_total.mean()
    acc_principal = match_principal.mean()
    acc_alternativo = match_alternativo[~match_principal].mean() if (y_true_alt != "__none__").any() else 0.0

    st.write(f"✅ **{acc_label}:** {acc_total:.2%}")
    st.write(f"🎯 **Acurácia apenas na classificação principal (Top‑1):** {acc_principal:.2%}")
    if (y_true_alt != "__none__").any():
        st.write(f"🧭 **Acurácia apenas na alternativa (sem acerto na principal):** {acc_alternativo:.2%}")

    # Relatório detalhado (macro/micro)
    relatorio = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    relatorio_df = pd.DataFrame(relatorio).transpose()
    st.dataframe(relatorio_df.style.format("{:.2f}"), use_container_width=True)

    # Matriz de confusão
    st.markdown("#### 🔀 Matriz de Confusão")
    etiquetas = sorted(set(y_true.unique()) | set(y_pred.unique()))
    normalizar = st.checkbox("Normalizar por linha (%)", value=True)
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas, normalize='true' if normalizar else None)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalizar else "d", cmap="Blues",
                xticklabels=etiquetas, yticklabels=etiquetas, ax=ax)
    plt.xlabel("LLM (predito)")
    plt.ylabel("Manual (verdade)")
    st.pyplot(fig)

    # Distribuições
    st.markdown("#### 📊 Distribuição das Classificações")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(y_true.value_counts().sort_values(ascending=False))
        st.caption("Distribuição da classificação manual principal")
    with col2:
        st.bar_chart(pd.Series(y_pred).value_counts().sort_values(ascending=False))
        st.caption("Distribuição da classificação LLM (Top‑1)")

    # Exportar para Excel
    st.markdown("#### 💾 Download do Relatório")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_exp = df.copy()
        # Acrescenta colunas de avaliação
        df_exp = df_exp.loc[filtro_validos.index]  # alinhar indices anteriores
        df_exp = df_exp[filtro_validos.values]
        df_exp = df_exp.copy()
        df_exp["Manual (principal)"] = y_true
        df_exp["Manual (alternativa)"] = y_true_alt
        df_exp["LLM (Top‑1)"] = y_pred
        df_exp["Acerto (principal)"] = match_principal.values
        if (y_true_alt != "__none__").any():
            df_exp["Acerto (alternativa)"] = match_alternativo.values
        df_exp["Acerto (global)"] = acerto_total
        df_exp.to_excel(writer, index=False, sheet_name="Classificacoes")
        relatorio_df.to_excel(writer, sheet_name="Metricas")
    st.download_button(
        "📥 Download do Excel com resultados",
        data=buffer.getvalue(),
        file_name="avaliacao_classificacao_llm.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Execução direta compatível
if __name__ == "__main__":
    run()
