import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from io import BytesIO

st.markdown("### 📊 Avaliação das Classificações com LLM")

# Tenta carregar os dados da sessão, se existirem
if "classificacoes_llm" in st.session_state:
    df = st.session_state.classificacoes_llm.copy()
    st.success("Usar resultados da classificação recente!")
    usar_upload = st.radio("Desejas carregar outro ficheiro?", ["Não", "Sim"])
else:
    usar_upload = "Sim"

# Se o utilizador quiser fazer upload manual
if usar_upload == "Sim":
    teste_file = st.file_uploader("📁 Upload do ficheiro com classificação LLM:", type=["xlsx"], key="avaliacao")
    if teste_file:
        df = pd.read_excel(teste_file)

if 'df' in locals():
    colunas = df.columns.tolist()

    col_manual_1 = st.selectbox("✅ Coluna com classificação manual principal:", colunas, index=colunas.index("Dominio ENEI") if "Dominio ENEI" in colunas else 0)
    col_manual_2 = st.selectbox("✅ Coluna com classificação manual alternativa (opcional):", ["Nenhuma"] + colunas, index=colunas.index("Dominio ENEI Projecto") + 1 if "Dominio ENEI Projecto" in colunas else 0)
    col_llm = st.selectbox("🤖 Coluna com a classificação LLM:", colunas, index=colunas.index("Domínio LLM") if "Domínio LLM" in colunas else 0)

    # Prepara os valores
    y_true = df[col_manual_1].astype(str).str.strip()
    if col_manual_2 != "Nenhuma":
        y_true_alt = df[col_manual_2].astype(str).str.strip()
    else:
        y_true_alt = pd.Series(["__none__"] * len(y_true))

    y_pred = df[col_llm].astype(str).str.strip()

    # Filtrar "Indefinido"
    filtro_validos = (y_pred != "Indefinido") & (y_true != "Indefinido")
    y_true = y_true[filtro_validos].reset_index(drop=True)
    y_true_alt = y_true_alt[filtro_validos].reset_index(drop=True)
    y_pred = y_pred[filtro_validos].reset_index(drop=True)

    # Avaliar se algum dos dois matches manuais bate com o LLM
    match_principal = y_true == y_pred
    match_alternativo = y_true_alt == y_pred
    acerto_total = match_principal | match_alternativo

    st.markdown("#### 📋 Métricas de Avaliação")
    acc_total = acerto_total.mean()
    acc_principal = match_principal.mean()
    acc_alternativo = match_alternativo[~match_principal].mean()

    st.write(f"✅ **Acurácia geral (principal ou alternativa):** {acc_total:.2%}")
    st.write(f"🎯 **Acurácia apenas na classificação principal:** {acc_principal:.2%}")
    if col_manual_2 != "Nenhuma":
        st.write(f"🧭 **Acurácia apenas na alternativa (sem acerto na principal):** {acc_alternativo:.2%}")

    # Relatório detalhado
    relatorio = classification_report(y_true, y_pred, output_dict=True)
    relatorio_df = pd.DataFrame(relatorio).transpose()
    st.dataframe(relatorio_df.style.format("{:.2f}"))

    # Matriz de confusão
    st.markdown("#### 🔀 Matriz de Confusão")
    etiquetas = sorted(set(y_true.unique()) | set(y_pred.unique()))
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=etiquetas, yticklabels=etiquetas, ax=ax)
    plt.xlabel("LLM")
    plt.ylabel("Manual")
    st.pyplot(fig)

    # Gráficos de distribuição
    st.markdown("#### 📊 Distribuição das Classificações")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(y_true.value_counts())
        st.caption("Distribuição da classificação manual principal")
    with col2:
        st.bar_chart(y_pred.value_counts())
        st.caption("Distribuição da classificação LLM")

    # Exportar para Excel
    st.markdown("#### 💾 Download do Relatório")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_final = df.copy()
        df_final["Acerto (principal)"] = match_principal
        df_final["Acerto (alternativa)"] = match_alternativo
        df_final["Acerto geral"] = acerto_total
        df_final.to_excel(writer, index=False, sheet_name="Classificacoes")
        relatorio_df.to_excel(writer, sheet_name="Metricas")
    st.download_button("📥 Download do Excel com resultados", data=buffer.getvalue(), file_name="avaliacao_classificacao_llm.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
