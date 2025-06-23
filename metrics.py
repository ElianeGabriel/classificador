import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from io import BytesIO

st.markdown("### 📊 Avaliação das Classificações com LLM")

# Upload do ficheiro classificado com LLM
teste_file = st.file_uploader("📁 Upload do ficheiro com classificação LLM:", type=["xlsx"], key="avaliacao")

if teste_file:
    df = pd.read_excel(teste_file)
    colunas = df.columns.tolist()

    col_manual = st.selectbox("✅ Coluna com a classificação manual:", colunas, index=colunas.index("Dominio ENEI") if "Dominio ENEI" in colunas else 0)
    col_llm = st.selectbox("🤖 Coluna com a classificação LLM:", colunas, index=colunas.index("Domínio LLM") if "Domínio LLM" in colunas else 0)

    df = df.dropna(subset=[col_manual, col_llm])
    df = df[(df[col_llm] != "Indefinido") & (df[col_manual] != "Indefinido")]

    y_true = df[col_manual].astype(str).str.strip()
    y_pred = df[col_llm].astype(str).str.strip()

    # Relatório de classificação
    st.markdown("#### 📋 Relatório de Métricas")
    relatorio = classification_report(y_true, y_pred, output_dict=True)
    relatorio_df = pd.DataFrame(relatorio).transpose()
    st.dataframe(relatorio_df.style.format("{:.2f}"))

    # Matriz de confusão
    st.markdown("#### 🔀 Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(10, 6))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y_true.unique()), yticklabels=sorted(y_true.unique()), ax=ax)
    plt.xlabel("LLM")
    plt.ylabel("Manual")
    st.pyplot(fig)

    # Gráfico de distribuição de domínios
    st.markdown("#### 📊 Distribuição das Classificações")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df[col_manual].value_counts())
        st.caption("Classificação manual")
    with col2:
        st.bar_chart(df[col_llm].value_counts())
        st.caption("Classificação LLM")

    # Exportar métricas
    st.markdown("#### 💾 Download do Relatório")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Classificacoes")
        relatorio_df.to_excel(writer, sheet_name="Metricas")
    st.download_button("📥 Download Excel com resultados", data=buffer.getvalue(), file_name="avaliacao_classificacao_llm.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")