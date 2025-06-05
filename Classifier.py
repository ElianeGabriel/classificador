import streamlit as st
import pandas as pd
import random
from io import BytesIO

# 🎨 Configuração de página
st.set_page_config(page_title="Classificador de Projetos I&D", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f9f9f9; padding: 2rem; }
        .block-container { padding-top: 1rem; }
        .stButton > button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🔎 Classificador de Projetos I&D")
st.markdown("Classifique projetos com base no **Sumário Executivo** e nos **Domínios Prioritários** definidos.")

# 📤 Upload de ficheiro
uploaded_file = st.file_uploader("Carregue o ficheiro Excel com os projetos e os domínios prioritários:", type=["xlsx"])

if uploaded_file:
    # 📚 Ler ficheiro Excel
    xls = pd.ExcelFile(uploaded_file)
    try:
        projetos_df = xls.parse("Projetos")
        dominios_df = xls.parse("Dominios")
    except Exception as e:
        st.error(f"❌ Erro ao ler as folhas 'Projetos' ou 'Dominios': {e}")
        st.stop()

    if "Sumario Executivo" not in projetos_df.columns:
        st.error("❗ A folha 'Projetos' precisa da coluna 'Sumario Executivo'.")
        st.stop()

    # 🔄 Processar os domínios
    domain_options = []
    current_domain = None
    current_description = []

    for _, row in dominios_df.iterrows():
        domain = row["Dominios"]
        description = str(row["Descrição"]).strip() if pd.notna(row.get("Descrição")) else ""

        if pd.notna(domain) and domain.strip():
            if current_domain:
                full_description = " ".join(current_description).strip()
                domain_options.append(f"{current_domain} - {full_description}")
            current_domain = domain.strip()
            current_description = [description] if description else []
        else:
            if description:
                current_description.append(description)

    if current_domain:
        full_description = " ".join(current_description).strip()
        domain_options.append(f"{current_domain} - {full_description}")

    # 🤖 Função simulada de classificação
    def classify_project(summary):
        possible_domains = [opt.split(" - ")[0] for opt in domain_options]
        return random.choice(possible_domains)

    # 🚀 Botão de classificação
    if st.button("🚀 Classificar Projetos"):
        with st.spinner("A classificar projetos..."):
            results = []
            for idx, row in projetos_df.iterrows():
                summary = row["Sumario Executivo"]
                domain = classify_project(summary)
                results.append({
                    "ID": idx + 1,
                    "Sumario Executivo": summary,
                    "Domínio Classificado": domain
                })

            result_df = pd.DataFrame(results)

            st.success("✅ Classificação concluída!")
            st.dataframe(result_df, use_container_width=True)

            # 📁 Preparar ficheiro para download
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                projetos_df.to_excel(writer, index=False, sheet_name='Projetos')
                dominios_df.to_excel(writer, index=False, sheet_name='Dominios')
                result_df.to_excel(writer, index=False, sheet_name='Classificação')

            st.download_button(
                label="📥 Descarregar Excel com classificação",
                data=output.getvalue(),
                file_name="classificacao_projetos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
