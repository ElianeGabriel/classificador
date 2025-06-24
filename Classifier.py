if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    st.markdown("### 📄 Seleção das folhas (sheets)")
    sheet_titulo = st.selectbox("📝 Escolhe a folha com os títulos:", xls.sheet_names, key="sheet_titulo")
    sheet_resumo = st.selectbox("📋 Escolhe a folha com os resumos:", xls.sheet_names, key="sheet_resumo")

    df_titulo = pd.read_excel(xls, sheet_name=sheet_titulo)
    df_resumo = pd.read_excel(xls, sheet_name=sheet_resumo)

    # Identificador comum
    colunas_comuns = list(set(df_titulo.columns).intersection(set(df_resumo.columns)))
    col_id = st.selectbox("🔗 Coluna para juntar as folhas (identificador comum):", colunas_comuns)

    df = pd.merge(df_titulo, df_resumo, on=col_id, how="inner")

    colunas = df.columns.tolist()
    col_titulo = st.selectbox("📝 Coluna do título:", colunas, index=colunas.index("Designacao Projecto") if "Designacao Projecto" in colunas else 0)
    col_resumo = st.selectbox("📋 Coluna da descrição/resumo:", colunas, index=colunas.index("Sumario Executivo") if "Sumario Executivo" in colunas else 0)

    col_manual1 = st.selectbox("✅ Classificação manual principal (opcional):", ["Nenhuma"] + colunas, index=colunas.index("Dominio ENEI") + 1 if "Dominio ENEI" in colunas else 0)
    col_manual2 = st.selectbox("📘 Classificação manual alternativa (opcional):", ["Nenhuma"] + colunas, index=colunas.index("Dominio ENEI Projecto") + 1 if "Dominio ENEI Projecto" in colunas else 0)

    dominios_enei = carregar_dominios_2020()

    st.markdown("### ⚙️ Quantos projetos queres classificar?")
    opcao_modo = st.radio("Modo:", ["Teste (1 projeto)", "5", "10", "20", "50", "Todos"])

    if opcao_modo == "Teste (1 projeto)":
        df = df.head(1)
    elif opcao_modo != "Todos":
        df = df.head(int(opcao_modo))

    n_proj = len(df)
    tokens_por_proj = 610
    total_tokens = n_proj * tokens_por_proj
    st.info(f"🧮 Estimativa: {total_tokens} tokens (aprox.) para {n_proj} projetos")

    if st.button("🚀 Classificar com LLM"):
        resultados = []
        with st.spinner("A classificar projetos..."):
            for _, row in df.iterrows():
                titulo = str(row.get(col_titulo, ""))
                resumo = str(row.get(col_resumo, ""))
                prompt = preparar_prompt(titulo, resumo, dominios_enei)
                classificacao = classificar_llm(prompt)

                linha = {
                    "NIPC": row.get("NIPC", ""),
                    "Projeto": titulo,
                    "Resumo": resumo,
                    "Domínio LLM": classificacao
                }

                if col_manual1 != "Nenhuma":
                    linha["Classificação Manual 1"] = row.get(col_manual1, "")
                if col_manual2 != "Nenhuma":
                    linha["Classificação Manual 2"] = row.get(col_manual2, "")

                resultados.append(linha)

        final_df = pd.DataFrame(resultados)
        st.success("✅ Classificação concluída com sucesso!")
        st.dataframe(final_df)

        st.session_state["classificacoes_llm"] = final_df.copy()

        buffer = BytesIO()
        final_df.to_excel(buffer, index=False)
        st.download_button(
            label="📥 Download (.xlsx)",
            data=buffer.getvalue(),
            file_name="classificacao_llm_enei2020.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
