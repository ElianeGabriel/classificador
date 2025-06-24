import streamlit as st

st.markdown("# 🏠 Bem-vindo ao Classificador de Projetos ENEI")
st.markdown("""
Este sistema permite classificar automaticamente projetos de I&D&I segundo os domínios prioritários da Estratégia Nacional de Especialização Inteligente (ENEI), nas versões **2020** ou **2030**.

---

## 🚀 Funcionalidades disponíveis

### 🧠 Classificação com LLM
Utiliza modelos da OpenAI (como o GPT-4o) para classificar projetos com base no título e resumo, identificando os **dois domínios mais prováveis**, com percentagens estimadas. Suporta:
- Classificação por **ENEI 2020** ou **ENEI 2030**
- Ficheiros com múltiplas sheets e colunas flexíveis
- Comparação com classificações manuais (opcional)

### 📈 Métricas e Visualizações
Compara os resultados da classificação automática com as classificações manuais. Gera:
- **Relatório de desempenho** (precisão, recall, f1-score)
- **Matriz de confusão**
- **Gráficos interativos**
- **Download do relatório** em Excel

---

## 📄 Como usar

1. Acede ao menu lateral e seleciona **"🧠 Classificação com LLM"**
2. Faz upload de um ficheiro `.xlsx` com os projetos reais
3. Define as colunas de título e resumo (e as classificações manuais, se existirem)
4. Classifica e analisa os resultados
5. Acede à secção **📈 Métricas e Visualizações** para avaliar o desempenho

---

## ℹ️ Sobre este projeto
Desenvolvido para apoiar a ANI (Agência Nacional de Inovação) na avaliação automática de projetos, este sistema combina a **potência dos modelos de linguagem** com a **estrutura da política pública de inovação** em Portugal.

""")
