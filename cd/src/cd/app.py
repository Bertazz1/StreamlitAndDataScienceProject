import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Acidentes",
    page_icon="🚗",
    layout="wide",
)
# Função para carregar os dados
@st.cache_data
def carregar_dados(ficheiro_carregado):
    """
    carrega o arquivo CSV
    """
    if ficheiro_carregado is not None:
        try:
            
            df = pd.read_csv(ficheiro_carregado, sep=';', encoding='latin-1')

            # Converte a coluna 'data_inversa' para o formato de data
            df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')

            # Remove linhas onde a data não pôde ser convertida
            df.dropna(subset=['data_inversa'], inplace=True)

            return df
        except Exception as e:
            st.error(f"Ocorreu um erro ao ler o ficheiro: {e}")
            return None
    return None


st.title("🚗 Análise Interativa de Acidentes de Trânsito")
st.markdown("Bem-vindo! Para começar, por favor, carrega o teu ficheiro de dados de acidentes (.csv).")


ficheiro_csv = st.file_uploader("Escolhe o ficheiro CSV", type=["csv"])

if ficheiro_csv is not None:
    df = carregar_dados(ficheiro_csv)

    if df is not None:
        st.success("Ficheiro carregado e processado com sucesso!")

        # Filtros
        st.sidebar.header("Filtros")
        # Filtro dos estados
        ufs_disponiveis = sorted(df['uf'].unique())
        uf_selecionada = st.sidebar.multiselect(
            "Seleciona o Estado (UF):",
            options=ufs_disponiveis,
            default=ufs_disponiveis  
        )

       
        if uf_selecionada:
            df_filtrado = df[df['uf'].isin(uf_selecionada)]
        else:
            df_filtrado = df 

       
        st.header("Visão Geral dos Dados Filtrados")
        st.dataframe(df_filtrado.head())

        total_acidentes = len(df_filtrado)
        st.metric(label="Total de Acidentes", value=f"{total_acidentes:,}".replace(",", "."))


     
        st.header("Análises Visuais")

        
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico 1: Acidentes por Estado 
            st.subheader("Acidentes por Estado")
            acidentes_por_uf = df_filtrado['uf'].value_counts().sort_values(ascending=True)
            fig_uf = px.bar(
                acidentes_por_uf,
                x=acidentes_por_uf.values,
                y=acidentes_por_uf.index,
                orientation='h',
                title="Número de Acidentes por Estado",
                labels={'x': 'Número de Acidentes', 'y': 'Estado (UF)'},
                text=acidentes_por_uf.values
            )
            fig_uf.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_uf, use_container_width=True)

        with col2:
            # Gráfico 2: Acidentes por Dia da Semana
            st.subheader("Acidentes por Dia da Semana")
            acidentes_dia_semana = df_filtrado['dia_semana'].value_counts()
        
            ordem_dias = ["domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado"]
            acidentes_dia_semana = acidentes_dia_semana.reindex(ordem_dias)

            fig_dia_semana = px.bar(
                acidentes_dia_semana,
                x=acidentes_dia_semana.index,
                y=acidentes_dia_semana.values,
                title="Distribuição de Acidentes por Dia da Semana",
                labels={'x': 'Dia da Semana', 'y': 'Número de Acidentes'},
                text=acidentes_dia_semana.values
            )
            st.plotly_chart(fig_dia_semana, use_container_width=True)

        # Gráfico 3: Tendência de Acidentes ao Longo do Tempo
        st.subheader("Tendência de Acidentes ao Longo do Tempo")
        acidentes_por_data = df_filtrado.set_index('data_inversa').resample('ME').size()
        fig_temporal = px.line(
            acidentes_por_data,
            x=acidentes_por_data.index,
            y=acidentes_por_data.values,
            title="Número de Acidentes por Mês",
            labels={'x': 'Data', 'y': 'Número de Acidentes'}
        )
        st.plotly_chart(fig_temporal, use_container_width=True)


        # Gráfico 4: Top 10 Causas de Acidentes
        st.subheader("Top 10 Causas de Acidentes")
        top_10_causas = df_filtrado['causa_acidente'].value_counts().nlargest(10).sort_values(ascending=True)
        fig_causas = px.bar(
            top_10_causas,
            x=top_10_causas.values,
            y=top_10_causas.index,
            orientation='h',
            title="As 10 Causas Mais Comuns de Acidentes",
            labels={'x': 'Número de Acidentes', 'y': 'Causa do Acidente'},
            text=top_10_causas.values
        )
        st.plotly_chart(fig_causas, use_container_width=True)

else:
    st.info("Por favor, carrega um ficheiro CSV para iniciar a análise.")