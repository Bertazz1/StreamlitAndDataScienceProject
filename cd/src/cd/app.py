import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Análise e Previsão de Acidentes",
    page_icon="🚗",
    layout="wide",
)

@st.cache_data
def carregar_dados(ficheiro_carregado):
    if ficheiro_carregado is not None:
        try:
            df = pd.read_csv(ficheiro_carregado, sep=';', encoding='latin-1')
            df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
            
            colunas_essenciais = [
                'data_inversa', 'classificacao_acidente', 'uf', 'dia_semana', 
                'causa_acidente', 'tipo_pista', 'condicao_metereologica', 
                'mortos', 'tipo_acidente', 'horario', 'latitude', 'longitude', 'tipo_veiculo'
            ]
            df.dropna(subset=colunas_essenciais, inplace=True)
            
            df['mortos'] = pd.to_numeric(df['mortos'], errors='coerce').fillna(0).astype(int)
            
            return df
        except Exception as e:
            st.error(f"Ocorreu um erro ao ler o ficheiro: {e}")
            return None
    return None

@st.cache_resource
def treinar_modelo_com_balanceamento(df, alvo, features):
    df_modelo = df.copy()
    
    if alvo == 'mortos':
        alvo_col = 'teve_morte'
        df_modelo[alvo_col] = df_modelo['mortos'].apply(lambda x: 1 if x > 0 else 0)
    else:
        alvo_col = alvo
        classes_comuns = df_modelo[alvo_col].value_counts().nlargest(2).index
        df_modelo = df_modelo[df_modelo[alvo_col].isin(classes_comuns)]

    X = pd.get_dummies(df_modelo[features]).astype(int)
    y = df_modelo[alvo_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train_res, y_train_res)
    
    y_pred = modelo.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return modelo, X.columns, report

def fazer_previsao(modelo, colunas_modelo, input_data):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=colunas_modelo, fill_value=0)
    
    previsao = modelo.predict(input_aligned)
    probabilidade = modelo.predict_proba(input_aligned)
    
    return previsao, probabilidade


st.title("🚗 Análise e Previsão de Acidentes de Trânsito")
st.markdown("Carregue os seus dados para explorar as análises e os modelos de previsão.")

ficheiro_csv = st.file_uploader("Escolha o ficheiro CSV de acidentes", type=["csv"])

if ficheiro_csv is not None:
    df = carregar_dados(ficheiro_csv)

    if df is not None:
        
        features_gravidade = ['uf', 'dia_semana', 'causa_acidente', 'tipo_pista', 'condicao_metereologica', 'tipo_veiculo', 'tipo_acidente']
        features_mortalidade = ['uf', 'dia_semana', 'causa_acidente', 'tipo_pista', 'condicao_metereologica', 'tipo_acidente', 'tipo_veiculo']

        modelo_gravidade, cols_grav, report_grav = treinar_modelo_com_balanceamento(df, 'classificacao_acidente', features_gravidade)
        modelo_morte, cols_morte, report_morte = treinar_modelo_com_balanceamento(df, 'mortos', features_mortalidade)

        st.success("Modelos de Machine Learning treinados com sucesso usando dados balanceados!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Performance - Modelo de Gravidade")
            st.json(report_grav)
        with col2:
            st.write("#### Performance - Modelo de Mortalidade")
            st.json(report_morte)

        tab_analise, tab_pred_gravidade, tab_pred_morte = st.tabs([
            "📊 Análise Exploratória", 
            "🤖 Previsão de Gravidade", 
            "💀 Previsão de Chance de Morte"
        ])
        
        with tab_analise:
            st.header("📊 Análise Exploratória dos Dados")
            st.markdown("Explore diferentes facetas dos dados de acidentes através dos gráficos interativos abaixo.")

            st.sidebar.header("Filtros para Análise")
            ufs = sorted(df['uf'].unique())
            uf_sel = st.sidebar.multiselect("Estado (UF):", options=ufs, default=ufs, key="analise_uf_multiselect")
            
            df_filtrado = df[df['uf'].isin(uf_sel)] if uf_sel else df

            st.dataframe(df_filtrado.head())
            st.metric("Total de Acidentes na Seleção", f"{len(df_filtrado):,}".replace(",", "."))
            st.divider()

            df_analise = df_filtrado.copy()
            df_analise['hora'] = pd.to_datetime(df_analise['horario'], format='%H:%M:%S', errors='coerce').dt.hour
            df_analise.dropna(subset=['hora'], inplace=True)
            df_analise['hora'] = df_analise['hora'].astype(int)

            st.subheader("Análise Temporal dos Acidentes")
            col1_graf, col2_graf = st.columns([2, 1])
            with col1_graf:
                st.markdown("##### Acidentes por Hora do Dia")
                acidentes_por_hora = df_analise['hora'].value_counts().sort_index()
                fig_hora = px.bar(x=acidentes_por_hora.index, y=acidentes_por_hora.values, labels={'x': 'Hora do Dia', 'y': 'Número de Acidentes'}, title="Picos de Acidentes ao Longo do Dia")
                fig_hora.update_layout(xaxis=dict(tickmode='linear'))
                st.plotly_chart(fig_hora, use_container_width=True)
            with col2_graf:
                st.markdown("##### Insights Temporais")
                if not acidentes_por_hora.empty:
                    pico_hora = acidentes_por_hora.idxmax()
                    vale_hora = acidentes_por_hora.idxmin()
                    st.info(f"**Pico de Acidentes:** Ocorreu às **{pico_hora}h**.")
                    st.warning(f"**Menor Incidência:** Registada às **{vale_hora}h** (madrugada).")
            
            st.divider()
            
            st.subheader("Mapa de Calor: Concentração de Acidentes")
            heatmap_data = df_analise.groupby(['dia_semana', 'hora']).size().reset_index(name='contagem')
            heatmap_pivot = heatmap_data.pivot_table(index='dia_semana', columns='hora', values='contagem', fill_value=0)
            ordem_dias = ["domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado"]
            heatmap_pivot = heatmap_pivot.reindex(ordem_dias)
            fig_heatmap = px.imshow(heatmap_pivot, labels=dict(x="Hora do Dia", y="Dia da Semana", color="Nº de Acidentes"), title="Concentração de Acidentes por Dia e Hora")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.divider()

            st.subheader("Análise por Características da Via e Veículos")
            col3_graf, col4_graf = st.columns(2)
            with col3_graf:
                st.markdown("##### Top 10 Tipos de Veículos Envolvidos")
                top_veiculos = df_analise['tipo_veiculo'].value_counts().nlargest(10)
                fig_veiculos = px.bar(top_veiculos, y=top_veiculos.index, x=top_veiculos.values, orientation='h', labels={'y': 'Tipo de Veículo', 'x': 'Número de Acidentes'}, text=top_veiculos.values)
                fig_veiculos.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_veiculos, use_container_width=True)
            with col4_graf:
                st.markdown("##### Acidentes por Tipo de Pista")
                acidentes_pista = df_analise['tipo_pista'].value_counts()
                fig_pista = px.pie(values=acidentes_pista.values, names=acidentes_pista.index, title="Distribuição por Tipo de Pista")
                st.plotly_chart(fig_pista, use_container_width=True)
            
            st.divider()
            
            st.subheader("Mapa Geográfico de Acidentes")
            df_mapa = df_analise.copy()
            df_mapa['latitude'] = df_mapa['latitude'].astype(str).str.replace(',', '.').astype(float)
            df_mapa['longitude'] = df_mapa['longitude'].astype(str).str.replace(',', '.').astype(float)
            df_mapa.dropna(subset=['latitude', 'longitude'], inplace=True)
            if len(df_mapa) > 10000:
                st.warning(f"A mostrar apenas 10,000 de {len(df_mapa)} acidentes no mapa para melhor performance.")
                df_mapa = df_mapa.sample(10000, random_state=42)
            st.map(df_mapa[['latitude', 'longitude']])

        with tab_pred_gravidade:
            st.header("🤖 Prever a Gravidade do Acidente")
            with st.form("form_gravidade"):
                col1, col2 = st.columns(2)
                with col1:
                    uf = st.selectbox("Estado (UF)", sorted(df['uf'].unique()), key="grav_uf")
                    causa = st.selectbox("Causa do Acidente", df['causa_acidente'].value_counts().nlargest(20).index, key="grav_causa")
                    tipo_veiculo = st.selectbox("Tipo de Veículo", df['tipo_veiculo'].value_counts().nlargest(20).index, key="grav_tipo_veiculo")
                with col2:
                    dia = st.selectbox("Dia da Semana", df['dia_semana'].unique(), key="grav_dia")
                    pista = st.selectbox("Tipo de Pista", df['tipo_pista'].unique(), key="grav_pista")
                    tipo_acidente = st.selectbox("Tipo de Acidente", df['tipo_acidente'].unique(), key="grav_tipo_acidente")
                
                condicao = st.selectbox("Condição Meteorológica", df['condicao_metereologica'].unique(), key="grav_condicao")
                
                submitted = st.form_submit_button("Prever Gravidade")
                if submitted:
                    input_data = {
                        'uf': uf, 
                        'dia_semana': dia, 
                        'causa_acidente': causa, 
                        'tipo_pista': pista, 
                        'condicao_metereologica': condicao,
                        'tipo_veiculo': tipo_veiculo,
                        'tipo_acidente': tipo_acidente
                    }
                    previsao, prob = fazer_previsao(modelo_gravidade, cols_grav, input_data)
                    st.subheader("Resultado da Previsão:")
                    st.warning(f"**Resultado:** {previsao[0]}")
                    st.dataframe(pd.DataFrame(prob, columns=modelo_gravidade.classes_, index=["Probabilidade"]))

        with tab_pred_morte:
            st.header("💀 Prever a Chance de Morte")
            with st.form("form_morte"):
                col1, col2 = st.columns(2)
                with col1:
                    uf_m = st.selectbox("Estado (UF)", sorted(df['uf'].unique()), key="morte_uf")
                    causa_m = st.selectbox("Causa do Acidente", df['causa_acidente'].value_counts().nlargest(20).index, key="morte_causa")
                    pista_m = st.selectbox("Tipo de Pista", df['tipo_pista'].unique(), key="morte_pista")
                with col2:
                    dia_m = st.selectbox("Dia da Semana", df['dia_semana'].unique(), key="morte_dia")
                    tipo_acidente_m = st.selectbox("Tipo de Acidente", df['tipo_acidente'].unique(), key="morte_tipo")
                    tipo_veiculo_m = st.selectbox("Tipo de Veículo", df['tipo_veiculo'].value_counts().nlargest(20).index, key="morte_tipo_veiculo")
                
                condicao_m = st.selectbox("Condição Meteorológica", df['condicao_metereologica'].unique(), key="morte_condicao")

                submitted_m = st.form_submit_button("Prever Chance de Morte")
                if submitted_m:
                    input_data_m = {
                        'uf': uf_m, 
                        'dia_semana': dia_m, 
                        'causa_acidente': causa_m, 
                        'tipo_pista': pista_m, 
                        'condicao_metereologica': condicao_m, 
                        'tipo_acidente': tipo_acidente_m,
                        'tipo_veiculo': tipo_veiculo_m
                    }
                    previsao, prob = fazer_previsao(modelo_morte, cols_morte, input_data_m)
                    st.subheader("Resultado da Previsão:")
                    resultado = "Provável Morte" if previsao[0] == 1 else "Provável Sem Morte"
                    
                    if resultado == "Provável Morte":
                        st.error(f"**Resultado:** {resultado}")
                    else:
                        st.success(f"**Resultado:** {resultado}")
                    
                    prob_df = pd.DataFrame(prob, columns=["Sem Morte", "Com Morte(s)"], index=["Probabilidade"])
                    st.dataframe(prob_df)
else:
    st.info("Por favor, carrega um ficheiro CSV para iniciar a análise e a previsão.")