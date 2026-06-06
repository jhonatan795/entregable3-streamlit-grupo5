
# ============================================================
# App Streamlit - Entregable 3
# Grupo 5 | Animal Nutrition | Predicción de sale_amount
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Animal_Nutrition",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Animal_nutrition")
st.caption("Grupo 5 | Animal Nutrition | Predicción de `sale_amount` con Regresión Lineal")


# ============================================================
# FUNCIONES
# ============================================================

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo)

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()

    # Limpieza básica de status_type
    if "status_type" in df.columns:
        def limpiar_status(x):
            x = str(x).strip().lower().replace(" ", "").replace("_", "")
            if x.startswith("log"):
                return "lograda"
            elif x.startswith("per"):
                return "perdida"
            elif x.startswith("abi"):
                return "abierta"
            else:
                return "otro"

        df["status_type"] = df["status_type"].apply(limpiar_status)

    # Conversión de fecha
    if "sale_at" in df.columns:
        df["sale_at"] = pd.to_datetime(df["sale_at"], errors="coerce")
        df["sale_year"] = df["sale_at"].dt.year
        df["sale_month"] = df["sale_at"].dt.month
        df["sale_quarter"] = df["sale_at"].dt.quarter

    # Eliminar duplicados sin considerar sale_id
    if "sale_id" in df.columns:
        columnas_sin_id = [c for c in df.columns if c != "sale_id"]
        df = df.drop_duplicates(subset=columnas_sin_id, keep="first")
    else:
        df = df.drop_duplicates()

    return df


def preparar_datos_modelo(df):
    variable_objetivo = "sale_amount"

    variables_numericas = [
        "quantity_kg_number",
        "price_amount",
        "import_cost_amount",
        "goal_RQE_amount"
    ]

    variables_categoricas = [
        "category_name",
        "product_family_name",
        "status_type",
        "segment_name",
        "origin_country_name",
        "customer_type"
    ]

    # Usar solo columnas disponibles
    variables_numericas = [c for c in variables_numericas if c in df.columns]
    variables_categoricas = [c for c in variables_categoricas if c in df.columns]

    columnas_modelo = [variable_objetivo] + variables_numericas + variables_categoricas
    data_modelo = df[columnas_modelo].copy()

    # Quitar filas sin variable objetivo
    data_modelo = data_modelo.dropna(subset=[variable_objetivo])

    X = data_modelo[variables_numericas + variables_categoricas]
    y = data_modelo[variable_objetivo]

    return X, y, variables_numericas, variables_categoricas


@st.cache_resource
def entrenar_modelo(df):
    X, y, variables_numericas, variables_categoricas = preparar_datos_modelo(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    transformador_numerico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler())
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", transformador_numerico, variables_numericas),
            ("cat", transformador_categorico, variables_categoricas)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocesamiento", preprocesador),
        ("modelo", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, np.nan, y_test))) * 100
    r2 = r2_score(y_test, y_pred)

    metricas = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

    return pipeline, metricas, X_test, y_test, y_pred, variables_numericas, variables_categoricas


def mostrar_kpis(df):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total ventas", f"${df['sale_amount'].sum():,.0f}")
    c2.metric("Venta promedio", f"${df['sale_amount'].mean():,.2f}")
    c3.metric("Registros", f"{len(df):,}")

    if "product_name" in df.columns:
        c4.metric("Productos únicos", f"{df['product_name'].nunique():,}")
    else:
        c4.metric("Columnas", f"{df.shape[1]:,}")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📁 Carga de datos")
archivo_csv = st.sidebar.file_uploader("Sube el archivo CSV", type=["csv"])

st.sidebar.divider()
pagina = st.sidebar.radio(
    "Navegación",
    [
        "1. Descripción",
        "2. Datos y limpieza",
        "3. EDA interactivo",
        "4. Correlaciones",
        "5. Pruebas de hipótesis",
        "6. Modelo predictivo",
        "7. Predicción"
    ]
)


# ============================================================
# VALIDACIÓN DE ARCHIVO
# ============================================================

if archivo_csv is None:
    st.info("⬅️ Sube el archivo CSV desde la barra lateral para iniciar.")
    st.stop()

df = cargar_datos(archivo_csv)

if "sale_amount" not in df.columns:
    st.error("La base de datos debe contener la variable objetivo `sale_amount`.")
    st.stop()


# ============================================================
# PÁGINA 1 - DESCRIPCIÓN
# ============================================================

if pagina == "1. Descripción":
    st.header("1. Descripción del proyecto")

    st.markdown("""
    Este proyecto desarrolla un proceso aplicado de analítica de datos sobre el histórico de ventas de Animal Nutrition.

    El objetivo principal es construir un modelo predictivo que permita estimar el monto de venta, representado por la variable **`sale_amount`**, a partir de variables comerciales, del producto, del cliente y de la transacción.

    La aplicación integra:

    - Carga y limpieza de datos.
    - Análisis exploratorio de datos.
    - Visualizaciones interactivas.
    - Análisis de correlaciones.
    - Pruebas de hipótesis.
    - Construcción de pipeline de preprocesamiento.
    - Entrenamiento de un modelo de Regresión Lineal.
    - Evaluación mediante MAE, RMSE, MAPE y R².
    - Predicción interactiva de `sale_amount`.
    """)

    st.subheader("Variable objetivo")
    st.success("La variable objetivo del modelo es `sale_amount`, correspondiente al monto total de venta.")


# ============================================================
# PÁGINA 2 - DATOS Y LIMPIEZA
# ============================================================

elif pagina == "2. Datos y limpieza":
    st.header("2. Datos y limpieza")

    st.write(f"Filas después de limpieza básica: **{df.shape[0]:,}**")
    st.write(f"Columnas disponibles: **{df.shape[1]:,}**")

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Tipos de datos")
    tipos = pd.DataFrame({
        "Variable": df.dtypes.index,
        "Tipo": df.dtypes.astype(str).values,
        "Nulos": df.isna().sum().values,
        "% Nulos": (df.isna().mean().values * 100).round(2)
    })
    st.dataframe(tipos, use_container_width=True, hide_index=True)


# ============================================================
# PÁGINA 3 - EDA
# ============================================================

elif pagina == "3. EDA interactivo":
    st.header("3. Análisis Exploratorio de Datos")

    mostrar_kpis(df)

    st.divider()

    dff = df.copy()

    col1, col2, col3 = st.columns(3)

    if "sale_year" in dff.columns:
        with col1:
            anios = sorted(dff["sale_year"].dropna().unique())
            sel_anios = st.multiselect("Año", anios, default=anios)
            dff = dff[dff["sale_year"].isin(sel_anios)]

    if "category_name" in dff.columns:
        with col2:
            categorias = sorted(dff["category_name"].dropna().unique())
            sel_categorias = st.multiselect("Categoría", categorias, default=categorias)
            dff = dff[dff["category_name"].isin(sel_categorias)]

    if "segment_name" in dff.columns:
        with col3:
            segmentos = sorted(dff["segment_name"].dropna().unique())
            sel_segmentos = st.multiselect("Segmento", segmentos, default=segmentos)
            dff = dff[dff["segment_name"].isin(sel_segmentos)]

    st.caption(f"Registros filtrados: {len(dff):,}")

    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Distribución de sale_amount")
        fig = px.histogram(dff, x="sale_amount", nbins=40, title="Histograma de sale_amount")
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        if "category_name" in dff.columns:
            st.subheader("sale_amount por categoría")
            fig = px.box(dff, x="category_name", y="sale_amount", color="category_name")
            st.plotly_chart(fig, use_container_width=True)

    if "sale_year" in dff.columns and "sale_month" in dff.columns:
        st.subheader("Evolución mensual de ventas")
        ventas_mes = dff.groupby(["sale_year", "sale_month"], as_index=False)["sale_amount"].sum()
        ventas_mes["periodo"] = ventas_mes["sale_year"].astype(str) + "-" + ventas_mes["sale_month"].astype(str).str.zfill(2)
        fig = px.line(ventas_mes, x="periodo", y="sale_amount", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    if "product_name" in dff.columns:
        st.subheader("Top 10 productos por monto de venta")
        top_productos = dff.groupby("product_name", as_index=False)["sale_amount"].sum()
        top_productos = top_productos.sort_values("sale_amount", ascending=False).head(10)
        fig = px.bar(top_productos, x="sale_amount", y="product_name", orientation="h", text_auto=True)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PÁGINA 4 - CORRELACIONES
# ============================================================

elif pagina == "4. Correlaciones":
    st.header("4. Correlaciones")

    variables_corr = [
        "sale_amount",
        "quantity_kg_number",
        "import_cost_amount",
        "price_amount",
        "goal_RQE_amount"
    ]
    variables_corr = [c for c in variables_corr if c in df.columns]

    corr = df[variables_corr].corr()

    fig = px.imshow(
        corr,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Matriz de correlación"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlaciones principales frente a sale_amount")
    if "sale_amount" in corr.columns:
        corr_objetivo = corr["sale_amount"].drop("sale_amount").sort_values(key=abs, ascending=False)
        tabla_corr = corr_objetivo.reset_index()
        tabla_corr.columns = ["Variable", "Correlación con sale_amount"]
        st.dataframe(tabla_corr, use_container_width=True, hide_index=True)

        st.markdown("""
        La correlación permite identificar qué tan relacionada está cada variable numérica con el monto de venta. 
        Valores cercanos a 1 indican relación positiva fuerte, valores cercanos a -1 indican relación negativa fuerte, 
        y valores cercanos a 0 indican baja relación lineal.
        """)


# ============================================================
# PÁGINA 5 - HIPÓTESIS
# ============================================================

elif pagina == "5. Pruebas de hipótesis":
    st.header("5. Pruebas de hipótesis")

    alpha = st.slider("Nivel de significancia α", 0.01, 0.10, 0.05, 0.01)

    st.subheader("Prueba 1: Ventas logradas vs no logradas")

    if "status_type" in df.columns:
        logradas = df[df["status_type"] == "lograda"]["sale_amount"].dropna()
        no_logradas = df[df["status_type"] != "lograda"]["sale_amount"].dropna()

        if len(logradas) > 1 and len(no_logradas) > 1:
            t_stat, p_two = stats.ttest_ind(logradas, no_logradas, equal_var=False)
            p_one = p_two / 2

            st.write("H₀: El promedio de ventas logradas es menor o igual al de ventas no logradas.")
            st.write("H₁: El promedio de ventas logradas es mayor al de ventas no logradas.")
            st.write(f"t = {t_stat:.4f}")
            st.write(f"p-valor = {p_one:.4f}")

            if p_one < alpha and t_stat > 0:
                st.success("Se rechaza H₀: las ventas logradas tienen un monto promedio significativamente mayor.")
            else:
                st.warning("No se rechaza H₀.")
        else:
            st.warning("No hay suficientes datos para esta prueba.")

    st.divider()

    st.subheader("Prueba 2: Diferencias entre categorías")

    if "category_name" in df.columns:
        grupos = [
            grupo["sale_amount"].dropna().values
            for _, grupo in df.groupby("category_name")
            if len(grupo["sale_amount"].dropna()) > 1
        ]

        if len(grupos) >= 2:
            f_stat, p_anova = stats.f_oneway(*grupos)
            st.write("H₀: Todas las categorías tienen el mismo promedio de venta.")
            st.write("H₁: Al menos una categoría tiene un promedio diferente.")
            st.write(f"F = {f_stat:.4f}")
            st.write(f"p-valor = {p_anova:.4f}")

            if p_anova < alpha:
                st.success("Se rechaza H₀: existen diferencias significativas entre categorías.")
            else:
                st.warning("No se rechaza H₀.")
        else:
            st.warning("No hay suficientes categorías para ANOVA.")

    st.divider()

    st.subheader("Prueba 3: Correlación entre cantidad y monto de venta")

    if "quantity_kg_number" in df.columns:
        datos_corr = df[["quantity_kg_number", "sale_amount"]].dropna()

        if len(datos_corr) > 2:
            r, p_corr = stats.pearsonr(datos_corr["quantity_kg_number"], datos_corr["sale_amount"])
            st.write("H₀: No existe correlación lineal entre cantidad vendida y monto de venta.")
            st.write("H₁: Existe correlación lineal entre cantidad vendida y monto de venta.")
            st.write(f"r = {r:.4f}")
            st.write(f"p-valor = {p_corr:.4f}")

            if p_corr < alpha:
                st.success("Se rechaza H₀: existe correlación lineal significativa.")
            else:
                st.warning("No se rechaza H₀.")


# ============================================================
# PÁGINA 6 - MODELO
# ============================================================

elif pagina == "6. Modelo predictivo":
    st.header("6. Modelo predictivo")

    st.markdown("""
    En esta sección la aplicación entrena automáticamente un modelo de **Regresión Lineal** para predecir `sale_amount`.
    El modelo se construye mediante un pipeline que imputa datos faltantes, escala variables numéricas, codifica variables categóricas y entrena el modelo.
    """)

    try:
        pipeline, metricas, X_test, y_test, y_pred, vars_num, vars_cat = entrenar_modelo(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{metricas['MAE']:,.2f}")
        c2.metric("RMSE", f"{metricas['RMSE']:,.2f}")
        c3.metric("MAPE", f"{metricas['MAPE']:,.2f}%")
        c4.metric("R²", f"{metricas['R2']:.4f}")

        st.subheader("Variables usadas")
        st.write("Variables numéricas:", vars_num)
        st.write("Variables categóricas:", vars_cat)

        resultados = pd.DataFrame({
            "Real": y_test.values,
            "Predicción": y_pred
        })
        resultados["Error"] = resultados["Real"] - resultados["Predicción"]

        st.subheader("Comparación real vs predicción")
        st.dataframe(resultados.head(20), use_container_width=True)

        fig = px.scatter(resultados, x="Real", y="Predicción", title="Valores reales vs predichos")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("No fue posible entrenar el modelo.")
        st.code(str(e))


# ============================================================
# PÁGINA 7 - PREDICCIÓN
# ============================================================

elif pagina == "7. Predicción":
    st.header("7. Predicción interactiva")

    try:
        pipeline, metricas, X_test, y_test, y_pred, vars_num, vars_cat = entrenar_modelo(df)

        st.markdown("Ingresa valores para estimar el monto de venta `sale_amount`.")

        entrada = {}

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Variables numéricas")
            for col in vars_num:
                valor_default = float(df[col].median()) if df[col].notna().any() else 0.0
                entrada[col] = st.number_input(col, value=valor_default)

        with col2:
            st.subheader("Variables categóricas")
            for col in vars_cat:
                opciones = sorted(df[col].dropna().astype(str).unique())
                entrada[col] = st.selectbox(col, opciones)

        if st.button("Predecir sale_amount", type="primary"):
            entrada_df = pd.DataFrame([entrada])
            prediccion = pipeline.predict(entrada_df)[0]
            prediccion = max(prediccion, 0)

            st.success(f"💰 Monto estimado de venta: ${prediccion:,.2f}")

            promedio = df["sale_amount"].mean()
            mediana = df["sale_amount"].median()

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicción", f"${prediccion:,.2f}")
            c2.metric("Promedio histórico", f"${promedio:,.2f}", delta=f"{prediccion - promedio:,.2f}")
            c3.metric("Mediana histórica", f"${mediana:,.2f}", delta=f"{prediccion - mediana:,.2f}")

    except Exception as e:
        st.error("No fue posible generar la predicción.")
        st.code(str(e))
