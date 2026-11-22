from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Data Sunda | Inflasi & Kebahagiaan",
    page_icon="📊",
    layout="wide",
)

DATA_FILE_CANDIDATES = [
    Path(__file__).with_name("WHI_Inflation.csv"),
]

FEATURE_SIMPLE = "Headline Consumer Price Inflation"
FEATURES_MULTI = [
    "Headline Consumer Price Inflation",
    "Energy Consumer Price Inflation",
    "Food Consumer Price Inflation",
    "GDP deflator Index growth rate",
]
TARGET = "Score"

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e7ebf2;
        padding: 14px 16px; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Dashboard Inflasi dan Kebahagiaan Antarnegara")
st.caption(
    "Proyek Akhir Data Science — Kelompok 24: Data Sunda | "
    "Periode pengamatan 2015–2023"
)

@st.cache_data
def load_data(file_bytes=None, file_name=None):
    if file_bytes is not None:
        from io import BytesIO
        data = pd.read_csv(BytesIO(file_bytes))
    else:
        path = next((p for p in DATA_FILE_CANDIDATES if p.exists()), None)
        if path is None:
            return None
        data = pd.read_csv(path)

    numeric_cols = [
        "Year", FEATURE_SIMPLE, "Energy Consumer Price Inflation",
        "Food Consumer Price Inflation", "Official Core Consumer Price Inflation",
        "Producer Price Inflation", "GDP deflator Index growth rate", TARGET,
        "GDP per Capita", "Social support", "Healthy life expectancy at birth",
        "Freedom to make life choices", "Generosity", "Perceptions of corruption",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

with st.sidebar:
    st.header("⚙️ Pengaturan Dashboard")
    uploaded = st.file_uploader("Unggah dataset CSV", type=["csv"])
    if uploaded is not None:
        df = load_data(uploaded.getvalue(), uploaded.name)
    else:
        df = load_data()

if df is None:
    st.error(
        "Dataset belum ditemukan. Letakkan `WHI_Inflation (1).csv` "
        "atau `WHI_Inflation.csv` satu folder dengan `app.py`, "
        "atau unggah CSV melalui sidebar."
    )
    st.stop()

required = {"Country", "Year", FEATURE_SIMPLE, TARGET}
missing = required - set(df.columns)
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")
    st.stop()

with st.sidebar:
    st.divider()
    year_values = sorted(df["Year"].dropna().astype(int).unique().tolist())
    selected_years = st.slider(
        "Rentang tahun",
        min_value=int(min(year_values)),
        max_value=int(max(year_values)),
        value=(int(min(year_values)), int(max(year_values))),
    )
    if "Continent/Region" in df.columns:
        regions = sorted(df["Continent/Region"].dropna().astype(str).unique())
        selected_regions = st.multiselect(
            "Wilayah", regions, default=regions
        )
    else:
        selected_regions = []

filtered = df[df["Year"].between(selected_years[0], selected_years[1])].copy()
if "Continent/Region" in filtered.columns and selected_regions:
    filtered = filtered[filtered["Continent/Region"].astype(str).isin(selected_regions)]

# Sidebar filters affect descriptive views. Model evaluation follows the notebook's
# complete-case data and fixed 80:20 split (random_state=42).
st.info(
    "Filter tahun dan wilayah memengaruhi ringkasan serta grafik eksplorasi. "
    "Evaluasi model mengikuti rancangan notebook: observasi lengkap dan "
    "pembagian train-test 80:20 dengan random_state=42."
)

valid = filtered.dropna(subset=[FEATURE_SIMPLE, TARGET])
n_countries = filtered["Country"].nunique()
avg_inflation = valid[FEATURE_SIMPLE].mean()
avg_score = valid[TARGET].mean()
corr = valid[FEATURE_SIMPLE].corr(valid[TARGET]) if len(valid) > 1 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah observasi", f"{len(filtered):,}")
k2.metric("Negara terwakili", f"{n_countries:,}")
k3.metric("Rata-rata inflasi utama", f"{avg_inflation:.2f}%" if pd.notna(avg_inflation) else "—")
k4.metric("Rata-rata skor kebahagiaan", f"{avg_score:.2f}" if pd.notna(avg_score) else "—")

tab_overview, tab_explore, tab_model, tab_predict, tab_data = st.tabs(
    ["🏠 Ringkasan", "🔎 Eksplorasi Data", "🤖 Pemodelan", "🧮 Simulasi Prediksi", "📁 Data"]
)

with tab_overview:
    st.subheader("Gambaran umum penelitian")
    st.write(
        "Dashboard ini membantu mengeksplorasi hubungan antara **Headline Consumer "
        "Price Inflation** dan **Score** kebahagiaan. Analisis regresi digunakan "
        "untuk melihat pola hubungan, bukan membuktikan sebab-akibat."
    )
    left, right = st.columns(2)
    with left:
        yearly = (
            valid.groupby("Year", as_index=False)
            .agg(Rata_rata_Inflasi=(FEATURE_SIMPLE, "mean"),
                 Jumlah_Observasi=(TARGET, "count"))
        )
        fig = px.line(
            yearly, x="Year", y="Rata_rata_Inflasi", markers=True,
            title="Tren Rata-rata Inflasi Utama",
            labels={"Rata_rata_Inflasi": "Inflasi rata-rata (%)", "Year": "Tahun"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        yearly_score = valid.groupby("Year", as_index=False)[TARGET].mean()
        fig = px.line(
            yearly_score, x="Year", y=TARGET, markers=True,
            title="Tren Rata-rata Skor Kebahagiaan",
            labels={TARGET: "Skor rata-rata", "Year": "Tahun"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hubungan inflasi dan skor kebahagiaan")
    fig = px.scatter(
        valid, x=FEATURE_SIMPLE, y=TARGET, color="Year",
        hover_name="Country", trendline="ols",
        labels={FEATURE_SIMPLE: "Inflasi utama (%)", TARGET: "Skor kebahagiaan"},
        title="Scatter Plot Inflasi vs Skor Kebahagiaan",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Korelasi Pearson pada data yang sedang difilter: {corr:.3f}. "
        "Nilai korelasi menunjukkan arah dan kekuatan hubungan linear, "
        "bukan hubungan sebab-akibat."
        if pd.notna(corr) else "Data tidak cukup untuk menghitung korelasi."
    )

with tab_explore:
    st.subheader("Eksplorasi dan kualitas data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Statistik deskriptif variabel utama**")
        columns_stats = [c for c in [FEATURE_SIMPLE, "Energy Consumer Price Inflation",
                                     "Food Consumer Price Inflation",
                                     "GDP deflator Index growth rate", TARGET]
                         if c in filtered.columns]
        st.dataframe(filtered[columns_stats].describe().T, use_container_width=True)
    with c2:
        st.markdown("**Jumlah nilai kosong per kolom**")
        missing_df = filtered.isna().sum().rename("Jumlah kosong").to_frame()
        st.dataframe(missing_df, use_container_width=True, height=300)

    st.markdown("**Distribusi skor kebahagiaan**")
    fig = px.histogram(valid, x=TARGET, nbins=30, title="Distribusi Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Negara dengan inflasi utama tertinggi pada filter aktif**")
    extremes = filtered.dropna(subset=[FEATURE_SIMPLE]).nlargest(
        10, FEATURE_SIMPLE
    )[["Country", "Year", FEATURE_SIMPLE, TARGET]]
    st.dataframe(extremes, use_container_width=True)

with tab_model:
    st.subheader("Perbandingan model regresi")
    st.write(
        "Perbandingan menggunakan observasi lengkap yang sama untuk empat prediktor "
        "agar model baseline, regresi sederhana, dan regresi berganda dievaluasi "
        "pada data uji yang setara."
    )
    missing_features = [c for c in FEATURES_MULTI if c not in df.columns]
    if missing_features:
        st.warning(f"Kolom untuk regresi berganda tidak tersedia: {missing_features}")
    else:
        model_data = df[[TARGET] + FEATURES_MULTI].dropna().copy()
        if len(model_data) < 10:
            st.warning("Observasi lengkap terlalu sedikit untuk evaluasi model.")
        else:
            train, test = train_test_split(
                model_data, test_size=0.20, random_state=42
            )
            y_train, y_test = train[TARGET], test[TARGET]

            simple = LinearRegression()
            simple.fit(train[[FEATURE_SIMPLE]], y_train)
            pred_simple = simple.predict(test[[FEATURE_SIMPLE]])

            multi = LinearRegression()
            multi.fit(train[FEATURES_MULTI], y_train)
            pred_multi = multi.predict(test[FEATURES_MULTI])

            pred_base = np.repeat(y_train.mean(), len(y_test))

            rows = []
            for name, pred in [
                ("Baseline rata-rata", pred_base),
                ("Regresi linear sederhana", pred_simple),
                ("Regresi linear berganda", pred_multi),
            ]:
                rows.append({
                    "Model": name,
                    "R²": r2_score(y_test, pred),
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                })
            metrics = pd.DataFrame(rows)
            st.dataframe(
                metrics.style.format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}),
                use_container_width=True,
            )
            best = metrics.sort_values(
                ["R²", "MAE", "RMSE"], ascending=[False, True, True]
            ).iloc[0]
            st.success(
                f"Model dengan hasil terbaik pada pembagian data ini: **{best['Model']}**. "
                "Pemilihan mempertimbangkan R² lebih tinggi serta MAE dan RMSE lebih rendah."
            )

            chosen = st.selectbox(
                "Tampilkan grafik prediksi untuk",
                ["Regresi linear sederhana", "Regresi linear berganda"],
            )
            if chosen == "Regresi linear sederhana":
                pred = pred_simple
            else:
                pred = pred_multi
            compare = pd.DataFrame({"Nilai aktual": y_test.values, "Nilai prediksi": pred})
            fig = px.scatter(
                compare, x="Nilai aktual", y="Nilai prediksi",
                title=f"Prediksi vs Nilai Aktual — {chosen}",
                opacity=0.65,
            )
            low = min(compare.min().min(), compare.max().min())
            high = max(compare.max().max(), compare.min().max())
            fig.add_shape(
                type="line", x0=low, y0=low, x1=high, y1=high,
                line=dict(dash="dash", color="red"),
            )
            st.plotly_chart(fig, use_container_width=True)

            error_df = compare.copy()
            error_df["Error absolut"] = (error_df["Nilai aktual"] - error_df["Nilai prediksi"]).abs()
            st.markdown("**Contoh lima prediksi dengan kesalahan terbesar (regresi sederhana)**")
            error_rows = test[["Score"]].copy()
            error_rows["Prediksi"] = pred_simple
            error_rows["Error absolut"] = np.abs(error_rows["Score"] - error_rows["Prediksi"])
            st.dataframe(
                error_rows.sort_values("Error absolut", ascending=False).head(5).round(4),
                use_container_width=True,
            )
            st.caption(
                "Evaluasi ini mereplikasi prinsip perbandingan notebook. "
                "Model sederhana dan berganda memakai train-test yang sama (80:20, seed 42)."
            )

with tab_predict:
    st.subheader("Simulasi estimasi skor dari inflasi")
    st.warning(
        "Simulasi edukatif, bukan ramalan resmi. Model dilatih pada seluruh observasi "
        "lengkap untuk demo. Hasil di luar rentang data merupakan ekstrapolasi dan "
        "perlu ditafsirkan dengan sangat hati-hati."
    )
    prediction_data = df[[FEATURE_SIMPLE, TARGET]].dropna().copy()
    if len(prediction_data) < 2:
        st.error("Data lengkap tidak cukup untuk melatih model simulasi.")
    else:
        demo_model = LinearRegression()
        demo_model.fit(prediction_data[[FEATURE_SIMPLE]], prediction_data[TARGET])
        min_x = float(prediction_data[FEATURE_SIMPLE].min())
        max_x = float(prediction_data[FEATURE_SIMPLE].max())
        default_x = float(np.clip(2.0, min_x, max_x))
        inflation_value = st.number_input(
            "Masukkan nilai inflasi utama (%)",
            min_value=float(min_x - 100),
            max_value=float(max_x + 100),
            value=default_x,
            step=0.1,
        )
        if st.button("Hitung estimasi skor", type="primary"):
            estimate = float(demo_model.predict(pd.DataFrame({FEATURE_SIMPLE: [inflation_value]}))[0])
            st.metric("Estimasi skor kebahagiaan", f"{estimate:.3f}")
            st.write(
                f"Persamaan model: Score = {demo_model.intercept_:.4f} "
                f"+ ({demo_model.coef_[0]:.4f} × Inflasi)"
            )
            if inflation_value < min_x or inflation_value > max_x:
                st.error(
                    f"Nilai inflasi berada di luar rentang data ({min_x:.2f}% hingga "
                    f"{max_x:.2f}%). Estimasi ini merupakan ekstrapolasi."
                )
            else:
                st.info(
                    f"Nilai inflasi berada dalam rentang data ({min_x:.2f}% hingga "
                    f"{max_x:.2f}%). Estimasi tetap memiliki ketidakpastian."
                )

with tab_data:
    st.subheader("Data yang digunakan")
    st.write(f"Ukuran dataset asli: **{df.shape[0]:,} baris × {df.shape[1]} kolom**")
    st.dataframe(filtered, use_container_width=True, height=450)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh data sesuai filter (CSV)",
        data=csv,
        file_name="data_sunda_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Kelompok 24 — Data Sunda | Analisis hubungan inflasi dan skor kebahagiaan "
    "antarnegara periode 2015–2023 menggunakan regresi linear."
)
