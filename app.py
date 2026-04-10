import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer
import gzip
import warnings
import os

# ── sklearn compatibility shims ───────────────────────────────────────────────
# Patch 1: _RemainderColsList removed in sklearn 1.5+
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        def __reduce__(self):
            return (self.__class__, (list(self),))
    _ct._RemainderColsList = _RemainderColsList

# Patch 2: SimpleImputer._fill_dtype added in sklearn 1.5 — old pickles lack it.
# We derive the correct dtype from statistics_ (set during fit) so that
# numeric imputers get float64 and string/object imputers get object dtype.
from sklearn.impute import SimpleImputer as _SI
import numpy as _np
_si_orig_transform = _SI.transform
def _si_transform_safe(self, X):
    if not hasattr(self, "_fill_dtype"):
        if hasattr(self, "statistics_"):
            self._fill_dtype = self.statistics_.dtype
        else:
            self._fill_dtype = _np.float64
    return _si_orig_transform(self, X)
_SI.transform = _si_transform_safe

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏍️ Bike Price Predictor",
    page_icon="🏍️",
    layout="centered",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, "Project", "Files")

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    ensemble = joblib.load(gzip.open(os.path.join(FILES_DIR, "tuned_ensemble_model.pkl.gz"), "rb"))
    knn      = joblib.load(gzip.open(os.path.join(FILES_DIR, "knn_model.pkl.gz"), "rb"))
    return ensemble, knn

# Defaults computed from all_bikez_curated.csv — avoids loading the LFS-tracked df_all_brands.pkl
DEFAULTS = {
    "Brand":        "bmw",
    "Bike":         "r 1200 gs",
    "Category":     "Sport",
    "Power [hp]":   50.8,
    "Displacement": 552.5,
    "Torque [Nm]":  64.5,
    "Mileage [km]": 15000.0,
    "Age [a]":      5,
}

# ── Constants ─────────────────────────────────────────────────────────────────
KNOWN_BRANDS = ["BMW", "Ducati", "KTM", "Royal Enfield", "Suzuki", "Yamaha"]
CATEGORIES   = [
    "Allround", "Classic", "Cross / motocross", "Custom / cruiser",
    "Enduro / offroad", "Naked bike", "Prototype / concept model",
    "Scooter", "Sport", "Sport touring", "Super motard",
    "Touring", "Trial", "Unspecified category",
]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏍️ Motorcycle Price Predictor")
st.markdown(
    "Enter your bike's details below and get an estimated **used market price** "
    "based on a trained machine learning model."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options=KNOWN_BRANDS, index=0)
    category = st.selectbox("Category", options=CATEGORIES,
                            index=CATEGORIES.index("Sport") if "Sport" in CATEGORIES else 0)
    power = st.number_input("Power (hp)", min_value=1.0, max_value=500.0,
                            value=DEFAULTS["Power [hp]"], step=1.0)
    displacement = st.number_input("Displacement (ccm)", min_value=50.0, max_value=3000.0,
                                   value=DEFAULTS["Displacement"], step=10.0)

with col2:
    bike_model = st.text_input("Model name", value=DEFAULTS["Bike"],
                               help="e.g. R1200GS, MT-07, 1290 Super Duke")
    torque = st.number_input("Torque (Nm)", min_value=1.0, max_value=500.0,
                             value=DEFAULTS["Torque [Nm]"], step=1.0)
    mileage = st.number_input("Mileage (km)", min_value=0.0, max_value=500_000.0,
                              value=DEFAULTS["Mileage [km]"], step=500.0)
    age = st.number_input("Age (years)", min_value=0, max_value=60,
                          value=int(DEFAULTS["Age [a]"]), step=1)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("💰 Predict Price", use_container_width=True, type="primary"):
    with st.spinner("Crunching numbers…"):
        try:
            ensemble_model, knn_model = load_models()

            input_data = {
                "Brand":            brand.lower(),
                "Bike":             bike_model.lower(),
                "Category":         category,
                "Power [hp]":       power,
                "Displacement [ccm]": displacement,
                "Torque [Nm]":      torque,
                "Mileage [km]":     mileage,
                "Age [a]":          int(age),
                "Condition":        mileage > 100,
            }

            X = pd.DataFrame([input_data])
            # pandas 3.0 defaults strings to StringDtype instead of object.
            # The fitted ColumnTransformer expects object dtype — cast explicitly.
            for col in ["Brand", "Bike", "Category"]:
                X[col] = X[col].astype(object)
            X["Condition"] = X["Condition"].astype(bool)

            transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

            warnings.simplefilter("ignore", category=UserWarning)
            ens_pred = ensemble_model.predict(X)
            knn_pred = knn_model.predict(X)

            ens_price = float(transformer.inverse_transform(ens_pred.reshape(-1, 1))[0])
            knn_price = float(transformer.inverse_transform(knn_pred.reshape(-1, 1))[0])
            avg_price = (ens_price + knn_price) / 2

            st.success("Prediction complete!")

            r1, r2, r3 = st.columns(3)
            r1.metric("🤖 Ensemble Model",  f"${ens_price:,.0f}")
            r2.metric("📍 KNN Model",       f"${knn_price:,.0f}")
            r3.metric("📊 Average",         f"${avg_price:,.0f}")

            st.caption(
                "Prices are estimates in USD based on training data. "
                "Actual market prices may vary."
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Make sure the model files are present in `Project/Files/`.")
