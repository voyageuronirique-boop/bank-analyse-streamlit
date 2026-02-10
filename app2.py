# app.py
# Analyse bancaire CSV — Workflow 7 étapes (Streamlit)
# Auteur: Adapté pour Jeremy Verhelst (CSV ; latin-1)
# Python 3.9+

import io
import re
import json
import unicodedata
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# =========================
# --------- CONFIG --------
# =========================
st.set_page_config(
    page_title="Analyse bancaire mensuelle",
    page_icon="💶",
    layout="wide"
)

# --------- STYLES / MOBILE -------
HIDE_SIDEBAR_CSS = """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem;}
.dataframe tbody tr th, .dataframe thead th {font-size: 0.9rem;}
.stButton>button, .stDownloadButton>button {border-radius: 8px; padding: 0.6rem 1rem; font-weight: 600;}
</style>
"""
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

# =========================
# ----- CONSTANTES --------
# =========================

DEFAULT_RECURRING_CREDITS = [
    {"label": "Participation Jeremy",   "amount": 1150.0},
    {"label": "Participation Vanessa",  "amount": 1050.0},
    {"label": "Participation Jeremy 2", "amount": 530.0},
]

DEFAULT_PROVIDER_PATTERNS = [
    {"Label": "Crédit immobilier", "Regex": r"(echeance.*pret|pret|credit|hypothec|immobilier|echeance\s*de\s*credit)"},
    {"Label": "Assurance habitation / BPCE", "Regex": r"(bpce\s+assurances?|multirisque|habitation)"},
    {"Label": "Assurance GENERALI IARD", "Regex": r"(generali\s+iard)"},
    {"Label": "Assurance GENERALI VIE", "Regex": r"(generali\s+vie)"},
    {"Label": "Freebox (Internet fixe)", "Regex": r"(free\s*telecom|freebox)"},
    {"Label": "Free Mobile", "Regex": r"(free\s*mobile)"},
    {"Label": "SFR (fixe/mobile)", "Regex": r"\bsfr\b"},
    {"Label": "Électricité — Sowee (EDF)", "Regex": r"(sowee\s*by\s*edf|sowee)"},
    {"Label": "Électricité — EDF", "Regex": r"\bedf\b"},
    {"Label": "Électricité — Bellenergie|Electricité de Provence", "Regex": r"(bellenergie|electricit[eé]\s*de\s*provence)"},
    {"Label": "Eau (SEM / régies)", "Regex": r"(soc(i[eé]t[eé])?\s*des\s*eaux|eau|veolia|suez|saur)"},
    {"Label": "Abonnements streaming", "Regex": r"(netflix|spotify|deezer|prime|canal\+|molotov|youtube\s*premium)"},
    {"Label": "Frais bancaires", "Regex": r"(cotis(ations)?\s+bancaires|frais\s+bancaires)"},
]

DEFAULT_PATTERN_ALIM = (
    r"(carrefour|leclerc|e\.?leclerc|intermarch[eé]|super\s*u|u\s?express|u drive|systeme\s?u|"
    r"auchan|lidl|aldi|monoprix|picard|grand\s*frais|biocoop|spar|casino|geant|franprix|"
    r"market|hyper|drive|"
    r"boucherie|charcuterie|boulangerie|patisserie|p[aâ]tisserie|"
    r"fromagerie|poissonnerie|primeur|mara[iî]cher|"
    r"marche\b|march[eé]\s+couvert|"
    r"thiriet|votre\s*marche|maxicoffee|"
    r"ubereats|uber\s*eats|deliveroo|just\s*eat|too\s*good\s*to\s*go)"
)
DEFAULT_PATTERN_ANIM = (
    r"(zooplus|bitiba|maxi\s*zoo|truffaut|animalis|botanic|jardiland|wanimo|zoofast|"
    r"ferme\s*des\s*animaux|medicanimal|"
    r"croquette|croquettes|liti[eè]re|friandises|"
    r"v[eé]t[eé]rinaire|veto|clinique\s*v[eé]t[eé]rinaire|antipuces|vermifuge|"
    r"royal\s*canin|pro\s*plan|purina|feliway|frontline|advantage|bravecto)"
)
DEFAULT_PATTERN_CARBURANT = (
    r"(total(?:energies)?|esso|bp|shell|avia|repsol|eni|dyneff|as24|cora\s*station|"
    r"leclerc\s*station|e\.?leclerc\s*station|intermarch[eé]\s*station|carrefour\s*station|"
    r"auchan\s*station|station\s*service|station\s*essence|"
    r"carburant|gasoil|gazole|diesel|sans\s*plomb|sp95|sp98)"
)
DEFAULT_PATTERN_CASH = (
    r"(retrait\s*(?:dab|gab)?|dab\b|gab\b|distributeur|"
    r"atm|atm\s*withdrawal|withdrawal|"
    r"retrait\s*especes|retrait\s*esp[eè]ces|"
    r"\bcash\b)"
)

# =========================
# ----- UTILITAIRES -------
# =========================

def normalize_str(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'\s+', ' ', s)
    return s

def try_read_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.read()
    def read_with(params):
        return pd.read_csv(io.BytesIO(raw), **params)
    trials = [
        dict(sep=";", encoding="utf-8", engine="python"),
        dict(sep=";", encoding="latin1", engine="python"),
        dict(sep=",", encoding="utf-8", engine="python"),
        dict(sep=",", encoding="latin1", engine="python"),
        dict(sep="\t", encoding="utf-8", engine="python"),
    ]
    last_err = None
    for p in trials:
        try:
            df = read_with(p)
            return df
        except Exception as e:
            last_err = e
    raise ValueError(f"Impossible de lire le CSV. Dernière erreur: {last_err}")

def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower().strip(): c for c in df.columns}
    candidates_date   = ["date operation", "date de comptabilisation", "date de valeur", "date", "valeur"]
    candidates_label  = ["libelle operation", "libellé operation", "libelle simplifie", "libelle", "libellé", "label", "description"]
    candidates_debit  = ["debit", "montant debit", "sortie", "debits"]
    candidates_credit = ["credit", "montant credit", "entree", "credits"]
    candidates_amount = ["montant", "amount", "valeur (€)", "value", "total", "solde mouvement"]

    def find(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        for k, v in cols.items():
            for c in cands:
                if c in k:
                    return v
        return None

    return {
        "date":   find(candidates_date),
        "label":  find(candidates_label),
        "debit":  find(candidates_debit),
        "credit": find(candidates_credit),
        "amount": find(candidates_amount)
    }

def coerce_amounts(df: pd.DataFrame, amount_col, debit_col, credit_col) -> pd.Series:
    def to_num(s):
        if pd.isna(s) or s == "":
            return 0.0
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s).replace("\u00A0", "").replace(" ", "")
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            s2 = re.sub(r"[^0-9\-\.+]", "", s)
            try:
                return float(s2)
            except Exception:
                return 0.0

    if amount_col and amount_col in df.columns:
        amt = df[amount_col].map(to_num).fillna(0.0)
        if debit_col and credit_col and debit_col in df.columns and credit_col in df.columns:
            deb = df[debit_col].map(to_num).fillna(0.0)
            cre = df[credit_col].map(to_num).fillna(0.0)
            signed = cre + deb  # exports FR: débits souvent déjà négatifs
            use_signed = signed.where(signed != 0, amt)
            return use_signed.fillna(0.0)
        return amt.fillna(0.0)

    deb = df[debit_col].map(to_num).fillna(0.0) if (debit_col and debit_col in df.columns) else pd.Series([0.0]*len(df))
    cre = df[credit_col].map(to_num).fillna(0.0) if (credit_col and credit_col in df.columns) else pd.Series([0.0]*len(df))
    return (cre + deb).fillna(0.0)

def extract_month_period(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M")

def month_name_fr(period: pd.Period) -> str:
    m = period.start_time.strftime("%B %Y")
    return m[0].upper() + m[1:] if m else ""

def last_day_of_period(p: pd.Period) -> pd.Timestamp:
    return p.asfreq('M').end_time.normalize()

def currency(x: float) -> str:
    try:
        return f"{x:,.2f} €".replace(",", " ").replace(".", ",")
    except Exception:
        return f"{x} €"

def download_button(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=file_name, mime="text/csv")

def sum_category_negative(df: pd.DataFrame, label_col: str, regex: str) -> float:
    if df.empty: return 0.0
    lab = df[label_col].fillna("").map(normalize_str)
    mask_cat = lab.str.contains(regex, flags=re.IGNORECASE, regex=True, na=False)
    amount = -df.loc[(mask_cat) & (df["amount_signed"] < 0), "amount_signed"].sum()
    return float(max(amount, 0.0))

def detect_recurring(df: pd.DataFrame, label_col: str, min_months: int = 3) -> pd.DataFrame:
    if "mois" not in df.columns:
        return pd.DataFrame(columns=["libelle_norm", "mois_count", "operations", "montant_median"])
    labn = df[label_col].fillna("").map(normalize_str).str.lower()
    by = df.assign(libelle_norm=labn).groupby(["libelle_norm", "mois"], as_index=False).agg(
        montant_median=("amount_signed", "median"),
        n_ops=("amount_signed", "size")
    )
    months_per_label = by.groupby("libelle_norm")["mois"].nunique().reset_index(name="mois_count")
    ops_count = df.assign(libelle_norm=labn).groupby("libelle_norm")["amount_signed"].size().reset_index(name="operations")
    med = df.assign(libelle_norm=labn).groupby("libelle_norm")["amount_signed"].median().reset_index(name="montant_median")
    res = months_per_label.merge(ops_count, on="libelle_norm").merge(med, on="libelle_norm")
    res = res.sort_values(["mois_count", "operations"], ascending=[False, False])
    return res[res["mois_count"] >= min_months]

# --- Fournisseurs/contrats récurrents ---
def match_provider(label: str, provider_patterns: Dict[str, str]) -> List[str]:
    """Retourne la liste des catégories 'provider' détectées dans le libellé."""
    hits: List[str] = []
    lab = normalize_str(label).lower()
    for k, pat in provider_patterns.items():
        try:
            if re.search(pat, lab, flags=re.IGNORECASE):
                hits.append(k)
        except re.error:
            continue
    return hits

def summarize_contracts(df_month: pd.DataFrame, label_col: str, provider_patterns: Dict[str, str]) -> Dict[str, float]:
    out = {k: 0.0 for k in provider_patterns.keys()}
    if df_month.empty: return out
    charges = df_month[df_month["amount_signed"] < 0].copy()
    charges["provider_hits"] = charges[label_col].fillna("").apply(lambda x: match_provider(x, provider_patterns))
    for k in out.keys():
        mask = charges["provider_hits"].apply(lambda hits: k in hits)
        if mask.any():
            out[k] = float(-charges.loc[mask, "amount_signed"].sum())
    return out

def infer_provider_amount_and_day(df_all: pd.DataFrame, date_col: str, label_col: str, regex: str, months_back: int = 6) -> Tuple[Optional[float], Optional[int]]:
    if df_all.empty or not isinstance(regex, str) or regex.strip() == "":
        return None, None
    periods_sorted = np.sort(df_all["mois"].dropna().unique())
    if len(periods_sorted) == 0: return None, None
    last_p = periods_sorted[-1]
    hist_months = [last_p - i for i in range(1, months_back + 1)]
    hist_df = df_all[df_all["mois"].isin(hist_months)].copy()
    if hist_df.empty: return None, None
    try:
        pat = re.compile(regex, re.IGNORECASE)
    except re.error:
        return None, None
    lab = hist_df[label_col].fillna("").map(normalize_str).str.lower()
    mask = lab.apply(lambda s: bool(pat.search(s)))
    hist_df = hist_df[mask]
    hist_df = hist_df[hist_df["amount_signed"] < 0]
    if hist_df.empty: return None, None
    med_amount = float(hist_df["amount_signed"].abs().median())
    day_median = int(pd.Series(hist_df[date_col].dt.day).median())
    return med_amount, day_median

def is_upcoming_empty_or_zero(dfu: pd.DataFrame) -> bool:
    if dfu is None or dfu.empty: return True
    col = "Montant (€)"
    if col not in dfu.columns: return True
    try:
        return float(dfu[col].sum()) <= 0.0
    except Exception:
        return True

# =========================
# ------ APPLICATION -------
# =========================

# ⚠️ Rien avant l'étape 1 (Upload)
st.header("Étape 1 — Import du relevé")
uploaded = st.file_uploader("Télécharge ton relevé bancaire (CSV)", type=["csv"])
if uploaded is None:
    st.stop()

# Lecture + préparation
try:
    df = try_read_csv(uploaded)
except Exception as e:
    st.error(f"Erreur lecture CSV : {e}")
    st.stop()

cols_map = infer_columns(df)
date_col   = cols_map["date"]
label_col  = cols_map["label"]
debit_col  = cols_map["debit"]
credit_col = cols_map["credit"]
amount_col = cols_map["amount"]

missing = []
if not date_col: missing.append("date")
if not label_col: missing.append("libellé/description")
if (not amount_col) and (not (debit_col and credit_col)):
    missing.append("montant (ou débit+crédit)")
if missing:
    st.error(
        "Colonnes manquantes/non détectées: " + ", ".join(missing) + "\n\n"
        f"Colonnes trouvées: {list(df.columns)}\n"
        "Astuce: colonnes typiques 'Date operation', 'Libelle operation', 'Debit', 'Credit'."
    )
    st.stop()

df = df.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df[~df[date_col].isna()]
df["amount_signed"] = coerce_amounts(df, amount_col, debit_col, credit_col)
df[label_col] = df[label_col].astype(str)
df["mois"] = extract_month_period(df[date_col])
mois_detectes = np.sort(df["mois"].dropna().unique())

# Sélection du mois
if len(mois_detectes) == 0:
    st.error("Aucun mois détecté.")
    st.stop()
elif len(mois_detectes) == 1:
    mois_choisi = mois_detectes[0]
else:
    mois_choisi = st.selectbox("Sélection du mois à analyser :", options=list(mois_detectes), format_func=month_name_fr)

df_month = df[df["mois"] == mois_choisi].copy()

# 🔵 NOUVEAU — INTRO : solde initial
st.info("Par défaut, **solde initial au 1er du mois = 0,00 €** (modifiable plus bas).")

# Règles variables (inchangé)
with st.expander("⚙️ Ajuster les règles de détection (regex avancées)"):
    pattern_alim = st.text_input("Regex Alimentaire", value=DEFAULT_PATTERN_ALIM)
    pattern_anim = st.text_input("Regex Animaux", value=DEFAULT_PATTERN_ANIM)
    pattern_fuel = st.text_input("Regex Carburant", value=DEFAULT_PATTERN_CARBURANT)
    pattern_cash = st.text_input("Regex Retraits / Espèces (DAB/ATM)", value=DEFAULT_PATTERN_CASH)

prev_month = mois_choisi - 1
prev_3_months = [mois_choisi - i for i in (1, 2, 3)]
df_prev_month = df[df["mois"] == prev_month].copy()
df_prev_3 = df[df["mois"].isin(prev_3_months)].copy()

alim_last_month = sum_category_negative(df_prev_month, label_col, pattern_alim)
alim_last_3     = sum_category_negative(df_prev_3,     label_col, pattern_alim)
anim_last_month = sum_category_negative(df_prev_month, label_col, pattern_anim)
anim_last_3     = sum_category_negative(df_prev_3,     label_col, pattern_anim)
fuel_last_month = sum_category_negative(df_prev_month, label_col, pattern_fuel)
fuel_last_3     = sum_category_negative(df_prev_3,     label_col, pattern_fuel)
cash_last_month = sum_category_negative(df_prev_month, label_col, pattern_cash)
cash_last_3     = sum_category_negative(df_prev_3,     label_col, pattern_cash)

# Étape 3 — Fournisseurs (éditables)
prov_df_default = pd.DataFrame(DEFAULT_PROVIDER_PATTERNS)
prov_df = st.data_editor(
    prov_df_default, num_rows="dynamic", use_container_width=True, key="prov_editor",
    column_config={"Label": st.column_config.TextColumn("Libellé (affichage)"),
                   "Regex": st.column_config.TextColumn("Regex de détection (libellé opération)")}
)
provider_patterns: Dict[str, str] = {}
for _, row in prov_df.dropna(subset=["Label", "Regex"]).iterrows():
    provider_patterns[str(row["Label"]).strip()] = str(row["Regex"]).strip()

if len(mois_detectes) >= 3:
    recur_global = detect_recurring(df, label_col, min_months=3)
    if not recur_global.empty:
        with st.expander("📈 Références récurrentes détectées (≥ 3 mois)"):
            st.dataframe(recur_global.head(50), use_container_width=True)

# Détection ce mois (réel passé)
detected_by_provider = summarize_contracts(df_month, label_col, provider_patterns)

colY, colN = st.columns(2)
with colY:
    want_update = st.button("🟩 Modifier les coûts détectés", use_container_width=True)
with colN:
    no_update = st.button("🟥 Conserver tel quel", use_container_width=True)

# 🔵 NOUVEAU — attendu mensuel pré‑calculé (pour afficher 'passée / attendue' en Étape 4)
expected_map: Dict[str, float] = {}
for k, rgx in provider_patterns.items():
    # base = valeur saisissable ultérieurement ; si aucune, on estime via l'historique
    est_amt, _ = infer_provider_amount_and_day(df, date_col, label_col, rgx, months_back=6)
    expected_map[k] = float(est_amt or 0.0)

updated_costs = detected_by_provider.copy()  # sera écrasé par les saisies

if want_update:
    st.subheader("Étape 4 — Modification guidée des coûts mensuels")
    st.caption("Pour chaque contrat : **passée = réel à date** / **attendue = référence mensuelle** (historique médian ou valeur saisie).")
    contrats = list(provider_patterns.keys())
    # ordre : ceux avec réel détecté en premier
    contrats = sorted(contrats, key=lambda c: (-detected_by_provider.get(c, 0.0), c.lower()))

    for k in contrats:
        passed = float(detected_by_provider.get(k, 0.0))
        expected = float(expected_map.get(k, 0.0))
        st.markdown(f"**{k}** — passée : **{currency(passed)}**  |  attendue : **{currency(expected)}**")  # 🔵 NOUVEAU
        new_val = st.number_input(
            f"Montant mensuel attendu pour {k}",
            min_value=0.0, step=1.0, value=expected, key=f"cost_{k}"
        )
        updated_costs[k] = new_val
    st.info("Les montants saisis remplacent la détection automatique pour la projection.")
elif no_update:
    st.write("Très bien, je conserve la détection actuelle pour la suite.")

# Mois courant ?
is_current_month = (str(mois_choisi) == pd.Timestamp.now(tz="Europe/Paris").strftime("%Y-%m"))

# Étape 5 — Échéances à venir (pré-remplissage + formulaire)
upcoming_df = pd.DataFrame(columns=["Label", "Date", "Montant (€)"])
if is_current_month:
    if (("upcoming_df" not in st.session_state)
        or (st.session_state.get("upcoming_period") != str(mois_choisi))
        or is_upcoming_empty_or_zero(st.session_state.get("upcoming_df"))):
        rows_auto = []

        # Crédit immo (exemple par défaut si non détectable)
        credit_label = "Crédit immobilier"
        default_credit_contract = float(updated_costs.get(credit_label, 0.0) or 0.0)
        credit_regex = provider_patterns.get(
            credit_label, r"(echeance.*pret|pret|credit|hypothec|immobilier|echeance\s*de\s*credit)"
        )
        if default_credit_contract <= 0:
            est_amt, _ = infer_provider_amount_and_day(df, date_col, label_col, credit_regex, months_back=6)
            default_credit_contract = est_amt if (est_amt and est_amt > 0) else 1411.0

        last_day = last_day_of_period(mois_choisi)
        _, inferred_day = infer_provider_amount_and_day(df, date_col, label_col, credit_regex, months_back=6)
        default_day = max(1, min((inferred_day or 28), last_day.day))
        default_date = pd.Timestamp(year=last_day.year, month=last_day.month, day=default_day)
        rows_auto.append({"Label": credit_label, "Date": default_date, "Montant (€)": float(default_credit_contract)})

        # Autres fournisseurs pas encore passés ce mois
        lab_norm_month = df_month[label_col].fillna("").map(normalize_str).str.lower()
        for k, rgx in provider_patterns.items():
            if k == credit_label: continue
            try:
                pat = re.compile(rgx, re.IGNORECASE)
            except re.error:
                continue
            deja_passe = lab_norm_month.apply(lambda s: bool(pat.search(s))).any()
            if not deja_passe:
                est_amt, est_day = infer_provider_amount_and_day(df, date_col, label_col, rgx, months_back=6)
                if est_amt and est_amt > 0:
                    day = max(1, min((est_day or 28), last_day.day))
                    rows_auto.append({
                        "Label": k,
                        "Date": pd.Timestamp(year=last_day.year, month=last_day.month, day=day),
                        "Montant (€)": float(est_amt)
                    })

        upcoming_df = pd.DataFrame(rows_auto, columns=["Label", "Date", "Montant (€)"])
        st.session_state["upcoming_df"] = upcoming_df
        st.session_state["upcoming_period"] = str(mois_choisi)
    else:
        upcoming_df = st.session_state["upcoming_df"]

    # Formulaire simple (tu peux ajuster ou ajouter d'autres échéances)
    last_day = last_day_of_period(mois_choisi)
    default_day = min(28, last_day.day)
    default_date = pd.Timestamp(year=last_day.year, month=last_day.month, day=default_day)
    with st.form("form_upcoming"):
        rows = []
        use_credit = st.checkbox("Inclure échéance Crédit immobilier", value=True)
        credit_amount = st.number_input("Montant Crédit immobilier (€)", min_value=0.0, step=1.0,
                                        value=float(upcoming_df[upcoming_df["Label"]=="Crédit immobilier"]["Montant (€)"].sum() or 1411.0))
        credit_date = st.date_input("Date échéance Crédit immobilier", value=default_date.date(),
                                    min_value=last_day.replace(day=1).date(), max_value=last_day.date())
        extra_n = st.number_input("Échéances supplémentaires (nombre)", min_value=0, max_value=10, value=0, step=1)
        extra_items = []
        for i in range(int(extra_n)):
            st.markdown(f"**Échéance #{i+1}**")
            lbl = st.text_input(f"Libellé #{i+1}", value="", key=f"extra_lbl_{i}")
            amt = st.number_input(f"Montant #{i+1} (€)", min_value=0.0, step=1.0, value=0.0, key=f"extra_amt_{i}")
            dte = st.date_input(f"Date #{i+1}",
                                value=default_date.date(),
                                min_value=last_day.replace(day=1).date(),
                                max_value=last_day.date(),
                                key=f"extra_date_{i}")
            extra_items.append((lbl, amt, pd.Timestamp(dte)))
        submitted = st.form_submit_button("Enregistrer les échéances")
        if submitted:
            if use_credit and credit_amount > 0:
                rows.append({"Label": "Crédit immobilier", "Date": pd.Timestamp(credit_date), "Montant (€)": float(credit_amount)})
            for lbl, amt, dte in extra_items:
                if lbl and amt > 0:
                    rows.append({"Label": lbl, "Date": dte, "Montant (€)": float(amt)})  # ✅ corrige la coquille
            upcoming_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Label", "Date", "Montant (€)"])
            st.session_state["upcoming_df"] = upcoming_df
            st.success("Échéances enregistrées ✅")

# =========================
# Charges fixes — statut / tableau
# =========================
planned_lookup: Dict[str, Tuple[float, pd.Timestamp]] = {}
if is_current_month:
    updf = st.session_state.get("upcoming_df", pd.DataFrame(columns=["Label", "Date", "Montant (€)"]))
    if isinstance(updf, pd.DataFrame) and not updf.empty:
        planned_lookup = {str(r["Label"]).strip(): (float(r["Montant (€)"]), pd.to_datetime(r["Date"], errors="coerce"))
                          for _, r in updf.iterrows()}

rows = []
total_expected = 0.0
total_detected = 0.0
total_a_venir = 0.0

for k, rgx in provider_patterns.items():
    detected = round(float(detected_by_provider.get(k, 0.0)), 2)  # POSITIF pour affichage (réel à date)
    # attendu = valeur saisie si fournie, sinon estimation
    expected = float(updated_costs.get(k, 0.0) or expected_map.get(k, 0.0) or 0.0)
    expected = round(expected, 2)

    statut = "réglée" if detected > 0 else ("à venir" if is_current_month else "non détectée")
    date_prev = None
    montant_av = 0.0

    if detected > 0 and expected > 0 and detected + 1e-6 < expected:
        statut = "partiellement réglée"
        if is_current_month:
            if k in planned_lookup:
                montant_av, date_prev = planned_lookup[k]
            else:
                montant_av = max(expected - detected, 0.0)
                _, est_day = infer_provider_amount_and_day(df, date_col, label_col, rgx, months_back=6)
                if est_day:
                    last_day = last_day_of_period(mois_choisi)
                    date_prev = pd.Timestamp(year=last_day.year, month=last_day.month, day=min(est_day, last_day.day))
            total_a_venir += float(montant_av or 0.0)
    elif detected == 0:
        if is_current_month:
            statut = "à venir"
            if k in planned_lookup:
                montant_av, date_prev = planned_lookup[k]
            else:
                montant_av = expected
                _, est_day = infer_provider_amount_and_day(df, date_col, label_col, rgx, months_back=6)
                if est_day:
                    last_day = last_day_of_period(mois_choisi)
                    date_prev = pd.Timestamp(year=last_day.year, month=last_day.month, day=min(est_day, last_day.day))
            total_a_venir += float(montant_av or 0.0)
        else:
            statut = "non détectée"

    total_expected += expected
    total_detected += detected

    rows.append({
        "Contrat": k,
        "Réel à date (€)": detected,
        "Prévu mois (€)": expected,
        "Statut": statut,
        "Date prévue": (date_prev.strftime("%Y-%m-%d") if (date_prev is not None and str(date_prev) != "NaT") else ""),
        "Montant à venir (€)": round(float(montant_av or 0.0), 2)
    })

fixed_status_df = pd.DataFrame(rows)
charges_df = fixed_status_df[["Contrat", "Prévu mois (€)"]].rename(columns={"Prévu mois (€)": "Montant (€/mois)"})
total_charges = float(charges_df["Montant (€/mois)"].sum())

# =========================
# Projection (résultats en haut)
# =========================

# Revenus fixes (éditables – pour les agrégats)
if "credits_rows" not in st.session_state:
    st.session_state["credits_rows"] = DEFAULT_RECURRING_CREDITS
credit_cols = st.columns(3)
edited_credits = []
for i, c in enumerate(st.session_state["credits_rows"]):
    with credit_cols[i % 3]:
        label_val = st.text_input(f"Libellé crédit #{i+1}", value=c["label"], key=f"rc_lbl_{i}")
        amt_val = st.number_input(f"Montant #{i+1} (€)", min_value=0.0, step=10.0, value=float(c["amount"]), key=f"rc_amt_{i}")
        edited_credits.append({"label": label_val, "amount": float(amt_val)})

with st.expander("➕ Ajouter un crédit récurrent"):
    new_lbl = st.text_input("Libellé (nouveau crédit)")
    new_amt = st.number_input("Montant (€) — nouveau crédit", min_value=0.0, step=10.0, value=0.0)
    if st.button("Ajouter"):
        if new_lbl and new_amt > 0:
            st.session_state["credits_rows"].append({"label": new_lbl, "amount": float(new_amt)})
            st.experimental_rerun()

sum_fixed_credits = float(sum(c["amount"] for c in edited_credits))
other_incomes = st.number_input("Autres revenus (mensuels)", min_value=0.0, step=10.0, value=0.0)

# Variables (observées ou moyenne 3m)
suggested_var_3m_total = float(sum([
    round(alim_last_3 / 3.0, 2),
    round(anim_last_3 / 3.0, 2),
    round(fuel_last_3 / 3.0, 2),
    round(cash_last_3 / 3.0, 2),
]))
compiled_fixed = []
for pat in provider_patterns.values():
    try: compiled_fixed.append(re.compile(pat, re.IGNORECASE))
    except re.error: continue
def is_fixed(label: str) -> bool:
    lab = normalize_str(label)
    return any(p.search(lab) for p in compiled_fixed) if compiled_fixed else False
df_month["is_fixed_guess"] = df_month[label_col].apply(is_fixed)
variable_spend_obs = -df_month.loc[(df_month["amount_signed"] < 0) & (~df_month["is_fixed_guess"]), "amount_signed"].sum()
variable_spend_obs = float(max(variable_spend_obs, 0.0))
use_3m = st.checkbox("Utiliser la moyenne des 3 derniers mois comme base pour 'Variables'", value=False)
var_input = st.number_input("Dépenses variables (observées/estimées, modifiable)", min_value=0.0, step=10.0,
                            value=round(suggested_var_3m_total if use_3m else variable_spend_obs, 2))
buffer_safety = st.number_input("Marge de sécurité / aléas (optionnel)", min_value=0.0, step=10.0, value=0.0)

sum_incomes = float(sum_fixed_credits + other_incomes)
sum_fixed   = float(total_charges)
sum_var     = float(var_input + buffer_safety)
projected_balance_full_month = sum_incomes - (sum_fixed + sum_var)

# 🔵 NOUVEAU — Paramètres de trésorerie
solde_initial = st.number_input("Solde initial au 1er du mois (€)", value=0.0, step=50.0)
other_income_date = st.date_input("Date pour 'Autres revenus' (facultatif)", value=None, key="other_income_date")

# ---- Résultats (en haut)
st.subheader("Résultats de projection — Mois complet")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenus totaux", currency(sum_incomes))
c2.metric("Charges fixes (plein mois)", currency(sum_fixed))
c3.metric("Dépenses variables", currency(sum_var))
c4.metric("Solde prévisionnel (plein mois)", currency(projected_balance_full_month))

# =========================
# Graphique “Trésorerie du mois (réel & projection)”
# =========================
first_day = last_day_of_period(mois_choisi).replace(day=1)
today_local = pd.Timestamp.now(tz="Europe/Paris").normalize()
cutoff_date = min(today_local, last_day_of_period(mois_choisi))
max_real_date = df_month[date_col].max() if not df_month.empty else first_day
cutoff_date = min(cutoff_date, max_real_date)

# Série réelle (cumul)
events_real = df_month.groupby(df_month[date_col].dt.date)["amount_signed"].sum().rename("amount").reset_index()
events_real["date"] = pd.to_datetime(events_real[date_col].astype(str))
events_real = events_real[["date", "amount"]]
dates_all_to_cutoff = pd.date_range(first_day, cutoff_date, freq="D")
real_daily = pd.DataFrame({"date": dates_all_to_cutoff}).merge(events_real, on="date", how="left").fillna({"amount": 0.0})
real_daily["balance"] = solde_initial + real_daily["amount"].cumsum()

# 🔵 NOUVEAU — Série projetée : on prend les **revenus fixes datés dans le CSV** + charges à venir
proj_events = []

# A) Charges fixes restantes
restants = fixed_status_df[(fixed_status_df["Montant à venir (€)"] > 0)]
for _, r in restants.iterrows():
    if r["Date prévue"]:
        d = pd.to_datetime(r["Date prévue"])
        if d.date() > cutoff_date.date():
            proj_events.append({"date": d.normalize(), "amount": -float(r["Montant à venir (€)"])})

# B) Revenus **datés dans le CSV** (opérations positives après la date d’extraction)
incomes_future_csv = df_month[(df_month["amount_signed"] > 0) & (df_month[date_col] > cutoff_date)]
if not incomes_future_csv.empty:
    inc_by_date = incomes_future_csv.groupby(incomes_future_csv[date_col].dt.normalize())["amount_signed"].sum().reset_index()
    inc_by_date.rename(columns={date_col: "date", "amount_signed": "amount"}, inplace=True)
    for _, row in inc_by_date.iterrows():
        proj_events.append({"date": row["date"], "amount": float(row["amount"])})

# C) Autres revenus facultatifs (si datés)
if other_incomes > 0 and other_income_date:
    dt = pd.Timestamp(other_income_date)
    if dt.date() > cutoff_date.date():
        proj_events.append({"date": dt.normalize(), "amount": float(other_incomes)})

# D) Variables : on met le **reste** en fin de mois pour l’impact max (inchangé)
remaining_variables = max(0.0, sum_var - max(0.0, -df_month.loc[(df_month["amount_signed"] < 0) & (~df_month["is_fixed_guess"]), "amount_signed"].sum()))
last_day = last_day_of_period(mois_choisi)
if remaining_variables > 0:
    proj_events.append({"date": last_day, "amount": -remaining_variables})

proj_df = pd.DataFrame(proj_events)
if not proj_df.empty:
    proj_df = proj_df.groupby("date", as_index=False)["amount"].sum()
    balance_cutoff = float(real_daily.loc[real_daily["date"] == cutoff_date, "balance"].iloc[0] if (cutoff_date in real_daily["date"].values) else real_daily["balance"].iloc[-1])
    dates_future = pd.date_range(cutoff_date + pd.Timedelta(days=1), last_day, freq="D")
    future_daily = pd.DataFrame({"date": dates_future}).merge(proj_df, on="date", how="left").fillna({"amount": 0.0})
    future_daily["balance"] = balance_cutoff + future_daily["amount"].cumsum()
    real_plot = real_daily.assign(kind="Réel")[["date", "balance", "kind"]]
    proj_plot = future_daily.assign(kind="Projection")[["date", "balance", "kind"]]
    treasury_plot = pd.concat([real_plot, proj_plot], ignore_index=True)
else:
    treasury_plot = real_daily.assign(kind="Réel")[["date", "balance"]]
    treasury_plot["kind"] = "Réel"

st.subheader("Trésorerie du mois (réel & projection)")
line = alt.Chart(treasury_plot).mark_line(size=2).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("balance:Q", title="Trésorerie (€)"),
    color=alt.Color("kind:N", scale=alt.Scale(domain=["Réel", "Projection"], range=["#2C7BE5", "#2C7BE5"]), legend=None),
    strokeDash=alt.condition(alt.datum.kind == "Projection", alt.value([6,4]), alt.value([1])),
    tooltip=[alt.Tooltip("date:T", title="Date"),
             alt.Tooltip("balance:Q", title="Trésorerie", format=",.2f")]
).properties(height=320)
st.altair_chart(line, use_container_width=True)

# _Expanders_ demandés (échéances planifiées / charges)
if is_current_month:
    with st.expander("Échéances planifiées / prévues ce mois", expanded=False):
        updf_show = st.session_state.get("upcoming_df", pd.DataFrame(columns=["Label","Date","Montant (€)"]))
        if isinstance(updf_show, pd.DataFrame) and not updf_show.empty:
            st.dataframe(updf_show.sort_values("Date"), hide_index=True, use_container_width=True)
        else:
            st.info("Aucune échéance planifiée pour l’instant.")

with st.expander(f"Charges du mois — Réel vs Prévu ({month_name_fr(mois_choisi)})", expanded=False):
    st.dataframe(
        fixed_status_df[["Contrat", "Réel à date (€)", "Prévu mois (€)", "Montant à venir (€)", "Date prévue", "Statut"]],
        hide_index=True, use_container_width=True
    )
with st.expander("Charges fixes — (table des attendus mensuels)", expanded=False):
    st.dataframe(charges_df, hide_index=True, use_container_width=True)
    st.success(f"**Total charges fixes prévues (mois complet) : {currency(total_charges)}**")

# Exports
with st.expander("Exports"):
    download_button(charges_df, "⬇️ Télécharger les charges fixes (CSV)", f"charges_fixes_{mois_choisi}.csv")
    projection_payload = {
        "mois": str(mois_choisi),
        "revenus_fixes": {c["label"]: c["amount"] for c in edited_credits},
        "revenus_autres": other_incomes,
        "revenus_total": sum_incomes,
        "charges_fixes": sum_fixed,
        "variables": var_input,
        "marge_securite": buffer_safety,
        "solde_previsionnel_plein_mois": projected_balance_full_month,
        "charges_detail": {k: float(v or 0.0) for k, v in updated_costs.items()}
    }
    if is_current_month:
        updf = st.session_state.get("upcoming_df", pd.DataFrame(columns=["Label","Date","Montant (€)"]))
        if isinstance(updf, pd.DataFrame) and (not updf.empty):
            projection_payload["echeances_a_venir"] = [
                {"label": str(r["Label"]), "date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"), "montant": float(r["Montant (€)"])}
                for _, r in updf.iterrows()
            ]
            projection_payload["solde_previsionnel_fin_de_mois"] = float(sum_incomes - (var_input + buffer_safety + updf["Montant (€)"].sum()))
    st.download_button(
        "⬇️ Télécharger la projection (JSON)",
        data=json.dumps(projection_payload, indent=2).encode("utf-8"),
        file_name=f"projection_{mois_choisi}.json",
        mime="application/json"
    )
