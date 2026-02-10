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

# --- Fournisseurs connus (échantillon) ---
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

# --- Catégorisation variables ---
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

# --- Participations récurrentes — règles fixes (Vanessa / Jérémy)
PAT_VANESSA = re.compile(r"\bvanessa\b|participation.*vanessa", re.IGNORECASE)
PAT_JEREMY  = re.compile(r"\bjeremy\b|participation.*jeremy",   re.IGNORECASE)

PARTICIPATIONS_RULES = [
    {"label": "Participation Vanessa",  "person": "Vanessa", "amount": 1150.0, "day_start": 26, "day_end": 30},
    {"label": "Participation Jeremy",   "person": "Jeremy",  "amount": 1070.0, "day_start": 3,  "day_end": 3},
    {"label": "Participation Jeremy 2", "person": "Jeremy",  "amount":  530.0, "day_start": 3,  "day_end": 3},
]

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
    # Renvoie un Timestamp naïf (sans timezone) au dernier jour à minuit
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

def clamp_date_for_month(year: int, month: int, day: int) -> pd.Timestamp:
    last = pd.Period(f"{year}-{month:02d}").asfreq("M").end_time.normalize()
    safe_day = max(1, min(int(day), int(last.day)))
    return pd.Timestamp(year=year, month=month, day=safe_day)

def plan_participation_date(year: int, month: int, day_start: int, day_end: int) -> pd.Timestamp:
    """Jour prévu = milieu de la fenêtre (ou unique jour), clampé au dernier jour du mois."""
    target_day = int(round((day_start + day_end) / 2))
    return clamp_date_for_month(year, month, target_day)

# =========================
# ------ APPLICATION -------
# =========================

# ⚠️ Étape 1 — Upload
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

# --- Correctif : forcer l'usage de "Date operation" si elle existe
if "Date operation" in df.columns:
    date_col = "Date operation"
else:
    date_col = cols_map["date"]

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
# Convertit explicitement Date operation si présente
if "Date operation" in df.columns:
    df["Date operation"] = pd.to_datetime(df["Date operation"], errors="coerce", dayfirst=True)
# Fallback : convertit la colonne date_col si autre
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

df = df[~df[date_col].isna()]
df["amount_signed"] = coerce_amounts(df, amount_col, debit_col, credit_col)
df[label_col] = df[label_col].astype(str)

# --- Correctif : mois basé sur Date operation (ou date_col forcée)
df["mois"] = extract_month_period(df[date_col])
mois_detectes = np.sort(df["mois"].dropna().unique())

# Étape 2 — Mois
if len(mois_detectes) == 0:
    st.error("Aucun mois détecté.")
    st.stop()
elif len(mois_detectes) == 1:
    mois_choisi = mois_detectes[0]
else:
    st.header("Étape 2 — Choix du mois")
    mois_choisi = st.selectbox("Sélection du mois à analyser :", options=list(mois_detectes), format_func=month_name_fr)

df_month = df[df["mois"] == mois_choisi].copy()

# Info solde initial
st.info("Par défaut, **solde initial au 1er du mois = 0,00 €** (modifiable plus bas).")

# Étape 3 — Règles/Regex (expander)
with st.expander("⚙️ Ajuster les règles de détection (regex avancées)", expanded=False):
    pattern_alim = st.text_input("Regex Alimentaire", value=DEFAULT_PATTERN_ALIM, key="pat_alim")
    pattern_anim = st.text_input("Regex Animaux", value=DEFAULT_PATTERN_ANIM, key="pat_anim")
    pattern_fuel = st.text_input("Regex Carburant", value=DEFAULT_PATTERN_CARBURANT, key="pat_fuel")
    pattern_cash = st.text_input("Regex Retraits / Espèces (DAB/ATM)", value=DEFAULT_PATTERN_CASH, key="pat_cash")

# Stats historiques (basées sur la même colonne date_col)
prev_month = mois_choisi - 1
prev_3_months = [mois_choisi - i for i in (1, 2, 3)]
df_prev_month = df[df["mois"] == prev_month].copy()
df_prev_3     = df[df["mois"].isin(prev_3_months)].copy()

alim_curr = sum_category_negative(df_month,    label_col, pattern_alim)
anim_curr = sum_category_negative(df_month,    label_col, pattern_anim)
fuel_curr = sum_category_negative(df_month,    label_col, pattern_fuel)
cash_curr = sum_category_negative(df_month,    label_col, pattern_cash)

alim_last_3 = sum_category_negative(df_prev_3, label_col, pattern_alim)
anim_last_3 = sum_category_negative(df_prev_3, label_col, pattern_anim)
fuel_last_3 = sum_category_negative(df_prev_3, label_col, pattern_fuel)
cash_last_3 = sum_category_negative(df_prev_3, label_col, pattern_cash)

# Étape 3 bis — Fournisseurs/Contrats (expander)
with st.expander("🧾 Fournisseurs / contrats (libellés & regex de détection)", expanded=False):
    prov_df_default = pd.DataFrame(DEFAULT_PROVIDER_PATTERNS)
    prov_df = st.data_editor(
        prov_df_default, num_rows="dynamic", use_container_width=True, key="prov_editor",
        column_config={
            "Label": st.column_config.TextColumn("Libellé (affichage)"),
            "Regex": st.column_config.TextColumn("Regex de détection (libellé opération)")
        }
    )

provider_patterns: Dict[str, str] = {}
for _, row in (prov_df if 'prov_df' in locals() else pd.DataFrame(DEFAULT_PROVIDER_PATTERNS)).dropna(subset=["Label", "Regex"]).iterrows():
    lbl = str(row["Label"]).strip()
    rgx = str(row["Regex"]).strip()
    if lbl and rgx:
        provider_patterns[lbl] = rgx

if len(mois_detectes) >= 3:
    recur_global = detect_recurring(df, label_col, min_months=3)
    if not recur_global.empty:
        with st.expander("📈 Références récurrentes détectées (≥ 3 mois)", expanded=False):
            st.dataframe(recur_global.head(50), use_container_width=True)

# Détection ce mois (réel passé) pour fournisseurs
detected_by_provider = summarize_contracts(df_month, label_col, provider_patterns)

colY, colN = st.columns(2)
with colY:
    want_update = st.button("🟩 Modifier les coûts détectés", use_container_width=True)
with colN:
    no_update = st.button("🟥 Conserver tel quel", use_container_width=True)

# Attendus mensuels estimés (médiane historique)
expected_map: Dict[str, float] = {}
for k, rgx in provider_patterns.items():
    est_amt, _ = infer_provider_amount_and_day(df, date_col, label_col, rgx, months_back=6)
    expected_map[k] = float(est_amt or 0.0)

updated_costs = detected_by_provider.copy()

if want_update:
    st.subheader("Étape 4 — Modification guidée des coûts mensuels")
    st.caption("Pour chaque contrat : passée = réel à date / attendue = référence mensuelle (historique médian ou valeur saisie).")
    contrats = sorted(list(provider_patterns.keys()), key=lambda c: (-detected_by_provider.get(c, 0.0), c.lower()))
    for k in contrats:
        passed = float(detected_by_provider.get(k, 0.0))
        expected = float(expected_map.get(k, 0.0))
        st.markdown(f"**{k}** — passée : **{currency(passed)}**  |  attendue : **{currency(expected)}**")
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

        # Crédit immo
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
            if k == credit_label:
                continue
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

    # Formulaire simple
    last_day = last_day_of_period(mois_choisi)
    default_day = min(28, last_day.day)
    default_date = pd.Timestamp(year=last_day.year, month=last_day.month, day=default_day)
    with st.form("form_upcoming"):
        rows = []
        use_credit = st.checkbox("Inclure échéance Crédit immobilier", value=True, key="use_credit_ck")
        credit_amount = st.number_input(
            "Montant Crédit immobilier (€)", min_value=0.0, step=1.0,
            value=float(upcoming_df[upcoming_df["Label"]=="Crédit immobilier"]["Montant (€)"].sum() or 1411.0),
            key="credit_amt_in"
        )
        credit_date = st.date_input(
            "Date échéance Crédit immobilier",
            value=default_date.date(),
            min_value=last_day.replace(day=1).date(),
            max_value=last_day.date(),
            key="credit_date_in"
        )
        extra_n = st.number_input("Échéances supplémentaires (nombre)", min_value=0, max_value=10, value=0, step=1, key="extra_n")
        extra_items = []
        for i in range(int(extra_n)):
            st.markdown(f"**Échéance #{i+1}**")
            lbl = st.text_input(f"Libellé #{i+1}", value="", key=f"extra_lbl_{i}")
            amt = st.number_input(f"Montant #{i+1} (€)", min_value=0.0, step=1.0, value=0.0, key=f"extra_amt_{i}")
            dte = st.date_input(
                f"Date #{i+1}",
                value=default_date.date(),
                min_value=last_day.replace(day=1).date(),
                max_value=last_day.date(),
                key=f"extra_date_{i}"
            )
            extra_items.append((lbl, amt, pd.Timestamp(dte)))
        submitted = st.form_submit_button("Enregistrer les échéances")
        if submitted:
            if use_credit and credit_amount > 0:
                rows.append({"Label": "Crédit immobilier", "Date": pd.Timestamp(credit_date), "Montant (€)": float(credit_amount)})
            for lbl, amt, dte in extra_items:
                if lbl and amt > 0:
                    rows.append({"Label": lbl, "Date": dte, "Montant (€)": float(amt)})
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
        planned_lookup = {
            str(r["Label"]).strip(): (float(r["Montant (€)"]), pd.to_datetime(r["Date"], errors="coerce"))
            for _, r in updf.iterrows()
        }

rows = []
total_expected = 0.0
total_detected = 0.0
total_a_venir = 0.0

for k, rgx in provider_patterns.items():
    detected = round(float(detected_by_provider.get(k, 0.0)), 2)  # positif pour affichage
    expected = round(float(updated_costs.get(k, 0.0) or expected_map.get(k, 0.0) or 0.0), 2)

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
# PARTICIPATIONS — règles fixes + total visible + expander détail
# =========================

# Initialise une seule fois par période/mois
if ("participations_rows" not in st.session_state) or (st.session_state.get("participations_period") != str(mois_choisi)):
    st.session_state["participations_rows"] = [dict(r) for r in PARTICIPATIONS_RULES]
    st.session_state["participations_period"] = str(mois_choisi)

# Total visible
sum_fixed_credits = float(sum(row.get("amount", 0.0) for row in st.session_state["participations_rows"]))

st.subheader("Participations (revenus fixes)")
c_part_tot, c_part_other = st.columns([1,1])
with c_part_tot:
    st.metric("Participations — total (règles fixes)", currency(sum_fixed_credits))

# Détail (éditable)
with st.expander("Détail des participations (éditables)", expanded=False):
    cols = st.columns(3)
    edited_rows = []
    for i, row in enumerate(st.session_state["participations_rows"]):
        with cols[i % 3]:
            lbl  = st.text_input(f"Libellé participation #{i+1}", value=row.get("label", ""), key=f"p_lbl_{i}")
            amt  = st.number_input(f"Montant #{i+1} (€)", min_value=0.0, step=10.0, value=float(row.get("amount", 0.0)), key=f"p_amt_{i}")
            d_s  = st.number_input(f"Jour début #{i+1}", min_value=1, max_value=31, step=1, value=int(row.get("day_start", 28)), key=f"p_dstart_{i}")
            d_e  = st.number_input(f"Jour fin #{i+1}",   min_value=1, max_value=31, step=1, value=int(row.get("day_end",   28)), key=f"p_dend_{i}")
            person = row.get("person", "")
            edited_rows.append({"label": lbl, "amount": float(amt), "day_start": int(d_s), "day_end": int(d_e), "person": person})

    col_add1, col_add2, col_add3, col_add4 = st.columns([2,1,1,1])
    with col_add1:
        new_lbl = st.text_input("Libellé (nouvelle participation)", key="new_part_lbl")
    with col_add2:
        new_amt = st.number_input("Montant (€)", min_value=0.0, step=10.0, value=0.0, key="new_part_amt")
    with col_add3:
        new_day_start = st.number_input("Jour début", min_value=1, max_value=31, step=1, value=28, key="new_part_day_start")
    with col_add4:
        new_day_end   = st.number_input("Jour fin",   min_value=1, max_value=31, step=1, value=28, key="new_part_day_end")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Ajouter la participation"):
            if new_lbl and new_amt > 0:
                st.session_state["participations_rows"].append({
                    "label": new_lbl, "amount": float(new_amt),
                    "day_start": int(new_day_start), "day_end": int(new_day_end),
                    "person": ""})
                st.rerun()
    with b2:
        if st.button("Appliquer les modifications"):
            st.session_state["participations_rows"] = edited_rows
            st.rerun()

# Recalcule total après édition
sum_fixed_credits = float(sum(row.get("amount", 0.0) for row in st.session_state["participations_rows"]))

# Autres revenus (toujours séparé)
other_incomes = st.number_input("Autres revenus (mensuels)", min_value=0.0, step=10.0, value=0.0, key="other_inc")

# =========================
# DÉPENSES VARIABLES — total visible + expander 4 valeurs
# =========================
st.subheader("Dépenses variables")

# Pré-remplissage: courant (observé) ou moyenne 3 derniers mois
use_3m = st.checkbox("Pré-remplir avec la moyenne des 3 derniers mois", value=False, key="use_3m_prefill")

# Valeurs par défaut
if use_3m:
    var_alim_def = round(alim_last_3 / 3.0, 2)
    var_anim_def = round(anim_last_3 / 3.0, 2)
    var_fuel_def = round(fuel_last_3 / 3.0, 2)
    var_cash_def = round(cash_last_3 / 3.0, 2)
else:
    var_alim_def = round(alim_curr, 2)
    var_anim_def = round(anim_curr, 2)
    var_fuel_def = round(fuel_curr, 2)
    var_cash_def = round(cash_curr, 2)

# --- Correctif A : forcer la réinitialisation des inputs selon le mode
mode_suffix = "_3m" if use_3m else "_obs"

# Expander détail des 4 postes
with st.expander("Détail des variables (modifiable)", expanded=False):
    cva1, cva2, cva3, cva4 = st.columns(4)
    var_alim = cva1.number_input("Alimentaire (€)",      min_value=0.0, step=5.0, value=var_alim_def, key="var_alim_in"+mode_suffix)
    var_anim = cva2.number_input("Animaux (€)",          min_value=0.0, step=5.0, value=var_anim_def, key="var_anim_in"+mode_suffix)
    var_fuel = cva3.number_input("Carburant (€)",        min_value=0.0, step=5.0, value=var_fuel_def, key="var_fuel_in"+mode_suffix)
    var_cash = cva4.number_input("Retraits/espèces (€)", min_value=0.0, step=5.0, value=var_cash_def, key="var_cash_in"+mode_suffix)

# Total visible (hors marge) — calculé sur les variables locales
var_total = float(var_alim + var_anim + var_fuel + var_cash)

c_var_total, c_var_buf = st.columns([1,1])
with c_var_total:
    st.metric("Variables — total (hors marge)", currency(var_total))
with c_var_buf:
    buffer_safety = st.number_input("Marge de sécurité / aléas", min_value=0.0, step=10.0, value=0.0, key="var_buffer")

# =========================
# Agrégats & Projection (résultats en haut)
# =========================
sum_incomes = float(sum_fixed_credits + other_incomes)
sum_fixed   = float(total_charges)
sum_var     = float(var_total + buffer_safety)
projected_balance_full_month = sum_incomes - (sum_fixed + sum_var)

# Paramètres de trésorerie
solde_initial = st.number_input("Solde initial au 1er du mois (€)", value=0.0, step=50.0, key="solde_init")

# Date facultative pour autres revenus
use_other_income_date = st.checkbox("Définir une date pour 'Autres revenus' ?", value=False, key="use_oi_date")
other_income_date = None
if use_other_income_date:
    tmp_last = last_day_of_period(mois_choisi)
    tmp_first = tmp_last.replace(day=1)
    other_income_date = st.date_input(
        "Date pour 'Autres revenus' (facultatif)",
        value=tmp_first.date(),
        min_value=tmp_first.date(),
        max_value=tmp_last.date(),
        key="other_income_date"
    )

# ---- Résultats (en haut)
st.subheader("Résultats de projection — Mois complet")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenus totaux", currency(sum_incomes))
c2.metric("Charges fixes (plein mois)", currency(sum_fixed))
c3.metric("Dépenses variables (avec marge)", currency(sum_var))
c4.metric("Solde prévisionnel (plein mois)", currency(projected_balance_full_month))

# =========================
# Graphique “Trésorerie du mois (réel & projection)”
# =========================
first_day = last_day_of_period(mois_choisi).replace(day=1)

# 🔧 Correction TZ : tout en naïf
today_local = pd.Timestamp.now(tz="Europe/Paris").normalize().tz_localize(None)

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

# =========================
# Série projetée : fournisseurs restants + participations anticipées + autres revenus datés + CSV futurs + variables
# =========================
proj_events = []

last_day = last_day_of_period(mois_choisi)

# A) Charges fixes restantes (fournisseurs)
restants = fixed_status_df[(fixed_status_df["Montant à venir (€)"] > 0)]
for _, r in restants.iterrows():
    if r["Date prévue"]:
        d = pd.to_datetime(r["Date prévue"])
        if d.date() > cutoff_date.date():
            proj_events.append({"date": d.normalize(), "amount": -float(r["Montant à venir (€)"])})

# B) Participations anticipées (règles fixes) — évite doublon si déjà dans le CSV futur
def has_future_income_for_person(df_m: pd.DataFrame, regex_person: re.Pattern, after_date: pd.Timestamp) -> bool:
    if df_m.empty:
        return False
    mask = (df_m["amount_signed"] > 0) & (df_m[date_col] > after_date)
    subset = df_m[mask].copy()
    if subset.empty:
        return False
    lab = subset[label_col].fillna("").map(normalize_str).str.lower()
    return lab.apply(lambda s: bool(regex_person.search(s))).any()

has_future_vanessa = has_future_income_for_person(df_month, PAT_VANESSA, cutoff_date)
has_future_jeremy  = has_future_income_for_person(df_month, PAT_JEREMY,  cutoff_date)

for part in st.session_state["participations_rows"]:
    amount = float(part.get("amount", 0.0) or 0.0)
    d_s    = int(part.get("day_start", 28))
    d_e    = int(part.get("day_end",   28))
    person = part.get("person", "")
    if amount <= 0:
        continue

    # Date prévue ce mois (milieu de la fenêtre) avec clamp sur fin de mois (gère février)
    planned_date = plan_participation_date(last_day.year, last_day.month, d_s, d_e)

    # N'ajoute que si la date est dans le futur par rapport au cutoff
    if planned_date.date() <= cutoff_date.date():
        continue

    # Évite doublon si déjà présent après cutoff dans le CSV
    if person == "Vanessa" and has_future_vanessa:
        continue
    if person == "Jeremy"  and has_future_jeremy:
        continue

    proj_events.append({"date": planned_date.normalize(), "amount": float(amount)})

# C) Revenus datés dans le CSV (après cutoff)
incomes_future_csv = df_month[(df_month["amount_signed"] > 0) & (df_month[date_col] > cutoff_date)]
if not incomes_future_csv.empty:
    inc_by_date = incomes_future_csv.groupby(incomes_future_csv[date_col].dt.normalize())["amount_signed"].sum().reset_index()
    inc_by_date.rename(columns={date_col: "date", "amount_signed": "amount"}, inplace=True)
    for _, row in inc_by_date.iterrows():
        proj_events.append({"date": row["date"], "amount": float(row["amount"])})

# D) Autres revenus facultatifs (si datés)
if other_incomes > 0 and other_income_date:
    dt = pd.Timestamp(other_income_date)
    if dt.date() > cutoff_date.date():
        proj_events.append({"date": dt.normalize(), "amount": float(other_incomes)})

# E) Variables restantes (impact max en fin de mois)
if var_total > 0:
    proj_events.append({"date": last_day, "amount": -(var_total + buffer_safety)})

# Consolidation projection
proj_df = pd.DataFrame(proj_events)
if not proj_df.empty:
    proj_df = proj_df.groupby("date", as_index=False)["amount"].sum()
    if (cutoff_date in real_daily["date"].values):
        balance_cutoff = float(real_daily.loc[real_daily["date"] == cutoff_date, "balance"].iloc[0])
    else:
        balance_cutoff = float(real_daily["balance"].iloc[-1])
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

# _Expanders_ — Listes & exports
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

with st.expander("Exports", expanded=False):
    projection_payload = {
        "mois": str(mois_choisi),
        "participations": [
            {
                "label": r["label"],
                "montant": float(r["amount"]),
                "jour_debut": int(r["day_start"]),
                "jour_fin": int(r["day_end"]),
                "personne": r.get("person", "")
            }
            for r in st.session_state["participations_rows"]
        ],
        "revenus_autres": other_incomes,
        "revenus_total": float(sum_fixed_credits + other_incomes),
        "charges_fixes": sum_fixed,
        "variables": var_total,
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
            projection_payload["solde_previsionnel_fin_de_mois"] = float(
                (sum_fixed_credits + other_incomes) - (var_total + buffer_safety + updf["Montant (€)"].sum() + sum_fixed)
            )
    st.download_button(
        "⬇️ Télécharger la projection (JSON)",
        data=json.dumps(projection_payload, indent=2).encode("utf-8"),
        file_name=f"projection_{mois_choisi}.json",
        mime="application/json"
    )
