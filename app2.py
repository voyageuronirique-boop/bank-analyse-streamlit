# app.py
# Analyse bancaire CSV — Workflow 7 étapes (Streamlit)
# Auteur : adapté pour Jeremy Verhelst — robuste CSV FR/EN + catégorisation regex éditable
# Python 3.9+

from __future__ import annotations

import io
import re
import json
import base64
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
.stButton>button {border-radius: 8px; padding: 0.6rem 1rem; font-weight: 600;}
.stDownloadButton>button {border-radius: 8px; padding: 0.6rem 1rem; font-weight: 600;}
</style>
"""
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

# =========================
# ----- UTILITAIRES -------
# =========================
def normalize_str(s: str) -> str:
    """Normalise/latinise et compresse les espaces."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s

def try_read_csv(uploaded) -> pd.DataFrame:
    """Lit le CSV en essayant plusieurs couples séparateur/encodage."""
    raw = uploaded.read()

    def read_with(params):
        return pd.read_csv(io.BytesIO(raw), **params)

    trials = [
        dict(sep=";", encoding="utf-8", engine="python"),
        dict(sep=",", encoding="utf-8", engine="python"),
        dict(sep="\t", encoding="utf-8", engine="python"),
        dict(sep=";", encoding="latin1", engine="python"),
        dict(sep=",", encoding="latin1", engine="python"),
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
    """
    Devine les colonnes clés: date, libellé, débit, crédit, montant (selon banques/export FR/EN).
    """
    cols = {c.lower().strip(): c for c in df.columns}

    candidates_date   = ["date", "valeur", "date operation", "date_op", "operation date", "transaction date", "date de l'operation"]
    candidates_label  = ["libelle", "libellé", "label", "description", "motif", "details", "remarque"]
    candidates_debit  = ["debit", "debit (€)", "montant debit", "sortie", "debits"]
    candidates_credit = ["credit", "credit (€)", "montant credit", "entree", "credits"]
    candidates_amount = ["montant", "amount", "valeur (€)", "value", "total", "solde mouvement"]

    def find(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        # fallback partiel
        for k, v in cols.items():
            for c in cands:
                if c in k:
                    return v
        return None

    return {
        "date": find(candidates_date),
        "label": find(candidates_label),
        "debit": find(candidates_debit),
        "credit": find(candidates_credit),
        "amount": find(candidates_amount),
    }

def _to_num(s) -> float:
    """Convertit chaîne hétérogène en float, gère formats FR (1 234,56)."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("\u00A0", "").replace(" ", "")
    # Format FR : virgule décimale
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def coerce_amounts(df: pd.DataFrame, amount_col, debit_col, credit_col) -> pd.Series:
    """
    Crée 'amount_signed' (négatif pour sorties, positif pour entrées).
    Priorité : si 'amount' existe => signe conservé ; sinon crédit - débit.
    """
    if amount_col and amount_col in df.columns:
        amt = df[amount_col].map(_to_num)
        if debit_col and credit_col and (debit_col in df.columns) and (credit_col in df.columns):
            # parfois les exports posent signe inversé: si abs(amt) == debit or credit, on garde amt
            return amt
        return amt

    deb = df[debit_col].map(_to_num).fillna(0.0) if (debit_col and debit_col in df.columns) else 0.0
    cre = df[credit_col].map(_to_num).fillna(0.0) if (credit_col and credit_col in df.columns) else 0.0
    return cre - deb

# =========================
# ----- PATRONS/REGEX -----
# =========================
# Crédit récurrents (entrées) — éditables dans l'UI
DEFAULT_RECURRING_CREDITS = [
    {"label": "Participation Jeremy",  "amount": 1150.0},
    {"label": "Participation Vanessa", "amount": 1050.0},
    {"label": "Participation Jeremy 2","amount": 530.0},
]

# Fournisseurs / contrats récurrents (charges fixes) — éditables
DEFAULT_PROVIDER_PATTERNS = [
    {"Label": "Crédit immobilier",                       "Regex": r"(echeance.*pret|pret|credit|hypothec|immobilier|echeance\s*de\s*credit)"},
    {"Label": "Assurance habitation / BPCE",             "Regex": r"(bpce\s+assurances?|multirisque|habitation)"},
    {"Label": "Assurance GENERALI IARD",                 "Regex": r"(generali\s+iard)"},
    {"Label": "Assurance GENERALI VIE",                  "Regex": r"(generali\s+vie)"},
    {"Label": "Freebox (Internet fixe)",                 "Regex": r"(free\s*telecom|freebox)"},
    {"Label": "Free Mobile",                              "Regex": r"(free\s*mobile)"},
    {"Label": "SFR (fixe/mobile)",                        "Regex": r"\bsfr\b"},
    {"Label": "Électricité — Sowee (EDF)",               "Regex": r"(sowee\s*by\s*edf|sowee)"},
    {"Label": "Électricité — EDF",                        "Regex": r"\bedf\b"},
    {"Label": "Électricité — Bellenergie / EdP",         "Regex": r"(bellenergie|electricit[eé]\s*de\s*provence)"},
    {"Label": "Eau (SEM / régies)",                       "Regex": r"(soc(i[eé]t[eé])?\s*des\s*eaux|eau|veolia|suez|saur)"},
    {"Label": "Abonnements streaming",                    "Regex": r"(netflix|spotify|deezer|prime|canal\+|molotov|youtube\s*premium)"},
    {"Label": "Frais bancaires",                          "Regex": r"(cotis(ations)?\s+bancaires|frais\s+bancaires)"},
]

# Catégories variables — patrons longs (éditables)
DEFAULT_PATTERN_ALIM = (
    r"(carrefour|leclerc|e\.?leclerc|intermarch[eé]|super\s*u|u\s?express|u\s*drive|systeme\s?u|"
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
    r"(retrait\s*(?:dab|gab)?|dab\b|gab\b|distributeur|atm|atm\s*withdrawal|withdrawal|"
    r"retrait\s*especes|retrait\s*esp[eè]ces|\bcash\b)"
)

# =========================
# ----- DATA CLASSES ------
# =========================
@dataclass
class ColumnMap:
    date: Optional[str]
    label: Optional[str]
    debit: Optional[str]
    credit: Optional[str]
    amount: Optional[str]

# =========================
# --------- UI ------------
# =========================
st.title("Analyse bancaire CSV — Workflow 7 étapes")
st.caption("Charge un relevé CSV, catégorise automatiquement, puis visualise & exporte.")

with st.expander("ℹ️ Comment préparer le CSV ?", expanded=False):
    st.markdown(
        "- Export natif de ta banque (CSV, séparateur `;`, `,` ou `\\t`).\n"
        "- Encodage UTF‑8 ou Latin‑1 supporté.\n"
        "- Colonnes attendues (si dispo) : **date**, **libellé**, **débit**, **crédit**, **montant**."
    )

uploaded = st.file_uploader("Dépose ton fichier CSV", type=["csv"])

# Paramètres (éditables)
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("⚙️ Patrons fournisseurs (fixes)")
    providers_json = st.text_area(
        "Liste JSON de fournisseurs (Label/Regex)",
        value=json.dumps(DEFAULT_PROVIDER_PATTERNS, ensure_ascii=False, indent=2),
        height=220
    )
with colB:
    st.subheader("⚙️ Crédits récurrents (entrées fixes)")
    credits_json = st.text_area(
        "Liste JSON de crédits (label/amount)",
        value=json.dumps(DEFAULT_RECURRING_CREDITS, ensure_ascii=False, indent=2),
        height=220
    )

st.subheader("⚙️ Catégories variables (Regex, une par ligne)")
col1, col2, col3 = st.columns(3)
with col1:
    pat_alim = st.text_area("Alimentation / Hypermarchés", value=DEFAULT_PATTERN_ALIM, height=150)
with col2:
    pat_anim = st.text_area("Animaux", value=DEFAULT_PATTERN_ANIM, height=150)
with col3:
    pat_carb = st.text_area("Carburant / Stations", value=DEFAULT_PATTERN_CARBURANT, height=150)

pat_cash = st.text_area("Retraits espèces (DAB/ATM)", value=DEFAULT_PATTERN_CASH, height=100)

# =========================
# --- WORKFLOW (7 étapes) -
# =========================
if uploaded is not None:
    # 1) Lecture robuste
    df_raw = try_read_csv(uploaded)

    # 2) Normalisation des noms + inférence de colonnes
    colmap_dict = infer_columns(df_raw)
    cmap = ColumnMap(**colmap_dict)

    # 3) Préparation DataFrame standard
    df = df_raw.copy()
    # libellé
    if cmap.label and cmap.label in df.columns:
        df["label"] = df[cmap.label].astype(str).map(normalize_str).str.lower()
    else:
        df["label"] = ""

    # date
    if cmap.date and cmap.date in df.columns:
        df["date"] = pd.to_datetime(df[cmap.date], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        # fallback : aucune date => NaT
        df["date"] = pd.NaT

    # 4) Montants signés
    df["amount_signed"] = coerce_amounts(df, cmap.amount, cmap.debit, cmap.credit).fillna(0.0)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # 5) Catégorisation automatique
    def label_matches(pat: str, text: str) -> bool:
        try:
            return re.search(pat, text or "", flags=re.IGNORECASE) is not None
        except re.error:
            return False

    # Catégorie par défaut
    df["category"] = "Autres"

    # Variables
    df.loc[df["label"].apply(lambda x: label_matches(pat_alim, x)), "category"] = "Alimentation"
    df.loc[df["label"].apply(lambda x: label_matches(pat_anim, x)), "category"] = "Animaux"
    df.loc[df["label"].apply(lambda x: label_matches(pat_carb, x)), "category"] = "Carburant"
    df.loc[df["label"].apply(lambda x: label_matches(pat_cash, x)), "category"] = "Retraits/Especes"

    # Fixes via providers
    try:
        providers = json.loads(providers_json)
        for p in providers:
            lab, rgx = p.get("Label"), p.get("Regex")
            if lab and rgx:
                df.loc[df["label"].apply(lambda x: label_matches(rgx, x)), "category"] = lab
    except Exception as e:
        st.warning(f"Impossible de parser les fournisseurs : {e}")

    # 6) Ajout des crédits récurrents (en tant que lignes synthétiques optionnelles)
    add_fixed = st.checkbox("Ajouter les crédits récurrents mensuels au budget (lignes synthétiques)", value=True)
    synth_rows = []
    if add_fixed:
        try:
            rec_credits = json.loads(credits_json)
            # On ajoute pour le mois visible (ou tous les mois présents)
            months_present = sorted(df["month"].dropna().unique().tolist())
            target_months = months_present or [datetime.now().strftime("%Y-%m")]
            for m in target_months:
                for c in rec_credits:
                    synth_rows.append(
                        dict(date=pd.Period(m).to_timestamp(how="start"),
                             label=normalize_str(c.get("label", "")),
                             amount_signed=float(c.get("amount", 0.0)),
                             year=int(m.split("-")[0]),
                             month=m,
                             category="Crédits fixes")
                    )
        except Exception as e:
            st.warning(f"Impossible de parser les crédits récurrents : {e}")

    if synth_rows:
        df = pd.concat([df, pd.DataFrame(synth_rows)], ignore_index=True)

    # 7) Tableaux, filtres, graphiques, export
    st.divider()
    st.subheader("📊 Filtres")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        years = ["(Tous)"] + [str(y) for y in sorted(df["year"].dropna().unique())]
        ypick = st.selectbox("Année", options=years, index=0)
    with colf2:
        months_all = ["(Tous)"] + sorted(df["month"].dropna().unique().tolist())
        mpick = st.selectbox("Mois (AAAA-MM)", options=months_all, index=0)
    with colf3:
        cats_all = ["(Toutes)"] + sorted(df["category"].dropna().unique().tolist())
        cpick = st.selectbox("Catégorie", options=cats_all, index=0)

    dfv = df.copy()
    if ypick != "(Tous)":
        dfv = dfv[dfv["year"] == int(ypick)]
    if mpick != "(Tous)":
        dfv = dfv[dfv["month"] == mpick]
    if cpick != "(Toutes)":
        dfv = dfv[dfv["category"] == cpick]

    st.subheader("🧾 Transactions (après catégorisation)")
    st.dataframe(
        dfv.sort_values(["date"], ascending=[False])[["date", "label", "category", "amount_signed"]],
        use_container_width=True,
        height=340
    )

    # Agrégations
    st.subheader("📈 Synthèse par mois et par catégorie")
    g_month = dfv.groupby("month", dropna=True)["amount_signed"].sum().reset_index()
    g_cat   = dfv.groupby("category", dropna=True)["amount_signed"].sum().reset_index()

    colm, colc = st.columns(2)
    with colm:
        st.markdown("**Solde par mois** (positif = plus d'entrées que de sorties)")
        chart_m = alt.Chart(g_month).mark_bar(color="#2C7BE5").encode(
            x=alt.X("month:O", title="Mois"),
            y=alt.Y("amount_signed:Q", title="Solde (€)"),
            tooltip=["month", alt.Tooltip("amount_signed:Q", title="Solde", format=",.2f")]
        )
        st.altair_chart(chart_m.properties(height=300), use_container_width=True)
        st.metric("Solde total (période filtrée)", f"{g_month['amount_signed'].sum():,.2f} €".replace(",", " "))

    with colc:
        st.markdown("**Dépenses/recettes par catégorie**")
        chart_c = alt.Chart(g_cat).mark_bar().encode(
            x=alt.X("category:N", sort="-y", title="Catégorie"),
            y=alt.Y("amount_signed:Q", title="Montant (€)"),
            color=alt.condition(
                alt.datum.amount_signed < 0,
                alt.value("#E63757"),  # dépenses
                alt.value("#00D97E")   # entrées
            ),
            tooltip=[alt.Tooltip("category:N", title="Catégorie"), alt.Tooltip("amount_signed:Q", title="Montant", format=",.2f")]
        )
        st.altair_chart(chart_c.properties(height=300), use_container_width=True)

    # Export XLSX / CSV du jeu filtré + mapping colonnes
    st.subheader("📤 Export")
    def to_csv_bytes(df_: pd.DataFrame) -> bytes:
        return df_.to_csv(index=False).encode("utf-8")

    export_cols = ["date", "label", "category", "amount_signed", "year", "month"]
    csv_bytes = to_csv_bytes(dfv[export_cols])

    st.download_button(
        "⬇️ Télécharger CSV (filtré)",
        data=csv_bytes,
        file_name=f"banque_filtre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Rappel colonnes détectées
    with st.expander("🧩 Colonnes détectées / mapping", expanded=False):
        st.json(colmap_dict, expanded=True)

else:
    st.info("Dépose ton CSV pour démarrer l’analyse.")
