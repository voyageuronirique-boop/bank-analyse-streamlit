# app.py
# Analyse bancaire CSV — Workflow complet (Streamlit)
# - CSV FR/EN robuste (séparateurs/encodages)
# - Catégorisation regex éditable + consolidation EDF/Sowee
# - Synthèse mensuelle (Revenus/Charges/Dépenses/Net/Soldes rolling)
# - Prévisionnel mensuel intelligent + graphiques d'évolution
# Auteur : adapté pour Jeremy Verhelst — Python 3.9+

from __future__ import annotations

import io
import re
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

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
# ----- CONSTANTES --------
# =========================
DEFAULT_RECURRING_CREDITS = [
    {"label": "Participation Jeremy",  "amount": 1150.0},
    {"label": "Participation Vanessa", "amount": 1050.0},
    {"label": "Participation Jeremy 2","amount": 530.0},
]

DEFAULT_PROVIDER_PATTERNS = [
    {"Label": "Crédit immobilier",                       "Regex": r"(echeance.*pret|pret|credit|hypothec|immobilier|echeance\s*de\s*credit)"},
    {"Label": "Assurance habitation / BPCE",             "Regex": r"(bpce\s+assurances?|multirisque|habitation)"},
    {"Label": "Assurance GENERALI IARD",                 "Regex": r"(generali\s+iard)"},
    {"Label": "Assurance GENERALI VIE",                  "Regex": r"(generali\s+vie)"},
    {"Label": "Freebox (Internet fixe)",                 "Regex": r"(free\s*telecom|freebox)"},
    {"Label": "Free Mobile",                             "Regex": r"(free\s*mobile)"},
    {"Label": "SFR (fixe/mobile)",                       "Regex": r"\bsfr\b"},
    {"Label": "Électricité — Sowee (EDF)",               "Regex": r"(sowee\s*by\s*edf|sowee)"},
    {"Label": "Électricité — EDF",                       "Regex": r"\bedf\b"},
    {"Label": "Électricité — Bellenergie / EdP",         "Regex": r"(bellenergie|electricit[eé]\s*de\s*provence)"},
    {"Label": "Eau (SEM / régies)",                      "Regex": r"(soc(i[eé]t[eé])?\s*des\s*eaux|eau|veolia|suez|saur)"},
    {"Label": "Abonnements streaming",                   "Regex": r"(netflix|spotify|deezer|prime|canal\+|molotov|youtube\s*premium)"},
    {"Label": "Frais bancaires",                         "Regex": r"(cotis(ations)?\s+bancaires|frais\s+bancaires)"},
]

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

DEFAULT_EXPECTED_FIXED = []  # optionnel : overrides des charges fixes attendues
# Exemple:
# [{"Label":"Crédit immobilier","amount":950.0,"freq":"M"}, {"Label":"Électricité — EDF/Sowee","amount":160.0,"freq":"M"}]


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
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def coerce_amounts(df: pd.DataFrame, amount_col, debit_col, credit_col) -> pd.Series:
    """
    Crée 'amount_signed' (négatif sorties, positif entrées).
    Priorité : si 'amount' existe => signe conservé ; sinon crédit - débit.
    """
    if amount_col and amount_col in df.columns:
        amt = df[amount_col].map(_to_num)
        return amt.fillna(0.0)

    deb = df[debit_col].map(_to_num).fillna(0.0) if (debit_col and debit_col in df.columns) else 0.0
    cre = df[credit_col].map(_to_num).fillna(0.0) if (credit_col and credit_col in df.columns) else 0.0
    return (cre - deb).fillna(0.0)

def label_matches(pat: str, text: str) -> bool:
    try:
        return re.search(pat, text or "", flags=re.IGNORECASE) is not None
    except re.error:
        return False


# =========================
# ----- SESSION STATE -----
# =========================
def ss_init(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("providers_json", json.dumps(DEFAULT_PROVIDER_PATTERNS, ensure_ascii=False, indent=2))
ss_init("credits_json", json.dumps(DEFAULT_RECURRING_CREDITS, ensure_ascii=False, indent=2))
ss_init("pat_alim", DEFAULT_PATTERN_ALIM)
ss_init("pat_anim", DEFAULT_PATTERN_ANIM)
ss_init("pat_carb", DEFAULT_PATTERN_CARBURANT)
ss_init("pat_cash", DEFAULT_PATTERN_CASH)
ss_init("expected_fixed_json", json.dumps(DEFAULT_EXPECTED_FIXED, ensure_ascii=False, indent=2))
ss_init("add_fixed_credits", True)


# =========================
# --------- UI ------------
# =========================
st.title("Analyse bancaire CSV — Budget mensuel + Prévisionnel")
st.caption("Charge un relevé CSV, catégorise automatiquement, puis visualise le réel + un prévisionnel intelligent.")

with st.expander("ℹ️ Comment préparer le CSV ?", expanded=False):
    st.markdown(
        "- Export natif de ta banque (CSV, séparateur `;`, `,` ou `\\t`).\n"
        "- Encodage UTF‑8 ou Latin‑1 supporté.\n"
        "- Colonnes attendues (si dispo) : **date**, **libellé**, **débit**, **crédit**, **montant**."
    )

uploaded = st.file_uploader("Dépose ton fichier CSV", type=["csv"])


# =========================
# --- WORKFLOW PRINCIPAL --
# =========================
if uploaded is None:
    st.info("Dépose ton CSV pour démarrer l’analyse.")
else:
    # 1) Lecture robuste
    df_raw = try_read_csv(uploaded)

    # 2) Mapping colonnes
    colmap_dict = infer_columns(df_raw)
    cmap = ColumnMap(**colmap_dict)

    # 3) DataFrame standard
    df = df_raw.copy()

    # libellé normalisé
    if cmap.label and cmap.label in df.columns:
        df["label"] = df[cmap.label].astype(str).map(normalize_str).str.lower()
    else:
        df["label"] = ""

    # date
    if cmap.date and cmap.date in df.columns:
        df["date"] = pd.to_datetime(df[cmap.date], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        df["date"] = pd.NaT

    # 4) Montants signés
    df["amount_signed"] = coerce_amounts(df, cmap.amount, cmap.debit, cmap.credit).fillna(0.0)

    # Si la date est manquante, on met un mois NaN. Sinon mois AAAA-MM.
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # 5) Catégorisation automatique (fixes d'abord, puis variables)
    df["category"] = "Autres"

    # --- Parse providers + consolidation EDF/Sowee ---
    try:
        providers = json.loads(st.session_state["providers_json"])
        if not isinstance(providers, list):
            providers = []
    except Exception:
        providers = []

    # Consolidation EDF/Sowee : on supprime entrées EDF/Sowee et on ajoute une entrée unique
    consolidated = []
    elec_added = False
    for p in providers:
        lab = (p.get("Label") or "").strip()
        rgx = (p.get("Regex") or "").strip()
        if not lab or not rgx:
            continue
        if ("edf" in lab.lower()) or ("sowee" in lab.lower()):
            if not elec_added:
                consolidated.append({"Label": "Électricité — EDF/Sowee", "Regex": r"(sowee|edf)"})
                elec_added = True
            continue
        consolidated.append({"Label": lab, "Regex": rgx})
    providers = consolidated

    provider_labels = [p.get("Label") for p in providers if p.get("Label")]

    # Fixes (first match wins)
    for p in providers:
        lab, rgx = p.get("Label"), p.get("Regex")
        if lab and rgx:
            mask = (df["category"] == "Autres") & df["label"].apply(lambda x: label_matches(rgx, x))
            df.loc[mask, "category"] = lab

    # Variables (uniquement si encore Autres)
    pat_alim = st.session_state["pat_alim"]
    pat_anim = st.session_state["pat_anim"]
    pat_carb = st.session_state["pat_carb"]
    pat_cash = st.session_state["pat_cash"]

    mask_autres = df["category"] == "Autres"
    df.loc[mask_autres & df["label"].apply(lambda x: label_matches(pat_alim, x)), "category"] = "Alimentation"
    df.loc[mask_autres & df["label"].apply(lambda x: label_matches(pat_anim, x)), "category"] = "Animaux"
    df.loc[mask_autres & df["label"].apply(lambda x: label_matches(pat_carb, x)), "category"] = "Carburant"
    df.loc[mask_autres & df["label"].apply(lambda x: label_matches(pat_cash, x)), "category"] = "Retraits/Especes"

    # 6) Ajout des crédits récurrents (lignes synthétiques optionnelles)
    add_fixed = st.checkbox("Ajouter les crédits récurrents mensuels (lignes synthétiques)", value=st.session_state["add_fixed_credits"])
    st.session_state["add_fixed_credits"] = add_fixed

    synth_rows = []
    if add_fixed:
        try:
            rec_credits = json.loads(st.session_state["credits_json"])
            months_present = sorted(df["month"].dropna().unique().tolist())
            target_months = months_present or [datetime.now().strftime("%Y-%m")]
            for m in target_months:
                for c in rec_credits:
                    synth_rows.append(
                        dict(
                            date=pd.Period(m).to_timestamp(how="start"),
                            label=normalize_str(c.get("label", "")),
                            amount_signed=float(c.get("amount", 0.0)),
                            year=int(m.split("-")[0]),
                            month=m,
                            category="Crédits fixes"
                        )
                    )
        except Exception as e:
            st.warning(f"Impossible de parser les crédits récurrents : {e}")

    if synth_rows:
        df = pd.concat([df, pd.DataFrame(synth_rows)], ignore_index=True)

    # =========================
    # ---- SOLDE ROLLING -----
    # =========================
    st.info(
        "ℹ️ **Important**\n\n"
        "Merci d’ajouter **le solde du mois précédent**, tel qu’il apparaît dans "
        "**« Consulter les relevés de comptes »** du compte bancaire concerné.\n\n"
        "Ce solde est utilisé comme point de départ pour calculer automatiquement les soldes mensuels."
    )

    seed_initial = st.number_input(
        "💡 Solde au début du mois précédent le premier mois du CSV",
        help="Exemple : si le CSV commence en février, saisir le solde réel au 01/01.",
        value=0.0,
        step=50.0
    )

    # =========================
    # ---- SYNTHESE MENSUELLE (RÉEL) ---
    # =========================
    df_calc = df.dropna(subset=["month"]).copy()

    # Flags flux
    is_income = df_calc["amount_signed"] > 0
    is_fixed_charge = df_calc["category"].isin(provider_labels) & (df_calc["amount_signed"] < 0)
    is_variable_expense = (df_calc["amount_signed"] < 0) & (~df_calc["category"].isin(provider_labels))

    months = sorted(df_calc["month"].dropna().unique().tolist())

    m_income = df_calc[is_income].groupby("month")["amount_signed"].sum()
    m_fixed  = df_calc[is_fixed_charge].groupby("month")["amount_signed"].sum()         # négatif
    m_var    = df_calc[is_variable_expense].groupby("month")["amount_signed"].sum()     # négatif
    m_net    = df_calc.groupby("month")["amount_signed"].sum()

    summary = pd.DataFrame({"month": months})
    summary["revenus"] = summary["month"].map(m_income).fillna(0.0)
    summary["charges_fixes"] = summary["month"].map(m_fixed).fillna(0.0)               # négatif
    summary["depenses_variables"] = summary["month"].map(m_var).fillna(0.0)            # négatif
    summary["net_mois"] = summary["month"].map(m_net).fillna(0.0)

    # Rolling soldes
    summary["solde_debut"] = 0.0
    summary["solde_fin"] = 0.0
    running = float(seed_initial)
    for i in range(len(summary)):
        summary.loc[i, "solde_debut"] = running
        running = running + float(summary.loc[i, "net_mois"])
        summary.loc[i, "solde_fin"] = running

    # =========================
    # ---- AFFICHAGE RAPIDE ---
    # =========================
    st.subheader("💶 Synthèse mensuelle (lecture rapide) — Réel")
    colk1, colk2, colk3, colk4 = st.columns(4)

    colk1.metric("Revenus (total)", f"{summary['revenus'].sum():,.2f} €".replace(",", " "))
    colk2.metric("Charges fixes (total)", f"{summary['charges_fixes'].sum():,.2f} €".replace(",", " "))
    colk3.metric("Dépenses variables (total)", f"{summary['depenses_variables'].sum():,.2f} €".replace(",", " "))
    colk4.metric("Solde final (rolling)", f"{summary['solde_fin'].iloc[-1]:,.2f} €".replace(",", " ") if len(summary) else f"{seed_initial:,.2f} €".replace(",", " "))

    st.dataframe(
        summary[["month","revenus","charges_fixes","depenses_variables","net_mois","solde_debut","solde_fin"]],
        use_container_width=True,
        height=260
    )

    # =========================
    # ---- PREVISIONNEL INTELLIGENT ---
    # =========================
    st.divider()
    st.subheader("🔮 Prévisionnel mensuel (intelligent)")

    colp1, colp2, colp3 = st.columns([1, 1, 2])
    with colp1:
        horizon = st.slider("Horizon (mois)", min_value=1, max_value=24, value=6)
    with colp2:
        method_var = st.selectbox("Méthode dépenses variables", ["Médiane (robuste)", "Moyenne glissante"], index=0)
    with colp3:
        window = st.slider("Historique utilisé (mois)", min_value=1, max_value=12, value=3)

    expected_fixed_json = st.text_area(
        "⚙️ (Optionnel) Charges fixes attendues (JSON) — sinon estimation automatique (médiane historique)",
        value=st.session_state["expected_fixed_json"],
        height=140
    )
    st.session_state["expected_fixed_json"] = expected_fixed_json

    # 1) Revenus récurrents (mensuels)
    try:
        rec_credits = json.loads(st.session_state["credits_json"])
        monthly_rec_income = sum(float(c.get("amount", 0.0)) for c in rec_credits)
    except Exception:
        monthly_rec_income = 0.0

    # 2) Overrides charges fixes (facultatif)
    expected_fixed = {}
    try:
        tmp = json.loads(expected_fixed_json)
        if isinstance(tmp, list):
            for item in tmp:
                lab = (item.get("Label") or "").strip()
                amt = float(item.get("amount", 0.0))
                freq = (item.get("freq") or "M").strip()
                if lab:
                    expected_fixed[lab] = {"amount": amt, "freq": freq}
    except Exception:
        expected_fixed = {}

    # 3) Historique utilisé (dernier window mois)
    hist_months = summary["month"].tolist()
    hist_tail = hist_months[-window:] if len(hist_months) >= 1 else hist_months
    df_hist = df_calc[df_calc["month"].isin(hist_tail)].copy()

    # 4) Estimation auto des charges fixes (médiane des ABS par catégorie fixe)
    auto_fixed = {}
    for lab in provider_labels:
        vals = df_hist[(df_hist["category"] == lab) & (df_hist["amount_signed"] < 0)]["amount_signed"].abs()
        if len(vals) > 0:
            auto_fixed[lab] = float(vals.median())

    # 5) Estimation dépenses variables mensuelles (sur totaux variables/mois)
    df_hist["is_var"] = (df_hist["amount_signed"] < 0) & (~df_hist["category"].isin(provider_labels))
    var_by_month = df_hist[df_hist["is_var"]].groupby("month")["amount_signed"].sum().abs()
    if len(var_by_month) == 0:
        est_var = 0.0
    else:
        est_var = float(var_by_month.median()) if method_var.startswith("Médiane") else float(var_by_month.mean())

    # 6) Mois futurs
    if len(summary) > 0:
        last_month = pd.Period(summary["month"].iloc[-1], freq="M")
    else:
        last_month = pd.Period(datetime.now().strftime("%Y-%m"), freq="M")

    future_months = [(last_month + i).strftime("%Y-%m") for i in range(1, horizon + 1)]

    # 7) Charges fixes prévues (somme des catégories fixes)
    def fixed_total_for_month() -> float:
        total = 0.0
        for lab in provider_labels:
            if lab in expected_fixed:
                # fréquence simple: M (mensuel). Extensions possibles ensuite.
                total += float(expected_fixed[lab]["amount"])
            else:
                total += float(auto_fixed.get(lab, 0.0))
        return total

    fixed_total = fixed_total_for_month()

    forecast = pd.DataFrame({"month": future_months})
    forecast["revenus_prevus"] = float(monthly_rec_income)
    forecast["charges_fixes_prevues"] = float(fixed_total)
    forecast["depenses_variables_prevues"] = float(est_var)

    forecast["net_prev"] = forecast["revenus_prevus"] - forecast["charges_fixes_prevues"] - forecast["depenses_variables_prevues"]

    # Rolling solde sur prévisionnel depuis dernier solde réel
    start_balance = float(summary["solde_fin"].iloc[-1]) if len(summary) else float(seed_initial)
    forecast["solde_debut"] = 0.0
    forecast["solde_fin"] = 0.0
    running = start_balance
    for i in range(len(forecast)):
        forecast.loc[i, "solde_debut"] = running
        running = running + float(forecast.loc[i, "net_prev"])
        forecast.loc[i, "solde_fin"] = running

    # =========================
    # ---- REEL vs PREVU TABLE ---
    # =========================
    st.subheader("📅 Réel vs Prévisionnel (mensuel)")

    real_view = summary.copy()
    real_view = real_view.rename(columns={
        "revenus": "revenus_prevus",
        "charges_fixes": "charges_fixes_prevues",
        "depenses_variables": "depenses_variables_prevues",
        "net_mois": "net_prev",
    })
    real_view["type"] = "Réel"

    forecast_view = forecast.copy()
    forecast_view["type"] = "Prévisionnel"

    rv = pd.concat(
        [
            real_view[["month","type","revenus_prevus","charges_fixes_prevues","depenses_variables_prevues","net_prev","solde_debut","solde_fin"]],
            forecast_view[["month","type","revenus_prevus","charges_fixes_prevues","depenses_variables_prevues","net_prev","solde_debut","solde_fin"]],
        ],
        ignore_index=True
    )

    st.dataframe(rv, use_container_width=True, height=320)

    # =========================
    # ---- GRAPHIQUES EVOLUTION ---
    # =========================
    st.subheader("📈 Évolution du solde (réel + prévisionnel)")

    chart_balance = alt.Chart(rv).mark_line(point=True).encode(
        x=alt.X("month:O", title="Mois"),
        y=alt.Y("solde_fin:Q", title="Solde fin (€)"),
        color=alt.Color("type:N", title=""),
        tooltip=[
            alt.Tooltip("month:O", title="Mois"),
            alt.Tooltip("type:N", title="Type"),
            alt.Tooltip("solde_fin:Q", title="Solde fin", format=",.2f"),
            alt.Tooltip("net_prev:Q", title="Net", format=",.2f"),
        ]
    ).properties(height=320)

    st.altair_chart(chart_balance, use_container_width=True)

    st.subheader("📊 Net mensuel (réel + prévisionnel)")
    chart_net = alt.Chart(rv).mark_bar().encode(
        x=alt.X("month:O", title="Mois"),
        y=alt.Y("net_prev:Q", title="Net (€)"),
        color=alt.condition(
            alt.datum.net_prev < 0,
            alt.value("#E63757"),
            alt.value("#00D97E")
        ),
        tooltip=[
            alt.Tooltip("month:O", title="Mois"),
            alt.Tooltip("type:N", title="Type"),
            alt.Tooltip("net_prev:Q", title="Net", format=",.2f"),
        ]
    ).properties(height=220)

    st.altair_chart(chart_net, use_container_width=True)

    # =========================
    # ---- FILTRES + TABLE TRANSACTIONS ---
    # =========================
    st.divider()
    st.subheader("📊 Filtres (transactions)")

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

    # =========================
    # ---- EXPORTS ---
    # =========================
    st.subheader("📤 Exports")

    def to_csv_bytes(df_: pd.DataFrame) -> bytes:
        return df_.to_csv(index=False).encode("utf-8")

    export_cols = ["date", "label", "category", "amount_signed", "year", "month"]
    st.download_button(
        "⬇️ Télécharger CSV (transactions filtrées)",
        data=to_csv_bytes(dfv[export_cols]),
        file_name=f"banque_filtre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Télécharger CSV (synthèse réel + prévisionnel)",
        data=to_csv_bytes(rv),
        file_name=f"budget_reel_prevu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    with st.expander("🧩 Colonnes détectées / mapping", expanded=False):
        st.json(colmap_dict, expanded=True)

    # =========================
    # ---- PARAMETRES AVANCES (EN BAS) ---
    # =========================
    st.divider()
    with st.expander("⚙️ Paramètres avancés (catégories / fournisseurs / crédits) — à modifier si besoin", expanded=False):
        st.caption("Les modifications sont sauvegardées et appliquées au prochain recalcul automatique (Streamlit relance le script).")

        with st.form("advanced_params_form", clear_on_submit=False):
            colA, colB = st.columns([1, 1])
            with colA:
                st.subheader("Patrons fournisseurs (fixes)")
                providers_json_new = st.text_area(
                    "Liste JSON de fournisseurs (Label/Regex)",
                    value=st.session_state["providers_json"],
                    height=220
                )
            with colB:
                st.subheader("Crédits récurrents (entrées fixes)")
                credits_json_new = st.text_area(
                    "Liste JSON de crédits (label/amount)",
                    value=st.session_state["credits_json"],
                    height=220
                )

            st.subheader("Catégories variables (Regex)")
            col1, col2, col3 = st.columns(3)
            with col1:
                pat_alim_new = st.text_area("Alimentation / Hypermarchés", value=st.session_state["pat_alim"], height=150)
            with col2:
                pat_anim_new = st.text_area("Animaux", value=st.session_state["pat_anim"], height=150)
            with col3:
                pat_carb_new = st.text_area("Carburant / Stations", value=st.session_state["pat_carb"], height=150)

            pat_cash_new = st.text_area("Retraits espèces (DAB/ATM)", value=st.session_state["pat_cash"], height=110)

            submitted = st.form_submit_button("💾 Enregistrer les paramètres")
            if submitted:
                st.session_state["providers_json"] = providers_json_new
                st.session_state["credits_json"] = credits_json_new
                st.session_state["pat_alim"] = pat_alim_new
                st.session_state["pat_anim"] = pat_anim_new
                st.session_state["pat_carb"] = pat_carb_new
                st.session_state["pat_cash"] = pat_cash_new
                st.success("Paramètres enregistrés ✅ (l'application se recalculera automatiquement).")
