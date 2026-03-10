"""
================================================================================
PROJETO PORTFÓLIO — Análise de Gênero na Câmara dos Deputados
================================================================================

OBJETIVO:
    1) Proporção de homens e mulheres por legislatura
    2) Taxa de reeleição (proxy) por gênero — com IC 95% e N
    3) Regressão logística (chance de continuidade por gênero)
    4) Regressão linear (projeção de % de mulheres)

ENTREGÁVEIS:
    1) CSV long:                data/processed/deputados_long_limpo.csv
    2) KPI (leg × gênero):     data/processed/kpi_leg_genero.csv
    3) Gráficos (4):
       - grafico_representatividade_pctF.png
       - grafico_reeleicao_proxy_taxa.png
       - grafico_regressao_logistica.png   (forest plot)
       - grafico_projecao_pctF.png
    4) Insights:                data/processed/insights.md

ENTRADA:
    - deputados.csv (separador ';')

REQUISITOS:
    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

NOTAS METODOLÓGICAS:
    - "Reeleição" = proxy: presença consecutiva (L-1 → L).
    - Cadeiras: 487 (leg 48), 503 (leg 49), 513 (leg 50+).
    - IC de Wilson (95%) para taxas com N pequeno.
    - Base de transição com zeros (quem saiu) para regressão.
================================================================================
"""

# ============================================================
# PASSO 0 — IMPORTS
# ============================================================
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# PASSO 1 — CONFIGURAÇÃO
# ============================================================
INPUT_CSV = "deputados.csv"
SEP = ";"

ANALISE_A_PARTIR_DA_LEG = 48

CADEIRAS_POR_LEG = {
    48: 487, 49: 503,
    50: 513, 51: 513, 52: 513, 53: 513,
    54: 513, 55: 513, 56: 513, 57: 513,
}

PROJETAR_N_LEGS = 3

OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_LONG     = os.path.join(OUT_DIR, "deputados_long_limpo.csv")
OUT_KPI      = os.path.join(OUT_DIR, "kpi_leg_genero.csv")
OUT_INSIGHTS = os.path.join(OUT_DIR, "insights.md")

GENERO_MISSING = "NI"

COLS_DROP = [
    "cpf", "urlRedeSocial", "ufNascimento", "municipioNascimento",
]

COL_RENAME = {
    "nome":                  "nome_parlamentar",
    "siglaSexo":             "genero",
    "idLegislaturaInicial":  "leg_inicio",
    "idLegislaturaFinal":    "leg_fim",
}

# Cores padrão
PINK = "#C2185B"
BLUE = "#1565C0"
GRAY = "#888888"


# ============================================================
# PASSO 2 — FUNÇÕES AUXILIARES
# ============================================================
def extrair_id(uri):
    """Extrai o ID numérico do final da URI."""
    if pd.isna(uri):
        return pd.NA
    m = re.search(r"(\d+)$", str(uri).strip())
    return int(m.group(1)) if m else pd.NA


def explode_legislaturas(df_base):
    """1 linha por deputado → 1 linha por deputado por legislatura."""
    leg_min = int(df_base["leg_inicio"].min())
    leg_max = int(df_base["leg_fim"].max())

    rows = []
    for _, r in df_base.iterrows():
        if pd.isna(r["id_deputado"]) or pd.isna(r["leg_inicio"]) or pd.isna(r["leg_fim"]):
            continue
        li, lf = int(r["leg_inicio"]), int(r["leg_fim"])
        for leg in range(leg_min, leg_max + 1):
            if li <= leg <= lf:
                new = r.copy()
                new["legislatura"] = leg
                rows.append(new)

    return pd.DataFrame(rows)


def classificar_reeleicao(df_long):
    """
    reeleito (proxy): apareceu em L e em L-1
    estreante: primeira aparição no histórico
    """
    df = df_long.copy().sort_values(["id_deputado", "legislatura"])
    diff = df.groupby("id_deputado")["legislatura"].diff()
    df["reeleito"] = diff.eq(1)
    df["estreante"] = df.groupby("id_deputado").cumcount().eq(0)
    return df


def get_cadeiras(leg):
    return CADEIRAS_POR_LEG.get(int(leg), np.nan)


def wilson_ci(sucessos, n, alpha=0.05):
    """IC de Wilson para proporção (robusto para N pequeno)."""
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(sucessos, n, alpha=alpha, method="wilson")
    return (lo * 100, hi * 100)


def build_kpi(df_long_all, leg_min):
    """KPI por legislatura × gênero com IC 95% e N."""
    df = df_long_all.copy()
    df_a = df[df["legislatura"] >= leg_min].copy()

    # Contagem e %
    base = (
        df_a.groupby(["legislatura", "genero"])["id_deputado"]
        .nunique()
        .rename("deputados_observados")
        .reset_index()
    )
    total_obs = (
        df_a.groupby("legislatura")["id_deputado"]
        .nunique()
        .rename("total_observado")
        .reset_index()
    )
    base = base.merge(total_obs, on="legislatura")
    base["cadeiras_oficiais"] = base["legislatura"].apply(get_cadeiras)
    base["pct_genero"] = base["deputados_observados"] / base["total_observado"] * 100
    base["pct_sobre_cadeiras"] = np.where(
        base["cadeiras_oficiais"].notna(),
        base["deputados_observados"] / base["cadeiras_oficiais"] * 100,
        np.nan,
    )

    # Taxa de reeleição proxy + IC
    present = df[["id_deputado", "genero", "legislatura"]].drop_duplicates()

    next_leg = present[["id_deputado", "legislatura"]].copy()
    next_leg["legislatura"] = next_leg["legislatura"] - 1
    next_leg["continua"] = 1

    trans = present.merge(next_leg, on=["id_deputado", "legislatura"], how="left")
    trans["continua"] = trans["continua"].fillna(0).astype(int)
    trans["leg_destino"] = trans["legislatura"] + 1

    trans_a = trans[trans["leg_destino"] >= leg_min].copy()

    reel = (
        trans_a.groupby(["leg_destino", "genero"])
        .agg(
            n_elegiveis=("id_deputado", "nunique"),
            n_continuaram=("continua", "sum"),
        )
        .reset_index()
        .rename(columns={"leg_destino": "legislatura"})
    )
    reel["taxa_reeleicao_proxy"] = np.where(
        reel["n_elegiveis"] > 0,
        reel["n_continuaram"] / reel["n_elegiveis"] * 100,
        np.nan,
    )

    ic = reel.apply(
        lambda row: wilson_ci(int(row["n_continuaram"]), int(row["n_elegiveis"])),
        axis=1,
        result_type="expand",
    )
    ic.columns = ["ic_lower", "ic_upper"]
    reel = pd.concat([reel, ic], axis=1)

    out = (
        base.merge(
            reel[["legislatura", "genero", "n_elegiveis", "n_continuaram",
                  "taxa_reeleicao_proxy", "ic_lower", "ic_upper"]],
            on=["legislatura", "genero"],
            how="left",
        )
        .sort_values(["legislatura", "genero"])
    )
    return out


def format_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.1f}%"


def get_kpi_value(kpi_df, leg, genero, col):
    s = kpi_df[(kpi_df["legislatura"] == leg) & (kpi_df["genero"] == genero)][col]
    return float(s.iloc[0]) if len(s) else np.nan


# ============================================================
# PASSO 3 — CARGA E LIMPEZA
# ============================================================
print("=" * 70)
print("PASSO 3: Carregando e limpando dados...")
print("=" * 70)

df_raw = pd.read_csv(INPUT_CSV, sep=SEP)
print(f"  Linhas brutas: {len(df_raw):,}")

df = df_raw.drop(columns=COLS_DROP, errors="ignore").copy()

df["id_deputado"] = df["uri"].apply(extrair_id).astype("Int64")
df["idLegislaturaInicial"] = pd.to_numeric(df.get("idLegislaturaInicial"), errors="coerce").astype("Int64")
df["idLegislaturaFinal"] = pd.to_numeric(df.get("idLegislaturaFinal"), errors="coerce").astype("Int64")

if "siglaSexo" in df.columns:
    df["siglaSexo"] = df["siglaSexo"].fillna(GENERO_MISSING)
else:
    df["siglaSexo"] = GENERO_MISSING

df = df.rename(columns=COL_RENAME)

print(f"  Deputados únicos: {df['id_deputado'].nunique(dropna=True):,}")
print(f"  Legislaturas: {int(df['leg_inicio'].min())} a {int(df['leg_fim'].max())}")
print()


# ============================================================
# PASSO 4 — FORMATO LONG + REELEIÇÃO
# ============================================================
print("=" * 70)
print("PASSO 4: Formato long + classificação de reeleição...")
print("=" * 70)

df_long_all = explode_legislaturas(df)
df_long_all = classificar_reeleicao(df_long_all)

print(f"  Linhas (completo): {len(df_long_all):,}")
print()


# ============================================================
# PASSO 5 — RECORTE >= 48 + CSV
# ============================================================
print("=" * 70)
print(f"PASSO 5: Recorte >= {ANALISE_A_PARTIR_DA_LEG}...")
print("=" * 70)

df_long = df_long_all[df_long_all["legislatura"] >= ANALISE_A_PARTIR_DA_LEG].copy()

cols_save = [
    "id_deputado", "nome_parlamentar", "uri", "genero",
    "leg_inicio", "leg_fim", "legislatura", "reeleito", "estreante",
]
cols_save = [c for c in cols_save if c in df_long.columns]
df_long[cols_save].to_csv(OUT_LONG, index=False, encoding="utf-8")
print(f"  Salvo: {OUT_LONG} ({len(df_long):,} linhas)")
print()


# ============================================================
# PASSO 6 — KPI
# ============================================================
print("=" * 70)
print("PASSO 6: KPIs...")
print("=" * 70)

kpi = build_kpi(df_long_all, ANALISE_A_PARTIR_DA_LEG)
kpi.to_csv(OUT_KPI, index=False, encoding="utf-8")
print(f"  Salvo: {OUT_KPI}")
print()


# ============================================================
# PASSO 7 — GRÁFICOS DESCRITIVOS (2)
# ============================================================
print("=" * 70)
print("PASSO 7: Gráficos descritivos...")
print("=" * 70)

sns.set_theme(style="whitegrid")

# --- Gráfico 1: Representatividade feminina (com anotações e marcos) ---
kpi_f = kpi[kpi["genero"] == "F"].copy()
col_pct = "pct_sobre_cadeiras" if kpi_f["pct_sobre_cadeiras"].notna().any() else "pct_genero"

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    kpi_f["legislatura"], kpi_f[col_pct],
    "o-", color=PINK, linewidth=2, markersize=8, zorder=5,
)

# Anota cada ponto com o valor
for _, row in kpi_f.iterrows():
    val = row[col_pct]
    if not pd.isna(val):
        ax.annotate(
            f"{val:.1f}%",
            xy=(row["legislatura"], val),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold", color=PINK,
        )

# Marcos legislativos
marcos = {
    50: "Lei 9.504/97\n(cota 30% candidaturas)",
    55: "Emenda 2015\n(reserva fundo partidário)",
}
for leg, texto in marcos.items():
    if leg in kpi_f["legislatura"].values:
        ax.axvline(x=leg, color=GRAY, linestyle=":", alpha=0.5)
        y_max = ax.get_ylim()[1]
        ax.annotate(
            texto, xy=(leg, y_max * 0.90),
            fontsize=7, color=GRAY, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

ax.set_title(
    "Representatividade feminina na Câmara dos Deputados\n"
    "Legislaturas 48 a 57 (1987–2027)",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Legislatura")
ax.set_ylabel("% de deputadas sobre o total de cadeiras")
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "grafico_representatividade_pctF.png"), dpi=150)
plt.close()
print("  OK: grafico_representatividade_pctF.png")


# --- Gráfico 2: Taxa de reeleição com IC + tabela de N ---
kpi_mf = kpi[kpi["genero"].isin(["M", "F"])].copy()

fig, ax = plt.subplots(figsize=(14, 7))

for genero, color, alpha_ic in [("M", BLUE, 0.10), ("F", PINK, 0.20)]:
    grp = kpi_mf[kpi_mf["genero"] == genero].sort_values("legislatura")
    legs = grp["legislatura"].values
    taxa = grp["taxa_reeleicao_proxy"].values
    lo = grp["ic_lower"].values
    hi = grp["ic_upper"].values

    ax.plot(legs, taxa, "o-", color=color, linewidth=2, markersize=7, label=f"{genero}", zorder=5)
    ax.fill_between(legs, lo, hi, alpha=alpha_ic, color=color, label=f"{genero} (IC 95%)")

# Tabela de N abaixo do gráfico
legs_unique = sorted(kpi_mf["legislatura"].unique())
n_m = []
n_f = []
for leg in legs_unique:
    nm = kpi_mf[(kpi_mf["legislatura"] == leg) & (kpi_mf["genero"] == "M")]["n_elegiveis"]
    nf = kpi_mf[(kpi_mf["legislatura"] == leg) & (kpi_mf["genero"] == "F")]["n_elegiveis"]
    n_m.append(f"{int(nm.iloc[0]):,}" if len(nm) and not pd.isna(nm.iloc[0]) else "–")
    n_f.append(f"{int(nf.iloc[0]):,}" if len(nf) and not pd.isna(nf.iloc[0]) else "–")

fig.subplots_adjust(bottom=0.22)

table = ax.table(
    cellText=[n_m, n_f],
    rowLabels=["N (M)", "N (F)"],
    colLabels=[str(int(l)) for l in legs_unique],
    loc="bottom",
    cellLoc="center",
    bbox=[0.0, -0.28, 1.0, 0.15],
)
table.auto_set_font_size(False)
table.set_fontsize(8)

# Colore labels das linhas da tabela
for i, color in enumerate([BLUE, PINK]):
    cell = table[i + 1, -1]
    cell.set_text_props(color=color, fontweight="bold")

ax.set_title(
    "Taxa de reeleição (proxy) por gênero — com IC 95%\n"
    "Legislaturas 48 a 57 (1987–2027)",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("")
ax.set_ylabel("% dos deputados em L-1 que continuam em L")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", framealpha=0.9)

fig.savefig(os.path.join(OUT_DIR, "grafico_reeleicao_proxy_taxa.png"), dpi=150)
plt.close()
print("  OK: grafico_reeleicao_proxy_taxa.png")
print()


# ============================================================
# PASSO 8 — REGRESSÃO LOGÍSTICA (base de transição correta)
# ============================================================
print("=" * 70)
print("PASSO 8: Regressão logística...")
print("=" * 70)

# ---- 8.1 Construir base de transição ----
present = df_long_all[["id_deputado", "genero", "legislatura"]].drop_duplicates()

next_leg = present[["id_deputado", "legislatura"]].copy()
next_leg["legislatura"] = next_leg["legislatura"] - 1
next_leg["continuou"] = 1

df_trans = present.merge(
    next_leg,
    on=["id_deputado", "legislatura"],
    how="left",
)
df_trans["continuou"] = df_trans["continuou"].fillna(0).astype(int)

leg_max = int(df_trans["legislatura"].max())

df_trans = df_trans[
    (df_trans["legislatura"] >= ANALISE_A_PARTIR_DA_LEG)
    & (df_trans["legislatura"] < leg_max)
    & (df_trans["genero"].isin(["M", "F"]))
].copy()

df_trans["genero_F"] = (df_trans["genero"] == "F").astype(int)

# ---- 8.2 Diagnóstico ----
print("\nDIAGNÓSTICO DA BASE DE TRANSIÇÃO:")
print(f"  Observações: {len(df_trans):,}")
print(f"  Continuou (1): {df_trans['continuou'].sum():,} ({df_trans['continuou'].mean()*100:.1f}%)")
print(f"  Saiu (0):      {(df_trans['continuou'] == 0).sum():,} ({(1 - df_trans['continuou'].mean())*100:.1f}%)")
print()
print("  Por gênero:")
diag = df_trans.groupby("genero").agg(
    n=("continuou", "count"),
    continuaram=("continuou", "sum"),
    taxa=("continuou", "mean"),
)
diag["taxa"] = (diag["taxa"] * 100).round(1)
print(diag.to_string())
print()

# ---- 8.3 Regressão logística ----
X = df_trans[["genero_F", "legislatura"]].copy()
X = sm.add_constant(X)
y = df_trans["continuou"]

modelo_logit = sm.Logit(y, X).fit(disp=False)

print(modelo_logit.summary2())
print()

# Odds Ratios + IC 95%
odds_ratios = np.exp(modelo_logit.params)
ci = modelo_logit.conf_int()
ci_or = np.exp(ci)
ci_or.columns = ["OR_lower", "OR_upper"]

print("Odds Ratios (com IC 95%):")
for var in odds_ratios.index:
    print(
        f"  {var}: OR = {odds_ratios[var]:.4f}  "
        f"[{ci_or.loc[var, 'OR_lower']:.4f} – {ci_or.loc[var, 'OR_upper']:.4f}]  "
        f"p = {modelo_logit.pvalues[var]:.4f}"
    )

or_genero = odds_ratios["genero_F"]
p_genero = modelo_logit.pvalues["genero_F"]
or_lo = ci_or.loc["genero_F", "OR_lower"]
or_hi = ci_or.loc["genero_F", "OR_upper"]

# ---- 8.4 Interpretação automática ----
if p_genero < 0.05:
    sig = "significativo (p < 0.05)"
    if or_genero > 1:
        direcao = f"mulheres têm {((or_genero - 1) * 100):.1f}% mais chance de continuar"
    else:
        direcao = f"mulheres têm {((1 - or_genero) * 100):.1f}% menos chance de continuar"
else:
    sig = "NÃO significativo (p >= 0.05)"
    direcao = (
        "não há diferença estatisticamente significativa entre gêneros na continuidade. "
        "Provável causa: N reduzido de mulheres e/ou ausência de variáveis de controle"
    )

print(f"\n  >>> OR(F) = {or_genero:.3f} [{or_lo:.3f} – {or_hi:.3f}]")
print(f"  >>> {direcao}")
print(f"  >>> {sig}")
print()

# ---- 8.5 Gráfico 3: Forest plot do Odds Ratio ----
fig, ax = plt.subplots(figsize=(10, 4))

vars_plot = ["genero_F", "legislatura"]
labels_plot = ["Gênero (F vs M)", "Legislatura (+1)"]
colors_plot = [PINK, BLUE]

for i, (var, label, color) in enumerate(zip(vars_plot, labels_plot, colors_plot)):
    or_val = odds_ratios[var]
    lo = ci_or.loc[var, "OR_lower"]
    hi = ci_or.loc[var, "OR_upper"]
    p = modelo_logit.pvalues[var]

    ax.errorbar(
        or_val, i, xerr=[[or_val - lo], [hi - or_val]],
        fmt="o", color=color, markersize=10, capsize=5, linewidth=2,
    )

    sig_marker = " *" if p < 0.05 else ""
    ax.annotate(
        f"OR = {or_val:.3f} [{lo:.3f}–{hi:.3f}]  p = {p:.3f}{sig_marker}",
        xy=(hi + 0.01, i),
        fontsize=9, va="center",
    )

ax.axvline(x=1, color="red", linestyle="--", alpha=0.7, label="Sem efeito (OR=1)")

ax.set_yticks(range(len(labels_plot)))
ax.set_yticklabels(labels_plot, fontsize=11)
ax.set_xlabel("Odds Ratio (IC 95%)")
ax.set_title(
    "Regressão Logística — Fatores associados à continuidade\n"
    "(* = significativo a 5%)",
    fontsize=12, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "grafico_regressao_logistica.png"), dpi=150)
plt.close()
print("  OK: grafico_regressao_logistica.png (forest plot)")
print()


# ============================================================
# PASSO 9 — REGRESSÃO LINEAR (projeção %F com IC)
# ============================================================
print("=" * 70)
print("PASSO 9: Regressão linear (projeção %F)...")
print("=" * 70)

kpi_f_reg = kpi_f[["legislatura", col_pct]].dropna().copy()
kpi_f_reg = kpi_f_reg.rename(columns={col_pct: "pct_F"})

X_lin = kpi_f_reg[["legislatura"]].values
y_lin = kpi_f_reg["pct_F"].values

modelo_linear = LinearRegression().fit(X_lin, y_lin)

r2 = modelo_linear.score(X_lin, y_lin)
coef = modelo_linear.coef_[0]
intercept = modelo_linear.intercept_

print(f"  Slope: {coef:.3f} p.p./legislatura")
print(f"  R²: {r2:.4f}")

leg_max_obs = int(kpi_f_reg["legislatura"].max())
legs_futuras = np.arange(leg_max_obs + 1, leg_max_obs + 1 + PROJETAR_N_LEGS)
pct_projetado = modelo_linear.predict(legs_futuras.reshape(-1, 1))

print("  Projeções:")
for leg, pct in zip(legs_futuras, pct_projetado):
    print(f"    Legislatura {leg}: {pct:.1f}%")
print()

# IC simples da regressão (baseado nos resíduos)
y_pred_train = modelo_linear.predict(X_lin)
residuos = y_lin - y_pred_train
se = np.std(residuos)

legs_line = np.arange(int(kpi_f_reg["legislatura"].min()), int(legs_futuras.max()) + 1)
pct_line = modelo_linear.predict(legs_line.reshape(-1, 1))

# --- Gráfico 4: Tendência + projeção com IC ---
fig, ax = plt.subplots(figsize=(13, 6))

# Faixa de incerteza (±1.96 * SE)
ax.fill_between(
    legs_line, pct_line - 1.96 * se, pct_line + 1.96 * se,
    alpha=0.12, color=GRAY, label="IC 95% da tendência",
)

# Tendência
ax.plot(legs_line, pct_line, "--", color=GRAY, linewidth=1.5, label=f"Tendência (R²={r2:.2f})")

# Observado
ax.plot(
    kpi_f_reg["legislatura"], kpi_f_reg["pct_F"],
    "o-", color=PINK, linewidth=2, markersize=8, label="Observado", zorder=5,
)

# Destaque do último ponto observado
last_obs = kpi_f_reg.iloc[-1]
ax.annotate(
    f"Última obs: {last_obs['pct_F']:.1f}%",
    xy=(last_obs["legislatura"], last_obs["pct_F"]),
    xytext=(-80, 15), textcoords="offset points",
    fontsize=9, color=PINK, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=PINK),
)

# Projeção
ax.plot(legs_futuras, pct_projetado, "s", color=BLUE, markersize=10, label="Projeção", zorder=5)

for leg, pct in zip(legs_futuras, pct_projetado):
    ax.annotate(
        f"{pct:.1f}%", xy=(leg, pct), xytext=(0, 14),
        textcoords="offset points", ha="center", fontsize=10,
        color=BLUE, fontweight="bold",
    )

ax.set_title(
    "Projeção da representatividade feminina (%F) — Regressão Linear\n"
    "Tendência média: a projeção assume ritmo constante de crescimento",
    fontsize=12, fontweight="bold",
)
ax.set_xlabel("Legislatura")
ax.set_ylabel("% de deputadas")
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.legend(loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "grafico_projecao_pctF.png"), dpi=150)
plt.close()
print("  OK: grafico_projecao_pctF.png")
print()


# ============================================================
# PASSO 10 — INSIGHTS
# ============================================================
print("=" * 70)
print("PASSO 10: Insights...")
print("=" * 70)

legs_recorte = sorted(df_long["legislatura"].unique())
leg_first = int(legs_recorte[0]) if legs_recorte else None
leg_last  = int(legs_recorte[-1]) if legs_recorte else None

pctF_first = get_kpi_value(kpi, leg_first, "F", col_pct) if leg_first else np.nan
pctF_last  = get_kpi_value(kpi, leg_last,  "F", col_pct) if leg_last  else np.nan
delta_pctF = (pctF_last - pctF_first) if (not pd.isna(pctF_first) and not pd.isna(pctF_last)) else np.nan

txM = get_kpi_value(kpi, leg_last, "M", "taxa_reeleicao_proxy") if leg_last else np.nan
txF = get_kpi_value(kpi, leg_last, "F", "taxa_reeleicao_proxy") if leg_last else np.nan
gap = (txM - txF) if (not pd.isna(txM) and not pd.isna(txF)) else np.nan

nM = get_kpi_value(kpi, leg_last, "M", "n_elegiveis") if leg_last else np.nan
nF = get_kpi_value(kpi, leg_last, "F", "n_elegiveis") if leg_last else np.nan

limitacoes = [
    (
        '"Reeleição" é proxy (presença consecutiva L-1 → L), '
        "não resultado eleitoral oficial."
    ),
    (
        "Sem partido, UF ou cargo, os resultados são descritivos. "
        "Não permitem inferir causalidade."
    ),
    (
        "A proporção desigual de M e F gera volatilidade nas taxas femininas: "
        f"na última legislatura ({leg_last}), o grupo elegível era "
        f"N={int(nF) if not pd.isna(nF) else '?'} (F) vs "
        f"N={int(nM) if not pd.isna(nM) else '?'} (M). "
        "Os intervalos de confiança evidenciam essa incerteza."
    ),
    (
        "A regressão linear assume tendência constante. "
        "Mudanças legislativas (cotas, reforma eleitoral) podem alterar a trajetória."
    ),
]

achados = [
    (
        f"Representatividade: %F foi de {format_pct(pctF_first)} (leg {leg_first}) "
        f"para {format_pct(pctF_last)} (leg {leg_last}) — "
        f"variação de {format_pct(delta_pctF)} p.p."
    ),
    (
        f"Reeleição (proxy) na leg {leg_last}: "
        f"M = {format_pct(txM)} (N={int(nM) if not pd.isna(nM) else '?'}) vs "
        f"F = {format_pct(txF)} (N={int(nF) if not pd.isna(nF) else '?'}) — "
        f"gap M−F = {format_pct(gap)} p.p. "
        f"Logística: OR(F) = {or_genero:.3f} [{or_lo:.3f}–{or_hi:.3f}] — "
        f"{direcao} ({sig})."
    ),
    (
        f"Projeção: %F cresce ~{coef:.2f} p.p./legislatura (R²={r2:.2f}). "
        f"Estimativa para leg {int(legs_futuras[-1])}: {format_pct(pct_projetado[-1])}."
    ),
]

insights_md = f"""# Insights — Análise por gênero (legislaturas >= {ANALISE_A_PARTIR_DA_LEG})

## Entregáveis
| Arquivo | Descrição |
|---------|-----------|
| `{OUT_LONG}` | CSV long (deputado × legislatura) |
| `{OUT_KPI}` | KPI com IC 95% e N por grupo |
| `grafico_representatividade_pctF.png` | %F por legislatura com marcos |
| `grafico_reeleicao_proxy_taxa.png` | Reeleição M vs F com IC 95% e N |
| `grafico_regressao_logistica.png` | Forest plot (logística) |
| `grafico_projecao_pctF.png` | Projeção %F (linear) com IC |

## Modelos

### Logística: continuou ~ gênero + legislatura
- **Base:** {len(df_trans):,} observações (deputados em L → verificando L+1)
- **OR(F):** {or_genero:.3f} [{or_lo:.3f}–{or_hi:.3f}] — {direcao}
- **p-valor:** {p_genero:.4f} ({sig})
- **OR(legislatura):** {odds_ratios['legislatura']:.3f} — significativo (p = {modelo_logit.pvalues['legislatura']:.4f})

### Linear: %F ~ legislatura
- **Slope:** +{coef:.2f} p.p./legislatura
- **R²:** {r2:.2f}
- **Projeções:** {', '.join([f'leg {int(l)}: {format_pct(p)}' for l, p in zip(legs_futuras, pct_projetado)])}

## Nota sobre proporcionalidade e incerteza
A Câmara tem historicamente muito mais homens que mulheres.
Isso faz com que a taxa de reeleição feminina seja mais volátil
(amostras pequenas oscilam mais). O gráfico de reeleição inclui
intervalos de confiança (Wilson 95%) e o N de cada grupo para que
o leitor avalie a incerteza. Quando os intervalos se cruzam,
não é possível afirmar diferença estatística entre M e F.

## Limitações
{chr(10).join([f"- {x}" for x in limitacoes])}

## Conclusão (3 achados quantificados)
{chr(10).join([f"- {x}" for x in achados])}

## Cadeiras por legislatura
| Legislatura | Cadeiras |
|-------------|----------|
| 48 (1987–1991) | 487 |
| 49 (1991–1995) | 503 |
| 50+ (1995–hoje) | 513 |
"""

print()
for x in limitacoes:
    print(f"  - {x}")
print()
for x in achados:
    print(f"  - {x}")
print()

with open(OUT_INSIGHTS, "w", encoding="utf-8") as f:
    f.write(insights_md)
print(f"  Salvo: {OUT_INSIGHTS}")


# ============================================================
# PASSO 11 — RESUMO FINAL
# ============================================================
print()
print("=" * 70)
print("FINALIZADO — Arquivos gerados:")
print("=" * 70)
for arq in [
    OUT_LONG, OUT_KPI, OUT_INSIGHTS,
    os.path.join(OUT_DIR, "grafico_representatividade_pctF.png"),
    os.path.join(OUT_DIR, "grafico_reeleicao_proxy_taxa.png"),
    os.path.join(OUT_DIR, "grafico_regressao_logistica.png"),
    os.path.join(OUT_DIR, "grafico_projecao_pctF.png"),
]:
    print(f"  {arq}")
print("=" * 70)