"""
=============================================================
  Senior Data Science Dashboard — NZ Business Financial Data
  ANZSIC Industry Dataset | 2011–2024
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# GLOBAL STYLE CONFIG
# ─────────────────────────────────────────────
PALETTE = {
    'bg':       '#0D1117',
    'panel':    '#161B22',
    'accent1':  '#58A6FF',
    'accent2':  '#3FB950',
    'accent3':  '#F78166',
    'accent4':  '#D2A8FF',
    'accent5':  '#FFA657',
    'text':     '#E6EDF3',
    'muted':    '#8B949E',
    'grid':     '#21262D',
}

RICH_PALETTE = [
    '#58A6FF','#3FB950','#F78166','#D2A8FF',
    '#FFA657','#79C0FF','#56D364','#FFA198',
    '#BC8CFF','#FFB77B','#388BFD','#2EA043',
]

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  PALETTE['bg'],
        'axes.facecolor':    PALETTE['panel'],
        'axes.edgecolor':    PALETTE['grid'],
        'axes.labelcolor':   PALETTE['text'],
        'axes.titlecolor':   PALETTE['text'],
        'xtick.color':       PALETTE['muted'],
        'ytick.color':       PALETTE['muted'],
        'text.color':        PALETTE['text'],
        'grid.color':        PALETTE['grid'],
        'grid.linewidth':    0.6,
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'font.family':       'DejaVu Sans',
        'font.size':         10,
        'axes.titlesize':    12,
        'axes.titleweight':  'bold',
        'axes.labelsize':    9,
        'legend.facecolor':  PALETTE['panel'],
        'legend.edgecolor':  PALETTE['grid'],
        'legend.fontsize':   8,
    })

set_dark_style()


# ══════════════════════════════════════════════════════════
#  STEP 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════

print("📥  Loading data …")
raw = pd.read_csv('/mnt/user-data/uploads/data.csv')

# Drop unnamed/empty trailing columns
df = raw.loc[:, ~raw.columns.str.startswith('Unnamed')]

# Standardise column names
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(r'[^a-z0-9]+', '_', regex=True)
              .str.strip('_'))

# Convert value to numeric (some cells may be "..." or empty)
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Drop rows where both value and key identifiers are null
df.dropna(subset=['value'], inplace=True)

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
after  = len(df)

# Ensure year is int
df['year'] = df['year'].astype(int)

# Clean size group labels
size_map = {
    'a_0':              '0 employees',
    'b_1-5':            '1–5',
    'c_6-9':            '6–9',
    'd_10-19':          '10–19',
    'e_20-49':          '20–49',
    'f_50-99':          '50–99',
    'g_100-199':        '100–199',
    'h_200+':           '200+',
    'i_Industry_Total': 'Industry Total',
    'j_Grand_Total':    'Grand Total',
}
df['size_group'] = df['rme_size_grp'].map(size_map).fillna(df['rme_size_grp'])

print(f"✅  Cleaned: {after:,} rows (removed {before-after} dupes)")
print(f"    Shape : {df.shape}")
print(f"    Years : {df['year'].min()} – {df['year'].max()}")
print(f"    Industries : {df['industry_name_anzsic'].nunique()}")
print(f"    Variables : {df['variable'].unique().tolist()}")


# ══════════════════════════════════════════════════════════
#  STEP 2 — EDA HELPERS
# ══════════════════════════════════════════════════════════

# Pivot to wide form for correlation and KPI work
# Filter to grand-total rows only for macro analysis
grand = df[df['rme_size_grp'] == 'j_Grand_Total'].copy()
grand_wide = grand.pivot_table(index=['year','industry_name_anzsic'],
                                columns='variable', values='value', aggfunc='sum').reset_index()
grand_wide.columns.name = None

# Key financial metrics (dollar-valued)
FIN_VARS = [
    'Salaries and wages paid',
    'Sales, government funding, grants and subsidies',
    'Total income',
    'Total expenditure',
    'Operating profit before tax',
    'Total assets',
    'Fixed tangible assets',
]
FIN_VARS = [v for v in FIN_VARS if v in grand_wide.columns]

# ── Summary statistics ──────────────────────────────────
print("\n📊  Summary Statistics (Grand-Total rows):")
print(grand_wide[FIN_VARS].describe().round(1).to_string())

# ── Correlation matrix ──────────────────────────────────
corr = grand_wide[FIN_VARS].corr()


# ══════════════════════════════════════════════════════════
#  STEP 3 + 4 — DASHBOARD VISUALISATIONS
# ══════════════════════════════════════════════════════════

# ── Aggregate helpers ──────────────────────────────────
nz_total = grand_wide.groupby('year')[FIN_VARS].sum()          # economy-wide by year
ind_total = grand_wide.groupby('industry_name_anzsic')[FIN_VARS].sum()  # by industry (all years)

# Top 10 industries by total income
top10 = (ind_total['Total income']
         .sort_values(ascending=False)
         .head(10)
         .index.tolist()
         if 'Total income' in ind_total.columns else ind_total.index[:10].tolist())

# Size-band analysis (exclude totals)
size_df = df[df['rme_size_grp'].isin(
    ['a_0','b_1-5','c_6-9','d_10-19','e_20-49','f_50-99','g_100-199','h_200+']
)].copy()


# ╔══════════════════════════════════════════════════════╗
# ║  DASHBOARD 1 — EXECUTIVE OVERVIEW                   ║
# ╚══════════════════════════════════════════════════════╝
print("\n🎨  Rendering Dashboard 1 — Executive Overview …")

fig = plt.figure(figsize=(22, 16), facecolor=PALETTE['bg'])
fig.suptitle('NEW ZEALAND BUSINESS ECONOMY  ·  EXECUTIVE OVERVIEW  ·  2011–2024',
             fontsize=15, fontweight='bold', color=PALETTE['text'],
             y=0.98, x=0.5)

gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38,
              top=0.93, bottom=0.06, left=0.06, right=0.97)

# ── KPI cards (row 0) ──────────────────────────────────
kpi_metrics = {
    'Total Income\n(2024, $B)':           nz_total.loc[2024,'Total income']/1000       if 'Total income'  in nz_total.columns else np.nan,
    'Total Assets\n(2024, $B)':           nz_total.loc[2024,'Total assets']/1000       if 'Total assets'  in nz_total.columns else np.nan,
    'Operating Profit\n(2024, $B)':       nz_total.loc[2024,'Operating profit before tax']/1000  if 'Operating profit before tax' in nz_total.columns else np.nan,
    'Salaries Paid\n(2024, $B)':          nz_total.loc[2024,'Salaries and wages paid']/1000      if 'Salaries and wages paid' in nz_total.columns else np.nan,
}
kpi_colors = [PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent3'], PALETTE['accent4']]

for i, (label, val) in enumerate(kpi_metrics.items()):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(PALETTE['panel'])
    for sp in ax.spines.values(): sp.set_color(kpi_colors[i]); sp.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, f"${val:,.0f}B" if not np.isnan(val) else "N/A",
            ha='center', va='center', fontsize=22, fontweight='bold',
            color=kpi_colors[i], transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=9,
            color=PALETTE['muted'], transform=ax.transAxes)
    ax.set_title('', visible=False)

# ── Total Income trend (row 1, span 2) ─────────────────
ax1 = fig.add_subplot(gs[1, :2])
if 'Total income' in nz_total.columns:
    ax1.fill_between(nz_total.index, nz_total['Total income']/1000,
                     alpha=0.18, color=PALETTE['accent1'])
    ax1.plot(nz_total.index, nz_total['Total income']/1000,
             color=PALETTE['accent1'], lw=2.5, marker='o', ms=5)
    ax1.set_title('Economy-Wide Total Income ($B)', pad=8)
    ax1.set_xlabel('Year')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}B'))

# ── Operating profit trend ──────────────────────────────
ax2 = fig.add_subplot(gs[1, 2:])
if 'Operating profit before tax' in nz_total.columns:
    profit = nz_total['Operating profit before tax']/1000
    colors_bar = [PALETTE['accent2'] if v >= 0 else PALETTE['accent3'] for v in profit]
    ax2.bar(nz_total.index, profit, color=colors_bar, width=0.7, alpha=0.85)
    ax2.axhline(0, color=PALETTE['muted'], lw=0.8, ls='--')
    ax2.set_title('Economy-Wide Operating Profit Before Tax ($B)', pad=8)
    ax2.set_xlabel('Year')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}B'))

# ── Top 10 Industries by Total Income (row 2, span 2) ──
ax3 = fig.add_subplot(gs[2, :2])
if top10 and 'Total income' in ind_total.columns:
    vals = ind_total.loc[top10, 'Total income']/1000
    short_names = [n.split(',')[0][:28] for n in top10]
    bars = ax3.barh(short_names[::-1], vals.values[::-1],
                    color=RICH_PALETTE[:10], alpha=0.88)
    ax3.set_title('Top 10 Industries — Total Income (All Years, $B)', pad=8)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}B'))
    for bar, v in zip(bars, vals.values[::-1]):
        ax3.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                 f'${v:,.0f}B', va='center', fontsize=7.5, color=PALETTE['muted'])

# ── Salaries vs Total Expenditure scatter (row 2, span 2)
ax4 = fig.add_subplot(gs[2, 2:])
if 'Salaries and wages paid' in grand_wide.columns and 'Total expenditure' in grand_wide.columns:
    scat = grand_wide[grand_wide['industry_name_anzsic'].isin(top10)].dropna(
        subset=['Salaries and wages paid','Total expenditure'])
    industries = scat['industry_name_anzsic'].unique()
    for idx, ind in enumerate(industries):
        sub = scat[scat['industry_name_anzsic']==ind]
        ax4.scatter(sub['Salaries and wages paid']/1000,
                    sub['Total expenditure']/1000,
                    color=RICH_PALETTE[idx % len(RICH_PALETTE)],
                    alpha=0.75, s=45, label=ind.split(',')[0][:20])
    ax4.set_xlabel('Salaries & Wages Paid ($B)')
    ax4.set_ylabel('Total Expenditure ($B)')
    ax4.set_title('Salaries vs Total Expenditure — Top Industries', pad=8)
    ax4.legend(fontsize=6.5, ncol=2, loc='upper left')

plt.savefig('/mnt/user-data/outputs/dashboard_1_executive_overview.png',
            dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
plt.close()
print("   ✅  Saved dashboard_1_executive_overview.png")


# ╔══════════════════════════════════════════════════════╗
# ║  DASHBOARD 2 — TREND & TIME-SERIES ANALYSIS         ║
# ╚══════════════════════════════════════════════════════╝
print("🎨  Rendering Dashboard 2 — Trend Analysis …")

fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor=PALETTE['bg'])
fig.suptitle('TREND & TIME-SERIES ANALYSIS  ·  2011–2024',
             fontsize=15, fontweight='bold', color=PALETTE['text'], y=1.0)

for row in axes:
    for ax in row:
        ax.set_facecolor(PALETTE['panel'])

# 1. Multi-line — key financial vars over time
ax = axes[0,0]
for i, var in enumerate(FIN_VARS[:5]):
    if var in nz_total.columns:
        ax.plot(nz_total.index, nz_total[var]/1000,
                label=var[:30], color=RICH_PALETTE[i], lw=2, marker='o', ms=3)
ax.set_title('Key Financial Metrics Over Time ($B)')
ax.set_xlabel('Year'); ax.legend(fontsize=6.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 2. YoY growth — Total income
ax = axes[0,1]
if 'Total income' in nz_total.columns:
    yoy = nz_total['Total income'].pct_change()*100
    bar_colors = [PALETTE['accent2'] if v>=0 else PALETTE['accent3'] for v in yoy.dropna()]
    ax.bar(yoy.dropna().index, yoy.dropna(), color=bar_colors, width=0.7, alpha=0.85)
    ax.axhline(0, color=PALETTE['muted'], lw=0.8, ls='--')
    ax.set_title('Year-on-Year Growth — Total Income (%)')
    ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.1f}%'))
    # Annotate max growth
    mx = yoy.idxmax()
    ax.annotate(f'{yoy[mx]:.1f}%', xy=(mx, yoy[mx]),
                xytext=(mx+0.3, yoy[mx]+0.5), fontsize=7.5,
                color=PALETTE['accent2'], arrowprops=dict(arrowstyle='->', color=PALETTE['muted']))

# 3. Industry trends — top 5 by income
ax = axes[0,2]
top5 = top10[:5]
for i, ind in enumerate(top5):
    sub = grand_wide[grand_wide['industry_name_anzsic']==ind].sort_values('year')
    if 'Total income' in sub.columns:
        ax.plot(sub['year'], sub['Total income']/1000,
                label=ind.split(',')[0][:22], color=RICH_PALETTE[i], lw=2, marker='s', ms=3)
ax.set_title('Top 5 Industry Income Trends ($B)')
ax.set_xlabel('Year'); ax.legend(fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 4. Salaries trend with fill
ax = axes[1,0]
if 'Salaries and wages paid' in nz_total.columns:
    s = nz_total['Salaries and wages paid']/1000
    ax.fill_between(s.index, s, alpha=0.2, color=PALETTE['accent4'])
    ax.plot(s.index, s, color=PALETTE['accent4'], lw=2.5, marker='D', ms=5)
    ax.set_title('Total Salaries & Wages Paid ($B)')
    ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 5. Assets vs expenditure dual-y
ax = axes[1,1]
ax2b = ax.twinx()
ax2b.set_facecolor(PALETTE['panel'])
if 'Total assets' in nz_total.columns and 'Total expenditure' in nz_total.columns:
    ax.plot(nz_total.index, nz_total['Total assets']/1000,
            color=PALETTE['accent1'], lw=2, marker='o', ms=4, label='Total Assets')
    ax2b.plot(nz_total.index, nz_total['Total expenditure']/1000,
              color=PALETTE['accent5'], lw=2, marker='^', ms=4, label='Total Expenditure')
    ax.set_ylabel('Total Assets ($B)', color=PALETTE['accent1'])
    ax2b.set_ylabel('Total Expenditure ($B)', color=PALETTE['accent5'])
    ax2b.tick_params(colors=PALETTE['accent5'])
    ax.set_title('Assets vs Expenditure Over Time')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))
    ax2b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 6. Rolling 3-year average profit margin
ax = axes[1,2]
if 'Operating profit before tax' in nz_total.columns and 'Total income' in nz_total.columns:
    margin = (nz_total['Operating profit before tax'] / nz_total['Total income'] * 100)
    roll   = margin.rolling(3, center=True).mean()
    ax.fill_between(margin.index, margin, alpha=0.15, color=PALETTE['accent3'])
    ax.plot(margin.index, margin, color=PALETTE['accent3'], lw=1.5, alpha=0.5, ls='--', label='Annual')
    ax.plot(roll.index, roll, color=PALETTE['accent3'], lw=2.5, label='3-yr rolling avg')
    ax.axhline(margin.mean(), color=PALETTE['muted'], lw=1, ls=':')
    ax.set_title('Profit Margin % (Operating Profit / Total Income)')
    ax.set_xlabel('Year'); ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.1f}%'))

plt.tight_layout(pad=2)
plt.savefig('/mnt/user-data/outputs/dashboard_2_trend_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
plt.close()
print("   ✅  Saved dashboard_2_trend_analysis.png")


# ╔══════════════════════════════════════════════════════╗
# ║  DASHBOARD 3 — CATEGORY COMPARISONS                 ║
# ╚══════════════════════════════════════════════════════╝
print("🎨  Rendering Dashboard 3 — Category Comparisons …")

fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor=PALETTE['bg'])
fig.suptitle('INDUSTRY & SIZE-BAND COMPARISONS',
             fontsize=15, fontweight='bold', color=PALETTE['text'], y=1.0)
for row in axes:
    for ax in row: ax.set_facecolor(PALETTE['panel'])

# 1. Grouped bar — income vs expenditure by top 8 industries
ax = axes[0,0]
top8 = top10[:8]
if 'Total income' in ind_total.columns and 'Total expenditure' in ind_total.columns:
    names = [n.split(',')[0][:18] for n in top8]
    inc  = ind_total.loc[top8,'Total income']/1000
    exp_ = ind_total.loc[top8,'Total expenditure']/1000
    x = np.arange(len(top8))
    ax.bar(x-0.2, inc,  width=0.38, color=PALETTE['accent1'], alpha=0.85, label='Income')
    ax.bar(x+0.2, exp_, width=0.38, color=PALETTE['accent3'], alpha=0.85, label='Expenditure')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
    ax.set_title('Income vs Expenditure — Top 8 Industries ($B)')
    ax.legend(fontsize=8); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 2. Stacked bar — salary share of income by industry
ax = axes[0,1]
if 'Salaries and wages paid' in ind_total.columns and 'Total income' in ind_total.columns:
    sal_share = (ind_total.loc[top8,'Salaries and wages paid'] /
                 ind_total.loc[top8,'Total income'] * 100)
    names = [n.split(',')[0][:18] for n in top8]
    bars = ax.bar(names, sal_share, color=RICH_PALETTE[:8], alpha=0.85)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
    ax.set_title('Salary % of Total Income by Industry')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.0f}%'))
    for bar, v in zip(bars, sal_share):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}%', ha='center', fontsize=6.5, color=PALETTE['muted'])

# 3. Pie — income share by top industry groups
ax = axes[0,2]
if 'Total income' in ind_total.columns:
    pie_vals = ind_total['Total income'].nlargest(8)
    other = ind_total['Total income'].sum() - pie_vals.sum()
    pie_vals['Other'] = other
    labels = [l.split(',')[0][:20] for l in pie_vals.index]
    wedges, texts, autotexts = ax.pie(
        pie_vals, labels=None, autopct='%1.1f%%',
        colors=RICH_PALETTE[:9], startangle=140,
        pctdistance=0.75, wedgeprops={'linewidth':1.5,'edgecolor':PALETTE['bg']})
    for at in autotexts: at.set_fontsize(7); at.set_color(PALETTE['bg'])
    ax.legend(wedges, labels, loc='lower left', fontsize=6.5, framealpha=0.3)
    ax.set_title('Income Share by Industry Group')

# 4. Size-band comparison — employment count
ax = axes[1,0]
size_emp = (size_df[size_df['variable']=='Rolling mean employees']
            .groupby('size_group')['value'].mean()
            .reindex(['0 employees','1–5','6–9','10–19','20–49','50–99','100–199','200+']))
size_emp = size_emp.dropna()
bars = ax.bar(size_emp.index, size_emp, color=RICH_PALETTE[:len(size_emp)], alpha=0.85)
ax.set_xticklabels(size_emp.index, rotation=35, ha='right', fontsize=8)
ax.set_title('Avg Rolling Mean Employees by Size Band')
ax.set_ylabel('Avg Employees')

# 5. Box plots — profit by size band
ax = axes[1,1]
size_profit = (size_df[size_df['variable']=='Operating profit before tax']
               .assign(sg=lambda d: d['size_group'])
               .pivot_table(index=['year','industry_name_anzsic'],
                            columns='sg', values='value', aggfunc='sum'))
valid_cols = [c for c in ['0 employees','1–5','6–9','10–19','20–49','50–99','100–199','200+']
              if c in size_profit.columns]
if valid_cols:
    data_box = [size_profit[c].dropna().values/1000 for c in valid_cols]
    bp = ax.boxplot(data_box, patch_artist=True, showfliers=False,
                    medianprops={'color':'white','lw':2})
    for patch, color in zip(bp['boxes'], RICH_PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(valid_cols, rotation=35, ha='right', fontsize=7)
    ax.set_title('Operating Profit Distribution by Size Band ($B)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 6. Violin — total income distribution by decade-half
ax = axes[1,2]
grand_wide['period'] = pd.cut(grand_wide['year'],
                               bins=[2010,2015,2019,2024],
                               labels=['2011–2015','2016–2019','2020–2024'])
if 'Total income' in grand_wide.columns:
    period_data = [grand_wide[grand_wide['period']==p]['Total income'].dropna()/1000
                   for p in ['2011–2015','2016–2019','2020–2024']]
    parts = ax.violinplot(period_data, positions=[1,2,3], showmedians=True, widths=0.6)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(RICH_PALETTE[i*2]); pc.set_alpha(0.75)
    parts['cmedians'].set_color('white'); parts['cmedians'].set_lw(2)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['2011–2015','2016–2019','2020–2024'])
    ax.set_title('Total Income Distribution by Period ($B)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

plt.tight_layout(pad=2)
plt.savefig('/mnt/user-data/outputs/dashboard_3_category_comparisons.png',
            dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
plt.close()
print("   ✅  Saved dashboard_3_category_comparisons.png")


# ╔══════════════════════════════════════════════════════╗
# ║  DASHBOARD 4 — HEATMAPS & CORRELATION               ║
# ╚══════════════════════════════════════════════════════╝
print("🎨  Rendering Dashboard 4 — Heatmaps & Correlation …")

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=PALETTE['bg'])
fig.suptitle('CORRELATION & INDUSTRY HEATMAPS',
             fontsize=15, fontweight='bold', color=PALETTE['text'], y=1.01)

# 1. Correlation heatmap
ax = axes[0]; ax.set_facecolor(PALETTE['panel'])
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt='.2f', annot_kws={'size':8},
            linewidths=0.5, linecolor=PALETTE['bg'],
            cbar_kws={'shrink':0.8})
ax.set_title('Pearson Correlation Matrix — Financial Variables', pad=10)
ax.tick_params(labelsize=7)
short_labels = [l[:25] for l in corr.columns]
ax.set_xticklabels(short_labels, rotation=40, ha='right')
ax.set_yticklabels(short_labels, rotation=0)

# 2. Industry × Year income heatmap
ax = axes[1]; ax.set_facecolor(PALETTE['panel'])
if 'Total income' in grand_wide.columns:
    pivot_hy = grand_wide[grand_wide['industry_name_anzsic'].isin(top10)].pivot_table(
        index='industry_name_anzsic', columns='year', values='Total income', aggfunc='sum')
    pivot_hy.index = [i.split(',')[0][:28] for i in pivot_hy.index]
    sns.heatmap(pivot_hy/1000, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
                annot_kws={'size':6.5}, linewidths=0.4, linecolor=PALETTE['bg'],
                cbar_kws={'label':'Total Income ($B)','shrink':0.8})
    ax.set_title('Total Income Heatmap — Top 10 Industries × Year ($B)', pad=10)
    ax.set_xlabel('Year'); ax.set_ylabel('')
    ax.tick_params(labelsize=7.5)

plt.tight_layout(pad=2.5)
plt.savefig('/mnt/user-data/outputs/dashboard_4_heatmaps.png',
            dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
plt.close()
print("   ✅  Saved dashboard_4_heatmaps.png")


# ╔══════════════════════════════════════════════════════╗
# ║  DASHBOARD 5 — INSIGHTS & MULTI-VARIABLE            ║
# ╚══════════════════════════════════════════════════════╝
print("🎨  Rendering Dashboard 5 — Insights …")

fig, axes = plt.subplots(2, 2, figsize=(20, 13), facecolor=PALETTE['bg'])
fig.suptitle('ADVANCED INSIGHTS & MULTI-VARIABLE ANALYSIS',
             fontsize=15, fontweight='bold', color=PALETTE['text'], y=1.0)
for row in axes:
    for ax in row: ax.set_facecolor(PALETTE['panel'])

# 1. Profitability ranking — latest year
ax = axes[0,0]
latest = grand_wide[grand_wide['year']==grand_wide['year'].max()]
if 'Operating profit before tax' in latest.columns:
    rank = (latest.set_index('industry_name_anzsic')['Operating profit before tax']
            .sort_values(ascending=False).head(10)/1000)
    short = [r.split(',')[0][:22] for r in rank.index]
    colors_r = [PALETTE['accent2'] if v>=0 else PALETTE['accent3'] for v in rank]
    ax.barh(short[::-1], rank.values[::-1], color=colors_r[::-1], alpha=0.85)
    ax.set_title(f'Operating Profit Ranking ({grand_wide["year"].max()})')
    ax.axvline(0, color=PALETTE['muted'], lw=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))

# 2. Asset intensity (assets per $income dollar)
ax = axes[0,1]
if 'Total assets' in ind_total.columns and 'Total income' in ind_total.columns:
    asset_int = (ind_total.loc[top10,'Total assets'] /
                 ind_total.loc[top10,'Total income']).sort_values(ascending=False)
    short = [r.split(',')[0][:22] for r in asset_int.index]
    bars = ax.barh(short[::-1], asset_int.values[::-1],
                   color=RICH_PALETTE[:10][::-1], alpha=0.85)
    ax.set_title('Asset Intensity (Total Assets / Total Income) — Top 10 Industries')
    ax.set_xlabel('Asset Intensity Ratio')
    ax.axvline(1, color=PALETTE['muted'], lw=1, ls='--')
    ax.text(1.02, 0.02, 'Asset-heavy →', transform=ax.transAxes,
            fontsize=7, color=PALETTE['muted'])

# 3. Multi-variable radar chart — top 5 industries
ax = axes[1,0]
ax.remove()
ax = fig.add_subplot(2, 2, 3, polar=True, facecolor=PALETTE['panel'])
radar_vars = [v for v in ['Total income','Total assets','Salaries and wages paid',
                           'Total expenditure','Operating profit before tax']
              if v in ind_total.columns]
if len(radar_vars) >= 3:
    N = len(radar_vars)
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    top5_radar = top10[:5]
    for i, ind in enumerate(top5_radar):
        vals_r = ind_total.loc[ind, radar_vars]
        vals_norm = (vals_r - ind_total[radar_vars].min()) / \
                    (ind_total[radar_vars].max() - ind_total[radar_vars].min())
        vals_plot = list(vals_norm) + [vals_norm.iloc[0]]
        ax.plot(angles, vals_plot, color=RICH_PALETTE[i], lw=2, label=ind.split(',')[0][:20])
        ax.fill(angles, vals_plot, color=RICH_PALETTE[i], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([v[:22] for v in radar_vars], fontsize=7, color=PALETTE['text'])
    ax.tick_params(colors=PALETTE['muted'])
    ax.set_yticklabels([])
    ax.set_facecolor(PALETTE['panel'])
    ax.spines['polar'].set_color(PALETTE['grid'])
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=6.5)
    ax.set_title('Multi-Variable Industry Profile (Normalised)', pad=18,
                 color=PALETTE['text'], fontsize=11, fontweight='bold')

# 4. COVID impact — operating profit YoY
ax = axes[1,1]
if 'Operating profit before tax' in nz_total.columns:
    profit_ts = nz_total['Operating profit before tax']/1000
    ax.fill_between(profit_ts.index, profit_ts, alpha=0.15, color=PALETTE['accent5'])
    ax.plot(profit_ts.index, profit_ts, color=PALETTE['accent5'], lw=2.5, marker='o', ms=5)
    ax.axvspan(2019.5, 2021.5, alpha=0.12, color=PALETTE['accent3'], label='COVID period')
    ax.set_title('Operating Profit — COVID Impact Period Highlighted ($B)')
    ax.set_xlabel('Year'); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}B'))
    ax.annotate('COVID-19\nOnset', xy=(2020, profit_ts[2020]),
                xytext=(2017.5, profit_ts[2020]-3),
                color=PALETTE['accent3'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=PALETTE['accent3']))

plt.tight_layout(pad=2)
plt.savefig('/mnt/user-data/outputs/dashboard_5_insights.png',
            dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
plt.close()
print("   ✅  Saved dashboard_5_insights.png")

print("\n🏁  All dashboards rendered successfully!")
