import pandas as pd
import numpy as np
import datetime
import sys
import os
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Importamos la logica pesada de nuestro script principal
from calculadora_b2b import (
    parse_season, parse_date, fetch_season_games, process_back_to_backs,
    fetch_future_schedule, check_future_opponent_b2b, get_location, haversine, predict_log5,
    TEAM_NAME_TO_ABBR
)

# ─────────────────────────────────────────────
#  PDF CLASS
# ─────────────────────────────────────────────
class NBA_Report(FPDF):
    def header(self):
        self.set_fill_color(14, 27, 60)
        self.rect(0, 0, 210, 18, 'F')
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 18, '  [NBA]  CALCULADORA B2B  |  MODELO LOG5', ln=True, align='L')
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-13)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Generado con Calculadora B2B NBA  |  Página {self.page_no()}', align='C')
        self.set_text_color(0, 0, 0)

    def section_header(self, title):
        self.set_fill_color(14, 27, 60)
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 11)
        self.cell(0, 8, clean_txt(f'  {title}'), ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.set_font('helvetica', '', 10)
        self.ln(2)

    def alert_cell(self, text, level='warning'):
        # level: 'warning' = orange, 'danger' = red, 'info' = blue
        colors = {'warning': (220, 100, 0), 'danger': (200, 20, 20), 'info': (20, 80, 160)}
        r, g, b = colors.get(level, (0, 0, 0))
        self.set_text_color(r, g, b)
        self.set_font('helvetica', 'B', 10)
        self.cell(0, 6, clean_txt(text), ln=True)
        self.set_text_color(0, 0, 0)
        self.set_font('helvetica', '', 10)

def clean_txt(text):
    """Limpia caracteres especiales para fpdf standard (Latin-1)"""
    return str(text).encode('latin-1', 'replace').decode('latin-1')

# ─────────────────────────────────────────────
#  CHART 1: Donut — Probabilidades Log5
# ─────────────────────────────────────────────
def create_donut_chart(pWW, pWL, pLW, pLL, team_abbr):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor('#F8F9FB')
    ax.set_facecolor('#F8F9FB')

    labels = ['W-W\n(Ambos)', 'W-L\n(1ro)', 'L-W\n(2do)', 'L-L\n(Ninguno)']
    sizes  = [pWW, pWL, pLW, pLL]
    colors = ['#2E86AB', '#A8C5DA', '#E84855', '#3D405B']
    explode = [0.04, 0.02, 0.02, 0.02]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78, explode=explode,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=8.5, weight='bold')
    )
    for t in texts:    t.set_color('#333333')
    for at in autotexts: at.set_color('white'); at.set_fontsize(8); at.set_weight('bold')

    ax.text(0, 0, team_abbr, ha='center', va='center', fontsize=16, weight='bold', color='#14163C')
    ax.set_title('Próximo B2B', fontsize=11, weight='bold', color='#14163C', pad=14)
    plt.tight_layout()
    path = 'tmp_donut.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return path

# ─────────────────────────────────────────────
#  CHART 2: Stacked Bar — Real vs Log5
# ─────────────────────────────────────────────
def create_empirical_chart(team_pairs, pWW, pWL, pLW, pLL, team_abbr):
    sns.set_theme(style='whitegrid')
    total = len(team_pairs)
    if total > 0:
        e_ww = team_pairs['IS_WW'].sum() / total * 100
        e_wl = team_pairs['IS_WL'].sum() / total * 100
        e_lw = team_pairs['IS_LW'].sum() / total * 100
        e_ll = team_pairs['IS_LL'].sum() / total * 100
    else:
        e_ww = e_wl = e_lw = e_ll = 0.0

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_facecolor('#F8F9FB')
    ax.set_facecolor('#F8F9FB')

    x = np.arange(4)
    width = 0.38
    pal = ['#2E86AB', '#A8C5DA', '#E84855', '#3D405B']
    labels = ['W-W', 'W-L', 'L-W', 'L-L']
    emp  = [e_ww, e_wl, e_lw, e_ll]
    model= [pWW, pWL, pLW, pLL]

    b1 = ax.bar(x - width/2, emp,   width, color=[c+'BB' for c in pal], label='Histórico Real',   edgecolor='white', lw=1.5)
    b2 = ax.bar(x + width/2, model, width, color=pal,                    label='Predicción Log5', edgecolor='white', lw=1.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.0f}%',
                        ha='center', va='bottom', fontsize=7.5, weight='bold', color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, weight='bold')
    ax.set_ylabel('Probabilidad (%)', fontsize=9, weight='bold')
    ax.set_title('Desempeño B2B: Real vs Modelo', fontsize=11, weight='bold', color='#14163C', pad=12)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9)
    sns.despine(left=True)
    plt.tight_layout()
    path = 'tmp_emp.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return path

# ─────────────────────────────────────────────
#  CHART 3: Radar — Perfil del Equipo
# ─────────────────────────────────────────────
def create_radar_chart(team_abbr, true_talent, stats_df):
    base_wp = true_talent.get(team_abbr, 0.5) * 100
    b2b_row = stats_df[stats_df['TEAM_NAME'] == team_abbr]

    if not b2b_row.empty:
        p_sweep  = float(b2b_row['Ganan Ambos'].values[0])
        p_game2  = float(b2b_row['Ganan el Segundo'].values[0]) + p_sweep
    else:
        p_sweep  = (base_wp/100 * max(base_wp - 4.5, 0)/100) * 100
        p_game2  = max(base_wp - 4.5, 0)

    p_home = min(base_wp + 3.5, 100)

    categories = ['Talento\nBase', 'Ventaja\nLocal', 'Win\nGame 2', 'Sweep\nB2B']
    values     = [base_wp, p_home, p_game2, p_sweep]
    N = len(categories)

    angles_raw = np.linspace(0, 2 * np.pi, N, endpoint=False)
    vals_c   = np.concatenate((values,   [values[0]]))
    angles_c = np.concatenate((angles_raw, [angles_raw[0]]))

    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0E1B3C')
    ax.set_facecolor('#0E1B3C')

    ax.plot(angles_c, vals_c, color='#FF6B35', linewidth=2.5)
    ax.fill(angles_c, vals_c, color='#FF6B35', alpha=0.3)

    ax.set_ylim(0, 100)
    ax.set_xticks(angles_raw)
    ax.set_xticklabels(categories, fontsize=9, weight='bold', color='white')
    ax.tick_params(colors='white')
    ax.yaxis.set_tick_params(labelcolor='#aaaaaa', labelsize=7)
    ax.set_yticklabels([f'{v:.0f}%' for v in [20,40,60,80,100]], fontsize=7, color='#aaaaaa')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.grid(color='#334080', linestyle='--', linewidth=0.6)
    ax.spines['polar'].set_color('#334080')

    for angle, value in zip(angles_raw, values):
        ax.text(angle, value + 9, f'{value:.0f}%',
                ha='center', va='center', fontsize=8, weight='bold', color='#FF6B35')

    ax.set_title('Perfil del Equipo', fontsize=11, weight='bold', color='white', pad=18)
    plt.tight_layout()
    path = 'tmp_radar.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return path

# ─────────────────────────────────────────────
#  CHART 4: Horizontal Bar — Top 10 Resilience
# ─────────────────────────────────────────────
def create_top10_chart(stats_df, highlight_team=None):
    sns.set_theme(style='white')
    top10 = stats_df.sort_values(by='Ganan Ambos', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor('#F8F9FB')
    ax.set_facecolor('#F8F9FB')

    bars = ax.barh(
        top10['TEAM_NAME'], top10['Ganan Ambos'],
        color='#2E86AB', edgecolor='white', linewidth=1.2, height=0.65
    )

    # Highlight the team if provided
    if highlight_team:
        for bar, name in zip(bars, top10['TEAM_NAME']):
            if name == highlight_team:
                bar.set_color('#E84855')
                bar.set_edgecolor('#14163C')
                bar.set_linewidth(2)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%', va='center', ha='left', fontsize=9, weight='bold', color='#14163C')

    ax.set_xlabel('Probabilidad de Barrida B2B (W-W) %', fontsize=10, weight='bold', color='#333333')
    ax.set_title('Top 10 NBA — Franquicias Más Resilientes en B2B', fontsize=13, weight='bold',
                 color='#14163C', pad=14)
    ax.set_xlim(0, max(top10['Ganan Ambos']) + 8)
    ax.xaxis.set_visible(False)
    sns.despine(left=False, bottom=True)
    ax.spines['left'].set_color('#cccccc')
    plt.tight_layout()
    path = 'tmp_top10.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return path

# ─────────────────────────────────────────────
#  CHART 5: Fatigue Matrix — 2x2 Heatmap
# ─────────────────────────────────────────────
def create_fatigue_matrix(team_pairs):
    """
    Displays Win Rate in Game 2 broken down by 4 fatigue scenarios:
    (Team Fresh | Team B2B) x (Opp Fresh | Opp B2B)
    Uses Game 2 (WL_2) as the outcome since that's where fatigue matters.
    This is a 2x2 heatmap with the win % in each cell.
    """
    scenarios = {
        'Equipo\nFresco\nRival Fresco':    (False, False),
        'Equipo\nFresco\nRival Cansado':   (False, True),
        'Equipo\nCansado\nRival Fresco':   (True,  False),
        'Equipo\nCansado\nRival Cansado':  (True,  True),
    }

    matrix_labels = [['Rival Fresco', 'Rival Cansado'],
                     ['Rival Fresco', 'Rival Cansado']]
    row_labels = ['Equipo Fresco (Game 1)', 'Equipo Cansado (Game 2)']

    data = np.zeros((2, 2))
    counts = np.zeros((2, 2), dtype=int)

    # Game 1 is always "Fresh" for the team (is_b2b=False)
    # Game 2 is "Tired" (is_b2b=True)
    # OPP_B2B_1 / OPP_B2B_2 determines whether rival was fatigued

    # Row 0: team fresh (game 1)
    g1_fresh_fresh = team_pairs[team_pairs['OPP_B2B_1'] == False]
    g1_fresh_tired = team_pairs[team_pairs['OPP_B2B_1'] == True]
    # Row 1: team tired (game 2)
    g2_tired_fresh = team_pairs[team_pairs['OPP_B2B_2'] == False]
    g2_tired_tired = team_pairs[team_pairs['OPP_B2B_2'] == True]

    def wr(df, col): 
        if len(df) == 0: return np.nan
        return df[col].apply(lambda x: 1 if x == 'W' else 0).mean() * 100

    data[0, 0] = wr(g1_fresh_fresh, 'WL_1') if len(g1_fresh_fresh) > 0 else np.nan
    data[0, 1] = wr(g1_fresh_tired, 'WL_1') if len(g1_fresh_tired) > 0 else np.nan
    data[1, 0] = wr(g2_tired_fresh, 'WL_2') if len(g2_tired_fresh) > 0 else np.nan
    data[1, 1] = wr(g2_tired_tired, 'WL_2') if len(g2_tired_tired) > 0 else np.nan

    counts[0, 0] = len(g1_fresh_fresh)
    counts[0, 1] = len(g1_fresh_tired)
    counts[1, 0] = len(g2_tired_fresh)
    counts[1, 1] = len(g2_tired_tired)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.patch.set_facecolor('#0E1B3C')
    ax.set_facecolor('#0E1B3C')

    mask = np.isnan(data)
    display_data = np.where(mask, 0, data)

    im = ax.imshow(display_data, cmap='RdYlGn', vmin=30, vmax=75, aspect='auto')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Rival Fresco', 'Rival Cansado (B2B)'], fontsize=10, weight='bold', color='white')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Eq. Fresco\n(Game 1)', 'Eq. Cansado\n(Game 2)'], fontsize=10, weight='bold', color='white')
    ax.tick_params(colors='white')

    for i in range(2):
        for j in range(2):
            val = data[i, j]
            cnt = counts[i, j]
            txt = f'{val:.0f}%\n({cnt} partidos)' if not np.isnan(val) else 'N/D'
            color = 'black' if (not np.isnan(val) and 35 < val < 65) else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=11, weight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Win Rate %', color='white', fontsize=9)

    ax.set_title('Matriz de Fatiga — Win Rate por Escenario', fontsize=12, weight='bold',
                 color='white', pad=14)
    ax.spines[:].set_color('#334080')
    plt.tight_layout()
    path = 'tmp_matrix.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return path

# ─────────────────────────────────────────────
#  CHART 6: Game 2 Context Bar
# ─────────────────────────────────────────────
def create_game2_context_chart(team_pairs, team_abbr, t_wp):
    """
    Compares 3 scenarios for Game 2 win rate:
    1. All Game 2s ever played
    2. Game 2s where opponent was also in B2B (Rival también cansado)
    3. Game 2s where opponent was NOT in B2B (Rival descansado)
    Also plots the Log5 base talent as a reference line.
    """
    total = len(team_pairs)
    if total == 0:
        return None

    def wr2(df):
        if len(df) == 0: return 0.0
        return df['WL_2'].apply(lambda x: 1 if x == 'W' else 0).mean() * 100

    wr_all    = wr2(team_pairs)
    wr_tired  = wr2(team_pairs[team_pairs['OPP_B2B_2'] == True])
    wr_fresh  = wr2(team_pairs[team_pairs['OPP_B2B_2'] == False])
    n_tired   = len(team_pairs[team_pairs['OPP_B2B_2'] == True])
    n_fresh   = len(team_pairs[team_pairs['OPP_B2B_2'] == False])

    labels    = ['Todos\nGame 2', f'Rival\nFresco\n(n={n_fresh})', f'Rival\nCansado\n(n={n_tired})']
    values_g2 = [wr_all, wr_fresh, wr_tired]
    colors    = ['#2E86AB', '#E84855', '#2ECC71']

    sns.set_theme(style='white')
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.patch.set_facecolor('#F8F9FB')
    ax.set_facecolor('#F8F9FB')

    bars = ax.bar(labels, values_g2, color=colors, edgecolor='white', linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values_g2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold', color='#14163C')

    # Reference line: base talent
    ax.axhline(y=t_wp * 100, color='#FF6B35', linewidth=2, linestyle='--', label=f'Talento Base ({t_wp*100:.0f}%)')
    ax.axhline(y=50, color='#bbbbbb', linewidth=1, linestyle=':', label='50% (línea neutral)')

    ax.set_ylim(0, max(values_g2 + [t_wp * 100]) + 15)
    ax.set_ylabel('Win Rate Game 2 (%)', fontsize=9, weight='bold', color='#333333')
    ax.set_title(f'Win Rate en Game 2 — Contexto de Fatiga Rival', fontsize=11, weight='bold',
                 color='#14163C', pad=12)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc='lower right')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    path = 'tmp_game2.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return path

# ─────────────────────────────────────────────
#  PDF GENERATION
# ─────────────────────────────────────────────
def generate_pdf(team_abbr, season_parsed, cutoff_date, pairs_df, stats_df, true_talent):
    pdf = NBA_Report()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    team_abbr = team_abbr.upper()
    t_wp = true_talent.get(team_abbr, 0.500)

    # ── 1. CABECERA DEL EQUIPO ──────────────────
    pdf.set_font('helvetica', 'B', 15)
    pdf.set_text_color(14, 27, 60)
    pdf.cell(0, 10, f'Análisis Detallado: {team_abbr}  (Temporada {season_parsed})', ln=True)
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(80, 80, 80)
    cut_str = cutoff_date if cutoff_date else 'Sin corte'
    pdf.cell(0, 6, f'Fecha de corte: {cut_str}     |     Win Pct Base: {t_wp*100:.1f}%', ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── 2. ÚLTIMO B2B ──────────────────────────
    team_pairs = pairs_df[pairs_df['TEAM'].str.upper() == team_abbr]
    pdf.section_header('ÚLTIMO BACK-TO-BACK DISPUTADO')

    if team_pairs.empty:
        pdf.cell(0, 8, clean_txt(f'No hay B2B históricos para {team_abbr} con este corte de fecha.'), ln=True)
    else:
        last = team_pairs.iloc[-1]
        res1 = '[W]' if last['WL_1'] == 'W' else '[L]'
        res2 = '[W]' if last['WL_2'] == 'W' else '[L]'
        pdf.cell(0, 6, clean_txt(f"Game 1 ({last['DATE_1'].strftime('%d %b')}):  {last['MATCHUP_1']}  |  {res1}  {last['TEAM']} {int(last['PTS_1'])} pts"), ln=True)

        if last['OPP_B2B_1']:
            pdf.alert_cell('  [!] El rival tambien disputaba su 2do partido de B2B en este juego.', 'warning')

        pdf.cell(0, 6, clean_txt(f"Game 2 ({last['DATE_2'].strftime('%d %b')}):  {last['MATCHUP_2']}  |  {res2}  {last['TEAM']} {int(last['PTS_2'])} pts  |  Viaje: {int(last['DIST_KM'])} km"), ln=True)

        if last['OPP_B2B_2']:
            pdf.alert_cell('  [!] El rival tambien disputaba su 2do partido de B2B en este juego.', 'warning')

        pdf.ln(3)

        # Stats summary
        total_b2b = len(team_pairs)
        w2 = len(team_pairs[team_pairs['WL_2'] == 'W'])
        opp_b2b_g1 = len(team_pairs[team_pairs['OPP_B2B_1'] == True])
        shared_g2 = team_pairs[team_pairs['OPP_B2B_2'] == True]
        opp_b2b_g2 = len(shared_g2)
        w2_shared = len(shared_g2[shared_g2['WL_2'] == 'W'])
        shared_wr = (w2_shared / opp_b2b_g2 * 100) if opp_b2b_g2 > 0 else 0.0

        pdf.set_font('helvetica', 'B', 10)
        pdf.cell(0, 6, 'Rendimiento Empírico Específico', ln=True)
        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 5, f'Total B2B jugados este corte: {total_b2b}', ln=True)
        pdf.cell(0, 5, f'Win Rate en Game 2 (con fatiga): {w2}/{total_b2b} ({w2/total_b2b*100:.1f}%)', ln=True)
        pdf.cell(0, 5, f'Veces que el rival también venía cansado (Game 1): {opp_b2b_g1}', ln=True)
        pdf.cell(0, 5, f'Veces que el rival también venía cansado (Game 2): {opp_b2b_g2}', ln=True)
        if opp_b2b_g2 > 0:
            pdf.set_font('helvetica', 'B', 10)
            pdf.cell(0, 5, clean_txt(f'>> Win Rate Game 2 (AMBOS cansados): {w2_shared}/{opp_b2b_g2} ({shared_wr:.1f}%)'), ln=True)
            pdf.set_font('helvetica', '', 10)
        pdf.ln(4)

    # ── 3. PRÓXIMO B2B ─────────────────────────
    pdf.section_header('PRÓXIMO BACK-TO-BACK')

    future = fetch_future_schedule(team_abbr, season_parsed)
    pWW = pWL = pLW = pLL = None
    p1 = p2 = None

    if future is not None:
        f1, f2 = future
        d1, d2 = f1['Date_Parsed'].strftime('%d %b'), f2['Date_Parsed'].strftime('%d %b')
        op1, op2 = f1['Opponent'], f2['Opponent']
        loc1 = 'Visitante' if '@' in str(f1.get('Unnamed: 5', '')) else 'Local'
        loc2 = 'Visitante' if '@' in str(f2.get('Unnamed: 5', '')) else 'Local'
        m1 = f'@ {op1}' if loc1 == 'Visitante' else f'vs. {op1}'
        m2 = f'@ {op2}' if loc2 == 'Visitante' else f'vs. {op2}'

        l1 = get_location(team_abbr, m1)
        l2 = get_location(team_abbr, m2)
        dist_km = haversine(l1[0][0], l1[0][1], l2[0][0], l2[0][1])

        opp1_wp = true_talent.get(op1, 0.500)
        opp2_wp = true_talent.get(op2, 0.500)

        op1_abbr = TEAM_NAME_TO_ABBR.get(op1, op1)
        op2_abbr = TEAM_NAME_TO_ABBR.get(op2, op2)

        op1_streak, op1_dist = check_future_opponent_b2b(op1_abbr, f1['Date_Parsed'], season_parsed)
        op2_streak, op2_dist = check_future_opponent_b2b(op2_abbr, f2['Date_Parsed'], season_parsed)

        p1 = predict_log5(t_wp, opp1_wp, is_home=(loc1 == 'Local'), is_b2b=False, dist_km=0,
                          opp_b2b=(op1_streak > 1), opp_dist_km=op1_dist)
        p2 = predict_log5(t_wp, opp2_wp, is_home=(loc2 == 'Local'), is_b2b=True,  dist_km=dist_km,
                          opp_b2b=(op2_streak > 1), opp_dist_km=op2_dist)

        pdf.cell(0, 6, clean_txt(f'Fechas: {d1} y {d2}'), ln=True)
        pdf.cell(0, 6, clean_txt(f'Ruta: {loc1} contra {op1}, luego {loc2} contra {op2}.'), ln=True)
        pdf.ln(2)

        # Rival fatigue warnings
        if op1_streak == 2:
            pdf.alert_cell(f'[!] OJO: {op1} estara en su 2do partido del B2B (viajando {int(op1_dist)} km).', 'warning')
        elif op1_streak >= 3:
            pdf.alert_cell(f'[!!] PELIGRO: {op1} en seguidilla de {op1_streak} partidos (viaje: {int(op1_dist)} km).', 'danger')

        if op2_streak == 2:
            pdf.alert_cell(f'[!] OJO: {op2} estara en su 2do partido del B2B (viajando {int(op2_dist)} km).', 'warning')
        elif op2_streak >= 3:
            pdf.alert_cell(f'[!!] PELIGRO: {op2} en seguidilla de {op2_streak} partidos (viaje: {int(op2_dist)} km).', 'danger')

        pdf.ln(2)
        p1_opp = f' - Pen. Rival B2B ({int(op1_dist)}km)' if op1_streak > 1 else ''
        p2_opp = f' - Pen. Rival B2B ({int(op2_dist)}km)' if op2_streak > 1 else ''

        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 6, clean_txt(f'G1  {d1} | {m1} | Rival WP: {opp1_wp*100:.1f}%{p1_opp}  -> P(W) = {p1*100:.1f}%'), ln=True)
        pdf.cell(0, 6, clean_txt(f'G2  {d2} | {m2} | Rival WP: {opp2_wp*100:.1f}% + Fatiga + Viaje ({int(dist_km)}km){p2_opp}  -> P(W) = {p2*100:.1f}%'), ln=True)

        pWW = p1 * p2 * 100
        pWL = p1 * (1 - p2) * 100
        pLW = (1 - p1) * p2 * 100
        pLL = (1 - p1) * (1 - p2) * 100

        pdf.ln(3)
        pdf.set_font('helvetica', 'B', 10)
        pdf.cell(0, 6, f'[W-W] Ganan Ambos:    {pWW:.1f}%', ln=True)
        pdf.cell(0, 6, f'[W-L] Ganan el 1ro:  {pWL:.1f}%', ln=True)
        pdf.cell(0, 6, f'[L-W] Ganan el 2do:  {pLW:.1f}%', ln=True)
        pdf.cell(0, 6, f'[L-L] Pierden Ambos: {pLL:.1f}%', ln=True)
        pdf.set_font('helvetica', '', 10)
        pdf.ln(5)

        # ── Charts Row 1: Donut | Bar comparison | Radar ───
        donut_path = create_donut_chart(pWW, pWL, pLW, pLL, team_abbr)
        emp_path   = create_empirical_chart(team_pairs, pWW, pWL, pLW, pLL, team_abbr)
        radar_path = create_radar_chart(team_abbr, true_talent, stats_df)

        y_row1 = pdf.get_y()
        pdf.image(donut_path,  x=10,  y=y_row1, w=60)
        pdf.image(emp_path,    x=74,  y=y_row1, w=66)
        pdf.image(radar_path,  x=144, y=y_row1, w=57)
        pdf.ln(65)
        os.remove(donut_path)
        os.remove(emp_path)
        os.remove(radar_path)

    else:
        pdf.cell(0, 8, 'No se encontraron futuros B2B para este equipo en el calendario.', ln=True)
        pdf.ln(5)

    # ── Charts Row 2: Fatigue Matrix | Game2 Context ─────
    if not team_pairs.empty:
        pdf.add_page()
        pdf.section_header('ANÁLISIS DE FATIGA — CONTEXTO EMPÍRICO DE LA TEMPORADA')
        pdf.ln(2)

        matrix_path = create_fatigue_matrix(team_pairs)
        game2_path  = create_game2_context_chart(team_pairs, team_abbr, t_wp)

        y_row2 = pdf.get_y()
        pdf.image(matrix_path, x=10, y=y_row2, w=102)
        if game2_path:
            pdf.image(game2_path, x=114, y=y_row2, w=88)
            os.remove(game2_path)
        pdf.ln(75)
        os.remove(matrix_path)

    # ── 4. TABLA GLOBAL 30 EQUIPOS ─────────────
    pdf.add_page()
    pdf.section_header('TABLA GLOBAL — 30 EQUIPOS')
    pdf.set_font('helvetica', '', 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, 'Escenario: doble visita vs oponente .500 WP | 800 km de vuelo entre partidos', ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    top10_path = create_top10_chart(stats_df, highlight_team=team_abbr)
    pdf.image(top10_path, w=180)
    os.remove(top10_path)
    pdf.ln(6)

    # Table header
    pdf.set_fill_color(14, 27, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('helvetica', 'B', 9)
    pdf.cell(48, 7, 'Equipo',          border=0, fill=True)
    pdf.cell(18, 7, 'B2B',             border=0, fill=True, align='C')
    pdf.cell(30, 7, 'W-W %',           border=0, fill=True, align='C')
    pdf.cell(30, 7, 'W-L %',           border=0, fill=True, align='C')
    pdf.cell(30, 7, 'L-W %',           border=0, fill=True, align='C')
    pdf.cell(30, 7, 'L-L %',           border=0, fill=True, align='C')
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    pdf.set_font('helvetica', '', 9)
    fill = False
    for _, row in stats_df.sort_values('Ganan Ambos', ascending=False).iterrows():
        is_team = str(row['TEAM_NAME']).upper() == team_abbr
        if is_team:
            pdf.set_fill_color(230, 240, 255)
            pdf.set_font('helvetica', 'B', 9)
        else:
            pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.set_font('helvetica', '', 9)

        pdf.cell(48, 6, clean_txt(str(row['TEAM_NAME'])), border=0, fill=True)
        pdf.cell(18, 6, str(row['B2B_Jugados']),           border=0, fill=True, align='C')
        pdf.cell(30, 6, f"{row['Ganan Ambos']}%",          border=0, fill=True, align='C')
        pdf.cell(30, 6, f"{row['Ganan el Primero']}%",     border=0, fill=True, align='C')
        pdf.cell(30, 6, f"{row['Ganan el Segundo']}%",     border=0, fill=True, align='C')
        pdf.cell(30, 6, f"{row['Ninguno']}%",              border=0, fill=True, align='C')
        pdf.ln()
        fill = not fill

    # ── Output ─────────────────────────────────
    cut_str_f = cutoff_date.replace('/', '-') if cutoff_date else 'Actual'
    season_clean = season_parsed.replace('/', '-')
    filename = f'Reporte_B2B_{team_abbr}_{season_clean}_{cut_str_f}.pdf'

    try:
        pdf.output(filename)
        print(f'\n✅  Reporte PDF generado exitosamente: → {filename}')
    except Exception as e:
        print(f'\n❌  Error al guardar (cerrá el PDF si lo tenés abierto): {e}')


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  📄  CREADOR DE REPORTE PDF — B2B NBA  📄  ')
    print('=' * 60)

    season_input_raw = input('Ingresa la temporada (ej. 25/26 o 23/24) [Enter: 25/26]: ').strip()
    if not season_input_raw: season_input_raw = '25/26'
    season_parsed = parse_season(season_input_raw)

    cutoff_input = input('Fecha de corte DD/MM/YYYY (ej. 15/02/2026) [Enter: Sin corte]: ').strip()
    cutoff_date  = cutoff_input if cutoff_input else None

    team_input = input('Ingresa abreviatura del equipo (ej. NYK, LAL) [Enter: BOS]: ').strip().upper()
    if not team_input: team_input = 'BOS'

    df = fetch_season_games(season_parsed)
    if df is None: sys.exit(1)

    pairs, stats, true_talent = process_back_to_backs(df, cutoff_date)
    if stats is None:
        print('No se encontraron suficientes datos.')
        sys.exit(1)

    print(f'\nGenerando reporte PDF para {team_input}...')
    generate_pdf(team_input, season_parsed, cutoff_date, pairs, stats, true_talent)


if __name__ == '__main__':
    main()
