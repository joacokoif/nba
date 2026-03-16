import pandas as pd
import datetime
import math
import sys
import io
import requests
import urllib3
from nba_api.stats.endpoints import leaguegamelog

# Forzar UTF-8 en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
urllib3.disable_warnings()

ARENAS = {
    'ATL': (33.7573, -84.3963), 'BOS': (42.3662, -71.0621), 'BKN': (40.6826, -73.9754),
    'CHA': (35.2251, -80.8392), 'CHI': (41.8807, -87.6742), 'CLE': (41.4965, -81.6881),
    'DAL': (32.7905, -96.8103), 'DEN': (39.7486, -105.0075), 'DET': (42.3411, -83.0550),
    'GSW': (37.7680, -122.3877), 'HOU': (29.7508, -95.3621), 'IND': (39.7639, -86.1555),
    'LAC': (34.0430, -118.2673), 'LAL': (34.0430, -118.2673), 'MEM': (35.1381, -90.0506),
    'MIA': (25.7814, -80.1870),  'MIL': (43.0451, -87.9172),  'MIN': (44.9795, -93.2761),
    'NOP': (29.9490, -90.0821),  'NYK': (40.7505, -73.9934),  'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3839),  'PHI': (39.9012, -75.1720),  'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668), 'SAC': (38.5802, -121.4997), 'SAS': (29.4270, -98.4375),
    'TOR': (43.6435, -79.3791),  'UTA': (40.7683, -111.9011), 'WAS': (38.8981, -77.0209)
}

def parse_season(season_str):
    if '/' in season_str:
        parts = season_str.split('/')
        if len(parts) == 2 and len(parts[0]) == 2:
            return f"20{parts[0]}-{parts[1]}"
    return season_str

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    except Exception:
        return None

def compute_true_talent(df):
    """Calcula el Porcentaje de Victorias (%WP) real de cada equipo de la temporada"""
    # Usamos apply con lambda devolviendo float en lugar de boolean
    df['IS_WIN'] = df['WL'].apply(lambda x: 1.0 if x == 'W' else 0.0)
    df_wins = df.groupby('TEAM_ABBREVIATION').agg(
        WINS=('IS_WIN', 'sum'),
        GAMES=('GAME_ID', 'count')
    ).reset_index()
    
    df_wins['GAMES'] = df_wins['GAMES'].replace(0, 1)
    df_wins['WP'] = df_wins['WINS'] / df_wins['GAMES']
    return dict(zip(df_wins['TEAM_ABBREVIATION'], df_wins['WP']))

def predict_log5(team_wp, opp_wp, is_home, is_b2b, dist_km):
    """Fórmula Predictiva Log5 ajustada por factores de Ecosistema NBA"""
    home_adv = 0.035
    b2b_penalty = 0.045
    t_wp, o_wp = team_wp, opp_wp
    
    if is_home:
        t_wp += home_adv
        o_wp -= home_adv
    else:
        t_wp -= home_adv
        o_wp += home_adv
        
    if is_b2b:
        t_wp -= b2b_penalty
        if dist_km > 1000: t_wp -= 0.015
        if dist_km > 2000: t_wp -= 0.025
            
    t_wp = max(0.01, min(0.99, t_wp))
    o_wp = max(0.01, min(0.99, o_wp))
    
    p_team = (t_wp * (1 - o_wp)) / ((t_wp * (1 - o_wp)) + (o_wp * (1 - t_wp)))
    return p_team

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def get_location(team_abbr, matchup):
    if "vs." in str(matchup):
        return ARENAS.get(team_abbr, (0,0)), True, team_abbr
    else:
        opp = str(matchup)[-3:]
        return ARENAS.get(opp, ARENAS.get(team_abbr, (0,0))), False, opp

def fetch_season_games(season_str):
    print(f"\nDescargando datos de la API para temporada {season_str}...")
    try:
        game_log = leaguegamelog.LeagueGameLog(season=season_str, player_or_team_abbreviation='T')
        return game_log.get_data_frames()[0]
    except Exception as e:
        print(f"Error al descargar datos: {e}")
        return None

def fetch_future_schedule(team_abbr, season_str):
    try:
        from io import StringIO
        season_start = int(season_str[:4])
        season_end = season_start + 1
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/{season_end}_games.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        html = requests.get(url, headers=headers).text
        
        dfs = pd.read_html(StringIO(html))
        if not dfs: return None
        df = dfs[0].copy()
        
        if 'G' in df.columns:
            df = df[df['G'] != 'G'].copy()
        
        if 'Date' in df.columns:
            df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date_Parsed']).sort_values('Date_Parsed').copy()
            
            # Calcular en el df COMPLETO para no quebrar back-to-backs actuales
            df['PREV_DATE'] = df['Date_Parsed'].shift(1)
            df['DAYS_REST'] = (df['Date_Parsed'] - df['PREV_DATE']).dt.days
            
            if 'Tm' in df.columns:
                future = df[df['Tm'].isna()].copy()
            else:
                 future = df[df['Date_Parsed'] >= pd.Timestamp.today()].copy()
                 
            future = future.sort_values('Date_Parsed')
            
            b2b_labels = future[future['DAYS_REST'] == 1].index
            if len(b2b_labels) > 0:
                idx2_label = b2b_labels[0]
                idx2 = df.index.get_loc(idx2_label)
                idx1 = idx2 - 1
                return df.iloc[idx1].copy(), df.iloc[idx2].copy()
    except Exception as e:
        pass
    return None

def process_back_to_backs(df, cutoff_date=None):
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    if cutoff_date:
        parsed_cutoff = parse_date(cutoff_date)
        if parsed_cutoff:
            df = df[df['GAME_DATE'] <= parsed_cutoff]
            print(f"Filtro aplicado: partidos jugados hasta {parsed_cutoff.strftime('%d/%m/%Y')} inclusive.")
        
    true_talent = compute_true_talent(df)

    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    
    b2b_idx2 = df.index[df['DAYS_REST'] == 1]
    b2b_idx1 = b2b_idx2 - 1
    
    g1 = df.loc[b2b_idx1].reset_index(drop=True)
    g2 = df.loc[b2b_idx2].reset_index(drop=True)
    
    if g1.empty or g2.empty:
        return None, None, true_talent
        
    locs1 = [get_location(t, m) for t, m in zip(g1['TEAM_ABBREVIATION'], g1['MATCHUP'])]
    locs2 = [get_location(t, m) for t, m in zip(g2['TEAM_ABBREVIATION'], g2['MATCHUP'])]
    distances = [haversine(l1[0][0], l1[0][1], l2[0][0], l2[0][1]) for l1, l2 in zip(locs1, locs2)]
    
    w1 = (g1['WL'] == 'W')
    w2 = (g2['WL'] == 'W')

    pairs = pd.DataFrame({
        'TEAM': g1['TEAM_ABBREVIATION'],
        'TEAM_NAME': g1['TEAM_NAME'],
        'DATE_1': g1['GAME_DATE'],
        'MATCHUP_1': g1['MATCHUP'],
        'IS_H_1': [l[1] for l in locs1],
        'WL_1': g1['WL'],
        'PTS_1': g1['PTS'],
        'DATE_2': g2['GAME_DATE'],
        'MATCHUP_2': g2['MATCHUP'],
        'IS_H_2': [l[1] for l in locs2],
        'WL_2': g2['WL'],
        'PTS_2': g2['PTS'],
        'DIST_KM': distances,
        'IS_WW': w1 & w2,
        'IS_WL': w1 & ~w2,
        'IS_LW': ~w1 & w2,
        'IS_LL': ~w1 & ~w2
    })
    
    hist_stats = pairs.groupby('TEAM').agg(
        B2B_Jugados=('TEAM', 'count'),
        AVG_DIST_B2B=('DIST_KM', 'mean')
    ).reset_index().rename(columns={'TEAM': 'TEAM_ABBREVIATION'})
    
    global_stats = []
    for team_abbr, t_wp in true_talent.items():
        # Baseline simulation: Away game vs 0.500 opponent, then Away B2B vs 0.500 w/ 800km travel
        p1 = predict_log5(t_wp, 0.500, False, False, 0)
        p2 = predict_log5(t_wp, 0.500, False, True, 800)
        global_stats.append({
            'TEAM_ABBREVIATION': team_abbr,
            'Ganan Ambos': p1 * p2 * 100,
            'Ganan el Primero': p1 * (1 - p2) * 100,
            'Ganan el Segundo': (1 - p1) * p2 * 100,
            'Ninguno': (1 - p1) * (1 - p2) * 100
        })
        
    stats_df = pd.DataFrame(global_stats)
    names_df = df[['TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates()
    stats_df = pd.merge(stats_df, names_df, on='TEAM_ABBREVIATION', how='left')
    stats_df = pd.merge(stats_df, hist_stats, on='TEAM_ABBREVIATION', how='left')
    
    stats_df['B2B_Jugados'] = stats_df['B2B_Jugados'].fillna(0).astype(int)
    stats_df['AVG_DIST_B2B'] = stats_df['AVG_DIST_B2B'].fillna(0).round(0).astype(int)
    
    for col in ['Ganan Ambos', 'Ganan el Primero', 'Ganan el Segundo', 'Ninguno']:
        stats_df[col] = stats_df[col].round(1)
        
    stats_df = stats_df.sort_values(by='Ganan Ambos', ascending=False)
    
    return pairs, stats_df, true_talent

def print_team_deep_dive(pairs_df, team_abbr, season_input, true_talent):
    print(f"\n" + "="*70)
    print(f" 🏀 Analisis Detallado Back-to-Back: {team_abbr} ")
    print("="*70)
    
    t_wp = true_talent.get(team_abbr.upper(), 0.500)
    print(f"Base de Simulacion (True Talent Log5): {team_abbr} está al {t_wp*100:.1f}%\n")
    
    team_pairs = pairs_df[pairs_df['TEAM'].str.upper() == team_abbr.upper()]
    if team_pairs.empty:
        print(f"No hay datos B2B históricos jugados para {team_abbr} en este filtro.")
    else:
        last_pair = team_pairs.iloc[-1]
        date1, date2 = last_pair['DATE_1'].strftime('%d %b'), last_pair['DATE_2'].strftime('%d %b')
        res1 = '✅' if last_pair['WL_1']=='W' else '❌'
        res2 = '✅' if last_pair['WL_2']=='W' else '❌'
        
        print("📋 ÚLTIMO BACK-TO-BACK DISPUTADO")
        print("-" * 50)
        print(f"{date1:<10} | {last_pair['MATCHUP_1']:<15} | {res1} {last_pair['TEAM']} {last_pair['PTS_1']}")
        print(f"{date2:<10} | {last_pair['MATCHUP_2']:<15} | {res2} {last_pair['TEAM']} {last_pair['PTS_2']}")
        print(f"Distancia recorrida entre partidos: {int(last_pair['DIST_KM'])} km")
    
    print("\n🔮 PRÓXIMO BACK-TO-BACK (Simulación Predictiva Log5)")
    print("-" * 50)
    future = fetch_future_schedule(team_abbr, season_input)
    if future is not None:
        f1, f2 = future
        d1, d2 = f1['Date_Parsed'].strftime('%d %b'), f2['Date_Parsed'].strftime('%d %b')
        
        op1, op2 = f1['Opp'], f2['Opp']
        loc1 = "Visitante" if '@' in str(f1['Unnamed: 5']) else "Local"
        loc2 = "Visitante" if '@' in str(f2['Unnamed: 5']) else "Local"
        
        matchup1_str = f"@ {op1}" if loc1 == "Visitante" else f"vs. {op1}"
        matchup2_str = f"@ {op2}" if loc2 == "Visitante" else f"vs. {op2}"
        
        l1 = get_location(team_abbr, matchup1_str)
        l2 = get_location(team_abbr, matchup2_str)
        dist_km = haversine(l1[0][0], l1[0][1], l2[0][0], l2[0][1])
        
        opp1_wp = true_talent.get(op1, 0.500)
        opp2_wp = true_talent.get(op2, 0.500)
        
        p1 = predict_log5(t_wp, opp1_wp, is_home=(loc1=="Local"), is_b2b=False, dist_km=0)
        p2 = predict_log5(t_wp, opp2_wp, is_home=(loc2=="Local"), is_b2b=True, dist_km=dist_km)
        
        print(f"📅 Próximas Fechas: {d1} y {d2}")
        print(f"🚗 Ruta: Primero {loc1.lower()} contra {op1}, y al día siguiente {loc2.lower()} contra {op2}.")
        
        print("\nDetalle de Probabilidad por Partido:")
        print(f"1️⃣ {d1} | {matchup1_str:<12} | Rival WP: {opp1_wp*100:.1f}% -> P(W) = {p1*100:.1f}%")
        print(f"2️⃣ {d2} | {matchup2_str:<12} | Rival WP: {opp2_wp*100:.1f}% + Fatiga + Viaje ({int(dist_km)}km) -> P(W) = {p2*100:.1f}%")
        
        pWW = p1 * p2 * 100
        pWL = p1 * (1 - p2) * 100
        pLW = (1 - p1) * p2 * 100
        pLL = (1 - p1) * (1 - p2) * 100
        
        print("\n📊 ESCENARIOS LOG5")
        print("-" * 50)
        print(f"🟢 Ambos:              ~{pWW:.1f}%")
        print(f"🟡 El primero:         ~{pWL:.1f}%")
        print(f"🟡 El segundo:         ~{pLW:.1f}%")
        print(f"🔴 Ninguno:            ~{pLL:.1f}%")
    else:
        print("No se encontraron B2B futuros en el calendario (o error).")
        
    if not team_pairs.empty:
        total_b2b = len(team_pairs)
        ww, wl, lw, ll = team_pairs['IS_WW'].sum(), team_pairs['IS_WL'].sum(), team_pairs['IS_LW'].sum(), team_pairs['IS_LL'].sum()
        print("\n📈 CONTEXTO HISTÓRICO DE TEMPORADA")
        print("-" * 50)
        print(f"De los {total_b2b} back-to-backs ya jugados (Desempeño empírico):")
        print(f"- Pura victoria (Ambos): {ww}")
        print(f"- Ganan el primero (WL): {wl}")
        print(f"- Ganan el segundo (LW): {lw}")
        print(f"- Ninguno (LL): {ll}")
        print(f"Promedio de viaje en back-to-backs: {int(team_pairs['DIST_KM'].mean())} km")
    print("="*70 + "\n")

def main():
    print("="*60)
    print(" 🏀 CALCULADORA B2B NBA: MODELO LOG5 🏀 ")
    print("="*60)
    
    season_input_raw = input("Ingresa la temporada (ej. 25/26 o 23/24) [Enter: 23/24]: ").strip()
    if not season_input_raw: season_input_raw = '23/24'
    season_parsed = parse_season(season_input_raw)
        
    cutoff_input = input("Fecha de corte DD/MM/YYYY (ej. 15/02/2026) [Enter: Sin corte]: ").strip()
    cutoff_date = cutoff_input if cutoff_input else None
    
    team_input = input("Ingresa abreviatura de equipo a profundizar (ej. NYK, LAL) [Enter: Ver Todos]: ").strip()
    
    df = fetch_season_games(season_parsed)
    if df is None: sys.exit(1)
        
    pairs, stats, true_talent = process_back_to_backs(df, cutoff_date)
    
    if stats is None:
        print("No se encontraron suficientes datos.")
        sys.exit(0)
    
    if team_input:
        print_team_deep_dive(pairs, team_input, season_parsed, true_talent)
        
    print(f"\n Resultados Temporada: {season_parsed} - Performance Base Log5 en Average B2B")
    print("-" * 120)
    
    display_cols = ['TEAM_NAME', 'B2B_Jugados', 'Ganan Ambos', 'Ganan el Primero', 'Ganan el Segundo', 'Ninguno', 'AVG_DIST_B2B']
    print_table = stats[display_cols].to_string(index=False)
    for line in print_table.split('\n'):
        print(line)

if __name__ == "__main__":
    main()
