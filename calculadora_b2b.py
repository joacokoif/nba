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

TEAM_NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

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

def predict_log5(team_wp, opp_wp, is_home, is_b2b, dist_km, opp_b2b=False, opp_dist_km=0):
    """Fórmula Predictiva Log5 ajustada por factores de Ecosistema NBA"""
    home_adv = 0.035
    b2b_penalty = 0.045
    t_wp, o_wp = team_wp, opp_wp
    
    # Ajuste de Localia
    if is_home:
        t_wp += home_adv
        o_wp -= home_adv
    else:
        t_wp -= home_adv
        o_wp += home_adv
        
    # Ajuste de Fatiga Equipo Base
    if is_b2b:
        t_wp -= b2b_penalty
        if dist_km > 1000: t_wp -= 0.015
        if dist_km > 2000: t_wp -= 0.025
        
    # Ajuste de Fatiga Equipo Rival
    if opp_b2b:
        o_wp -= b2b_penalty
        if opp_dist_km > 1000: o_wp -= 0.015
        if opp_dist_km > 2000: o_wp -= 0.025
            
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

def fetch_bref_schedule(team_abbr, season_str):
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
            df['PREV_DATE'] = df['Date_Parsed'].shift(1)
            df['DAYS_REST'] = (df['Date_Parsed'] - df['PREV_DATE']).dt.days
            return df
    except Exception:
        pass
    return None

def fetch_future_schedule(team_abbr, season_str):
    df = fetch_bref_schedule(team_abbr, season_str)
    if df is not None:
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
    return None

def check_future_opponent_b2b(opp_abbr, match_date, season_str):
    df = fetch_bref_schedule(opp_abbr, season_str)
    if df is not None:
        match_row = df[df['Date_Parsed'] == match_date]
        if not match_row.empty:
            match_row = match_row.iloc[0]
            idx = df.index.get_loc(match_row.name)
            
            streak = 1
            curr_idx = idx
            total_dist = 0
            
            while curr_idx > 0:
                curr_row = df.iloc[curr_idx]
                if curr_row['DAYS_REST'] == 1:
                    streak += 1
                    
                    prev_row = df.iloc[curr_idx - 1]
                    opp1 = str(prev_row['Opponent'])
                    is_away_prev = '@' in str(prev_row.get('Unnamed: 5', ''))
                    prev_loc = get_location(opp_abbr, f"@ {opp1}" if is_away_prev else f"vs. {opp1}")
                    
                    opp2 = str(curr_row['Opponent'])
                    is_away_curr = '@' in str(curr_row.get('Unnamed: 5', ''))
                    curr_loc = get_location(opp_abbr, f"@ {opp2}" if is_away_curr else f"vs. {opp2}")
                    
                    dist = haversine(prev_loc[0][0], prev_loc[0][1], curr_loc[0][0], curr_loc[0][1])
                    total_dist += dist
                    
                    curr_idx -= 1
                else:
                    break
                    
            return streak, total_dist
    return 1, 0

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
    df['PREV_MATCHUP'] = df.groupby('TEAM_ID')['MATCHUP'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    
    # Map opp fatigue
    opp_fatigue_map = {}
    for idx, row in df.iterrows():
        if row['DAYS_REST'] == 1:
            curr_loc = get_location(row['TEAM_ABBREVIATION'], row['MATCHUP'])
            prev_loc = get_location(row['TEAM_ABBREVIATION'], row['PREV_MATCHUP'])
            dist = haversine(prev_loc[0][0], prev_loc[0][1], curr_loc[0][0], curr_loc[0][1])
            date_str = pd.to_datetime(row['GAME_DATE']).strftime('%Y-%m-%d')
            opp_fatigue_map[(date_str, row['TEAM_ABBREVIATION'])] = dist
            
    b2b_idx2 = df.index[df['DAYS_REST'] == 1]
    b2b_idx1 = b2b_idx2 - 1
    
    g1 = df.loc[b2b_idx1].reset_index(drop=True)
    g2 = df.loc[b2b_idx2].reset_index(drop=True)
    
    if g1.empty or g2.empty:
        return None, None, true_talent
        
    locs1 = [get_location(t, m) for t, m in zip(g1['TEAM_ABBREVIATION'], g1['MATCHUP'])]
    locs2 = [get_location(t, m) for t, m in zip(g2['TEAM_ABBREVIATION'], g2['MATCHUP'])]
    distances = [haversine(l1[0][0], l1[0][1], l2[0][0], l2[0][1]) for l1, l2 in zip(locs1, locs2)]
    
    # Fetch opp fatigue for each game explicitly
    opp_b2b1_list, opp_b2b2_list = [], []
    opp_dist1_list, opp_dist2_list = [], []
    
    for _, r in g1.iterrows():
        opp_abbr = get_location(r['TEAM_ABBREVIATION'], r['MATCHUP'])[2]
        d_str = pd.to_datetime(r['GAME_DATE']).strftime('%Y-%m-%d')
        if (d_str, opp_abbr) in opp_fatigue_map:
            opp_b2b1_list.append(True)
            opp_dist1_list.append(opp_fatigue_map[(d_str, opp_abbr)])
        else:
            opp_b2b1_list.append(False); opp_dist1_list.append(0)
            
    for _, r in g2.iterrows():
        opp_abbr = get_location(r['TEAM_ABBREVIATION'], r['MATCHUP'])[2]
        d_str = pd.to_datetime(r['GAME_DATE']).strftime('%Y-%m-%d')
        if (d_str, opp_abbr) in opp_fatigue_map:
            opp_b2b2_list.append(True)
            opp_dist2_list.append(opp_fatigue_map[(d_str, opp_abbr)])
        else:
            opp_b2b2_list.append(False); opp_dist2_list.append(0)
    
    w1 = (g1['WL'] == 'W')
    w2 = (g2['WL'] == 'W')

    pairs = pd.DataFrame({
        'TEAM': g1['TEAM_ABBREVIATION'],
        'TEAM_NAME': g1['TEAM_NAME'],
        'DATE_1': g1['GAME_DATE'],
        'MATCHUP_1': g1['MATCHUP'],
        'IS_H_1': [l[1] for l in locs1],
        'OPP_B2B_1': opp_b2b1_list,
        'OPP_DIST_1': opp_dist1_list,
        'WL_1': g1['WL'],
        'PTS_1': g1['PTS'],
        'DATE_2': g2['GAME_DATE'],
        'MATCHUP_2': g2['MATCHUP'],
        'IS_H_2': [l[1] for l in locs2],
        'OPP_B2B_2': opp_b2b2_list,
        'OPP_DIST_2': opp_dist2_list,
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
        print(f"No hay B2B históricos disputados por {team_abbr} hasta la fecha de corte.")
    else:
        print(" ÚLTIMO BACK-TO-BACK DISPUTADO")
        print("-" * 50)
        last = team_pairs.iloc[-1]
        res1 = '✅' if last['WL_1'] == 'W' else '❌'
        res2 = '✅' if last['WL_2'] == 'W' else '❌'
        
        d1 = pd.to_datetime(last['DATE_1']).strftime('%d %b')
        d2 = pd.to_datetime(last['DATE_2']).strftime('%d %b')
        m1 = last['MATCHUP_1']
        m2 = last['MATCHUP_2']
        p1 = last['PTS_1']
        p2 = last['PTS_2']
        
        print(f"Game 1 ({d1}) | {m1:<12} | {res1} {last['TEAM']} {p1}")
        if last['OPP_B2B_1']: print("        (El rival se encontraba en su 2do partido del B2B)")
        print(f"Game 2 ({d2}) | {m2:<12} | {res2} {last['TEAM']} {p2}")
        if last['OPP_B2B_2']: print("        (El rival se encontraba en su 2do partido del B2B)")
        print(f"Distancia del vuelo: {int(last['DIST_KM'])} km\n")
        
        # Nuevas metricas pedidas por el usuario
        total_b2b = len(team_pairs)
        w2_count = len(team_pairs[team_pairs['WL_2'] == 'W'])
        opp_b2b_g1_count = len(team_pairs[team_pairs['OPP_B2B_1'] == True])
        
        shared_g2_df = team_pairs[team_pairs['OPP_B2B_2'] == True]
        opp_b2b_g2_count = len(shared_g2_df)
        w2_shared_count = len(shared_g2_df[shared_g2_df['WL_2'] == 'W'])
        shared_win_rate = (w2_shared_count / opp_b2b_g2_count * 100) if opp_b2b_g2_count > 0 else 0.0
        
        print(" RENDIMIENTO HISTÓRICO ESPECÍFICO")
        print("-" * 50)
        print(f"🏀 B2B Jugados: {total_b2b}")
        print(f"🏅 Win Rate en el 2do Partido (Agotados): {w2_count}/{total_b2b} ({w2_count/total_b2b*100:.1f}%)")
        print(f"⚔️  Veces que el rival jugaba su segundo partido del B2B en el 1er juego: {opp_b2b_g1_count}")
        print(f"⚔️  Veces que el rival jugaba su segundo partido del B2B en el 2do juego: {opp_b2b_g2_count}")
        if opp_b2b_g2_count > 0:
            print(f"🔥 Prob. ganar Game 2 estando AMBOS agotados: {w2_shared_count}/{opp_b2b_g2_count} ({shared_win_rate:.1f}%)\n")
        else:
            print("\n")
    
    print("\n🔮 PRÓXIMO BACK-TO-BACK (Simulación Predictiva Log5)")
    print("-" * 50)
    future = fetch_future_schedule(team_abbr, season_input)
    if future is not None:
        f1, f2 = future
        d1, d2 = f1['Date_Parsed'].strftime('%d %b'), f2['Date_Parsed'].strftime('%d %b')
        
        op1, op2 = f1['Opponent'], f2['Opponent']
        loc1 = "Visitante" if '@' in str(f1.get('Unnamed: 5', '')) else "Local"
        loc2 = "Visitante" if '@' in str(f2.get('Unnamed: 5', '')) else "Local"
        
        matchup1_str = f"@ {op1}" if loc1 == "Visitante" else f"vs. {op1}"
        matchup2_str = f"@ {op2}" if loc2 == "Visitante" else f"vs. {op2}"
        
        l1 = get_location(team_abbr, matchup1_str)
        l2 = get_location(team_abbr, matchup2_str)
        dist_km = haversine(l1[0][0], l1[0][1], l2[0][0], l2[0][1])
        
        opp1_wp = true_talent.get(op1, 0.500)
        opp2_wp = true_talent.get(op2, 0.500)
        
        op1_abbr = TEAM_NAME_TO_ABBR.get(op1, op1)
        op2_abbr = TEAM_NAME_TO_ABBR.get(op2, op2)
        
        op1_streak, op1_dist = check_future_opponent_b2b(op1_abbr, f1['Date_Parsed'], season_input)
        op2_streak, op2_dist = check_future_opponent_b2b(op2_abbr, f2['Date_Parsed'], season_input)
        
        p1 = predict_log5(t_wp, opp1_wp, is_home=(loc1=="Local"), is_b2b=False, dist_km=0, opp_b2b=(op1_streak > 1), opp_dist_km=op1_dist)
        p2 = predict_log5(t_wp, opp2_wp, is_home=(loc2=="Local"), is_b2b=True, dist_km=dist_km, opp_b2b=(op2_streak > 1), opp_dist_km=op2_dist)
        
        print(f"📅 Próximas Fechas: {d1} y {d2}")
        print(f"🚗 Ruta: Primero {loc1.lower()} contra {op1}, y al día siguiente {loc2.lower()} contra {op2}.")
        
        if op1_streak == 2: print(f"⚠️  ¡Ojo! {op1} estará jugando su 2do partido del B2B para este juego (viajando {int(op1_dist)}km).")
        elif op1_streak >= 3: print(f"⚠️  ¡PELIGRO EXTREMO! {op1} vendrá de una seguidilla consecutiva de {op1_streak} partidos (viajando {int(op1_dist)}km).")
        
        if op2_streak == 2: print(f"⚠️  ¡Ojo! {op2} estará jugando su 2do partido del B2B para este juego (viajando {int(op2_dist)}km).")
        elif op2_streak >= 3: print(f"⚠️  ¡PELIGRO EXTREMO! {op2} vendrá de una seguidilla consecutiva de {op2_streak} partidos (viajando {int(op2_dist)}km).")
        
        p1_opp_str = f" - Penalidad Rival B2B ({int(op1_dist)}km)" if op1_streak > 1 else ""
        p2_opp_str = f" - Penalidad Rival B2B ({int(op2_dist)}km)" if op2_streak > 1 else ""

        print("\nDetalle de Probabilidad por Partido:")
        print(f"1️⃣ {d1} | {matchup1_str:<12} | Rival WP: {opp1_wp*100:.1f}%{p1_opp_str} -> P(W) = {p1*100:.1f}%")
        print(f"2️⃣ {d2} | {matchup2_str:<12} | Rival WP: {opp2_wp*100:.1f}% + Fatiga Propia + Viaje ({int(dist_km)}km){p2_opp_str} -> P(W) = {p2*100:.1f}%")
        
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
        if team_input not in pairs['TEAM'].values:
            print(f"\n⚠️ Aviso: {team_input} no registra juegos B2B disputados previamente en este corte.")
        print_team_deep_dive(pairs, team_input, season_parsed, true_talent)
    else:
        teams = pairs['TEAM'].unique()
        for t in teams:
            print_team_deep_dive(pairs, t, season_parsed, true_talent)
    print("-" * 120)
    
    display_cols = ['TEAM_NAME', 'B2B_Jugados', 'Ganan Ambos', 'Ganan el Primero', 'Ganan el Segundo', 'Ninguno', 'AVG_DIST_B2B']
    print_table = stats[display_cols].to_string(index=False)
    for line in print_table.split('\n'):
        print(line)

if __name__ == "__main__":
    main()
