import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import joblib
import requests
import json
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURACIÓN DE LA API-Football ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"
API_BASE_URL = f"https://{RAPIDAPI_HOST}/v3"

# --- CACHE EN MEMORIA PARA LAS LLAMADAS A LA API (Python) ---
api_cache = {}

def cached_api_call_python(endpoint, params, ttl_seconds=3600):
    cache_key = f"{endpoint}-{json.dumps(params, sort_keys=True)}"
    now = datetime.now()

    if cache_key in api_cache and (now - api_cache[cache_key]['timestamp']).total_seconds() < ttl_seconds:
        print(f"⚡️ Python Cache HIT: {endpoint} - {cache_key[:50]}...")
        return api_cache[cache_key]['data']

    print(f"🌍 Python Cache MISS: {endpoint} - {cache_key[:50]}... Fetching from API...")

    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST,
    }
    url = f"{API_BASE_URL}{endpoint}"
    
    MAX_RETRIES = 3
    for current_retry in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10) 
            response.raise_for_status() 
            data = response.json()

            if data.get('errors') and len(data['errors']) > 0:
                error_msg = ", ".join(data['errors'].values())
                raise Exception(f"API-Football Error: {error_msg}")

            api_cache[cache_key] = {'data': data, 'timestamp': now}
            return data

        except requests.exceptions.RequestException as e:
            if response is not None and response.status_code == 429:
                delay_time = (2 ** current_retry) + (np.random.rand() * 0.5) 
                print(f"⚠️ Rate limit exceeded (429) for {endpoint}. Retrying in {delay_time:.2f} seconds... (Attempt {current_retry + 1}/{MAX_RETRIES})")
                import time
                time.sleep(delay_time)
            else:
                print(f"❌ Error en cached_api_call_python para {endpoint}: {e}")
                raise
        except Exception as e:
            print(f"❌ Error inesperado en cached_api_call_python para {endpoint}: {e}")
            raise
    
    raise Exception(f"Fallo al obtener datos de {endpoint} después de {MAX_RETRIES} reintentos debido a límites de tasa.")


LEAGUE_ID_TO_NAME = {
    39: 'Premier League',
    140: 'La Liga España',
    135: 'Serie A ', 
    253: 'MLS y Legues Cup',
    71: 'Liga MX', 
    2: 'ChampionsLegue',
    3: 'Europa League',
    848: 'Conference League',
    262: 'Liga MX', 
    98: 'JLeague', 
    128: 'Liga Profesional Argentina',
    265: 'Primera División', 
}
LEAGUE_NAME_TO_ID = {v: k for k, v in LEAGUE_ID_TO_NAME.items()}


MODEL_DIR = 'trained_models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

leagues = {
    'Conference League': 'datoss_ConferenceLeague.csv',
    'MLS y Legues Cup': 'datos_partidos.csv',
    'Europa League': 'datos_EuropaLeague.csv',
    'Brasileirao': 'datos_Brasileirao.csv',
    'Eredivisie': 'datos_Eredivisie.csv',
    'Liga MX': 'datos_LigaMX.csv', 
    'Liga Portuguesa': 'datos_LigaNos.csv',
    'Liga Chilena': 'datos_ChileLigue.csv',
    'Liga Allsvenskan Suecia': 'datos_Allsvenskan.csv',
    'EFL League One': 'EFL League One.csv',
    'Premier League': 'Premier League.csv',
    'La Liga España': 'La Liga.csv',
    'Pro League Belgica': 'Belgica_Pro_League.csv',
    'USL Championship USA':'USL_Champions_USA.csv',
    'ChampionsLegue': 'ChampionsLegue.csv',
    'Ligue One France': 'LigueOneFrance.csv',
    'Serie A ': 'SerieAItalia.csv',
    'JLeague': 'JLeague.csv', 
    'Bundesliga' : 'Bundesliga.csv',
    'Ascenso MX' : 'MXAscenso.csv',
    'NationsLegueEUROPA': 'NationsLegueEUROPA.csv',
    'Liga Profesional Argentina': 'datos_LigaProfesionalArgentina.csv', 
    'Primera División': 'datos_PrimeraDivision.csv', 
}

def normalize_team_name(name):
    if isinstance(name, str):
        name = name.lower()
        name = re.sub(r'[^a-z\s]', '', name) 
        
        replacements = {
            r'\b(fc|club|de|santiago|atlético|atletico|unidas|unidos|unam|pumas|athletic|real|united|city|town|villa|sporting|cd|cf|sd|ac|as|ud|rc|sk|utd|st|gremio|sao|paulo|monterrey|tigres|queretaro|cordoba|central|universidad|nacional|autonoma|mexico)\b': '',
            r'\b(cf|ca|c a|k|kv|bvb|tsg)\b': '', 
            r'\s+': ' ', 
        }
        for pattern, replacement in replacements.items():
            name = re.sub(pattern, replacement, name)
        
        name = name.strip() 
        return name
    return name

dataframes = {}
for name, file in leagues.items():
    try:
        df = pd.read_csv(file)
        df['home_team_name_normalized'] = df['home_team_name'].apply(normalize_team_name)
        df['away_team_name_normalized'] = df['away_team_name'].apply(normalize_team_name)

        df['home_team_goals_avg'] = df.groupby('home_team_name_normalized')['home_team_goal_count'].transform('mean').fillna(df['home_team_goal_count'].mean())
        # CORRECCIÓN: Eliminado el paréntesis extra aquí
        df['away_team_goals_avg'] = df.groupby('away_team_name_normalized')['away_team_goal_count'].transform('mean').fillna(df['away_team_goal_count'].mean())
        df['home_team_goals_conceded_avg'] = df.groupby('home_team_name_normalized')['away_team_goal_count'].transform('mean').fillna(df['away_team_goal_count'].mean())
        df['away_team_goals_conceded_avg'] = df.groupby('away_team_name_normalized')['home_team_goal_count'].transform('mean').fillna(df['home_team_goal_count'].mean())
        df['home_team_yellow_cards_avg'] = df.groupby('home_team_name_normalized')['home_team_yellow_cards'].transform('mean').fillna(df['home_team_yellow_cards'].mean())
        df['away_team_yellow_cards_avg'] = df.groupby('away_team_name_normalized')['away_team_yellow_cards'].transform('mean').fillna(df['away_team_yellow_cards'].mean())
        df['home_team_corner_count_avg'] = df.groupby('home_team_name_normalized')['home_team_corner_count'].transform('mean').fillna(df['home_team_corner_count'].mean())
        df['away_team_corner_count_avg'] = df.groupby('away_team_name_normalized')['away_team_corner_count'].transform('mean').fillna(df['away_team_corner_count'].mean())
        df['first_half_goals'] = df['total_goals_at_half_time'].fillna(df['total_goals_at_half_time'].mean())
        df['second_half_goals'] = df['total_goal_count'] - df['total_goals_at_half_time']
        df['second_half_goals'] = df['second_half_goals'].fillna(df['second_half_goals'].mean())
        df['over_2_5_goals'] = df.apply(lambda row: 1 if row['total_goal_count'] > 2.5 else 0, axis=1)
        df['total_corners'] = (df['home_team_corner_count'] + df['away_team_corner_count']).fillna((df['home_team_corner_count'] + df['away_team_corner_count']).mean())
        df['home_team_possession_avg'] = df.groupby('home_team_name_normalized')['home_team_possession'].transform('mean').fillna(df['home_team_possession'].mean())
        df['away_team_possession_avg'] = df.groupby('away_team_name_normalized')['away_team_possession'].transform('mean').fillna(df['away_team_possession'].mean())
        df['total_shots'] = (df['home_team_shots'] + df['away_team_shots']).fillna((df['home_team_shots'] + df['away_team_shots']).mean())
        df['total_shots_on_target'] = (df['home_team_shots_on_target'] + df['away_team_shots_on_target']).fillna((df['home_team_shots_on_target'] + df['away_team_shots_on_target']).mean())
        df['total_yellow_cards'] = (df['home_team_yellow_cards'] + df['away_team_yellow_cards']).fillna((df['home_team_yellow_cards'] + df['away_team_yellow_cards']).mean())
        df['total_red_cards'] = (df['home_team_red_cards'] + df['away_team_red_cards']).fillna((df['home_team_red_cards'] + df['away_team_red_cards']).mean())
        df['home_goal_diff_avg'] = df['home_team_goals_avg'] - df['home_team_goals_conceded_avg']
        df['away_goal_diff_avg'] = df['away_team_goals_avg'] - df['away_team_goals_conceded_avg']
        
        league_avg_goals_for = df['total_goal_count'].mean() / 2 
        league_avg_goals_against = df['total_goal_count'].mean() / 2

        df['home_attack_strength'] = (df['home_team_goals_avg'] / league_avg_goals_for).fillna(1.0)
        df['home_defense_strength'] = (df['home_team_goals_conceded_avg'] / league_avg_goals_against).fillna(1.0)
        df['away_attack_strength'] = (df['away_team_goals_avg'] / league_avg_goals_for).fillna(1.0)
        df['away_defense_strength'] = (df['away_team_goals_conceded_avg'] / league_avg_goals_against).fillna(1.0)
        df['possession_diff'] = df['home_team_possession_avg'] - df['away_team_possession_avg']
        df['shots_on_target_diff'] = (df['home_team_shots_on_target'] - df['away_team_shots_on_target']).fillna(0)
        df['winner_encoded'] = df.apply(lambda row: 0 if row['home_team_goal_count'] > row['away_team_goal_count'] else 2 if row['home_team_goal_count'] < row['away_team_goal_count'] else 1, axis=1)
        df = df.fillna(df.mean(numeric_only=True)) 
        
        dataframes[name] = df
        print(f"Datos cargados y preprocesados para {name}.")
    except Exception as e:
        print(f"Error al cargar o preprocesar datos para {name} desde {file}: {e}")

def predict_match_poisson(home_goals_avg, away_goals_avg):
    home_goals_pred = poisson.pmf(np.arange(0, 6), home_goals_avg) 
    away_goals_pred = poisson.pmf(np.arange(0, 6), away_goals_avg)
    match_outcomes = np.outer(home_goals_pred, away_goals_pred)
    home_win_prob = np.sum(np.tril(match_outcomes, -1))
    draw_prob = np.sum(np.diag(match_outcomes))
    away_win_prob = np.sum(np.triu(match_outcomes, 1))
    return home_win_prob, draw_prob, away_win_prob

def predict_goals_poisson(home_goals_avg, away_goals_avg):
    home_goals_pred = poisson.pmf(np.arange(0, 6), home_goals_avg)
    away_goals_pred = poisson.pmf(np.arange(0, 6), away_goals_avg)
    match_outcomes = np.outer(home_goals_pred, away_goals_pred)
    over_2_5_prob = np.sum(match_outcomes[np.where(np.add.outer(np.arange(0,6), np.arange(0,6)) > 2.5)])
    btts_prob = 1 - (np.sum(match_outcomes[0, :]) + np.sum(match_outcomes[:, 0]) - match_outcomes[0, 0])
    return over_2_5_prob, btts_prob

def build_and_train_models(data, league_name):
    features_cols = [
        'home_team_goals_avg', 'away_team_goals_avg',
        'home_team_goals_conceded_avg', 'away_team_goals_conceded_avg',
        'home_team_yellow_cards_avg', 'away_team_yellow_cards_avg',
        'home_team_corner_count_avg', 'away_team_corner_count_avg',
        'home_team_possession_avg', 'away_team_possession_avg',
        'home_attack_strength', 'home_defense_strength',
        'away_attack_strength', 'away_defense_strength',
        'possession_diff', 'shots_on_target_diff', 
        'total_yellow_cards', 'total_red_cards', 
        'total_shots', 'total_shots_on_target' 
    ]
    
    data_filtered = data.dropna(subset=features_cols)
    if data_filtered.empty:
        print(f"Advertencia: No hay datos suficientes para entrenar modelos para {league_name} después de filtrar NaN.")
        return None

    features = data_filtered[features_cols]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{league_name}_scaler.pkl'))
    
    X_train_o25, X_test_o25, y_train_o25, y_test_o25 = train_test_split(features_scaled, data_filtered['over_2_5_goals'], test_size=0.2, random_state=42)
    X_train_corners, X_test_corners, y_train_corners, y_test_corners = train_test_split(features_scaled, data_filtered['total_corners'], test_size=0.2, random_state=42)
    X_train_winner, X_test_winner, y_train_winner, y_test_winner = train_test_split(features_scaled, data_filtered['winner_encoded'], test_size=0.2, random_state=42)
    X_train_fh, X_test_fh, y_train_fh, y_test_fh = train_test_split(features_scaled, data_filtered['first_half_goals'], test_size=0.2, random_state=42)
    X_train_sh, X_test_sh, y_train_sh, y_test_sh = train_test_split(features_scaled, data_filtered['second_half_goals'], test_size=0.2, random_state=42)

    model_o25 = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_o25.shape[1],)), 
        Dropout(0.3), 
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model_o25.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    model_winner_nn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_winner.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax') 
    ])
    model_winner_nn.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model_corners = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_corners.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    model_corners.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model_fh = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_fh.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='relu') 
    ])
    model_fh.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model_sh = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_sh.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='relu') 
    ])
    model_sh.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    model_winner_svm = SVC(probability=True, random_state=42) 

    print(f"Entrenando modelos para la liga: {league_name}")
    
    model_o25.fit(X_train_o25, y_train_o25, epochs=20, batch_size=32, validation_data=(X_test_o25, y_test_o25), verbose=0)
    model_o25.save(os.path.join(MODEL_DIR, f'{league_name}_model_o25.h5'))
    
    model_corners.fit(X_train_corners, y_train_corners, epochs=20, batch_size=32, validation_data=(X_test_corners, y_test_corners), verbose=0)
    model_corners.save(os.path.join(MODEL_DIR, f'{league_name}_model_corners.h5'))
    
    model_winner_nn.fit(X_train_winner, y_train_winner, epochs=20, batch_size=32, validation_data=(X_test_winner, y_test_winner), verbose=0)
    model_winner_nn.save(os.path.join(MODEL_DIR, f'{league_name}_model_winner_nn.h5'))

    model_fh.fit(X_train_fh, y_train_fh, epochs=20, batch_size=32, validation_data=(X_test_fh, y_test_fh), verbose=0)
    model_fh.save(os.path.join(MODEL_DIR, f'{league_name}_model_fh.h5'))

    model_sh.fit(X_train_sh, y_train_sh, epochs=20, batch_size=32, validation_data=(X_test_sh, y_test_sh), verbose=0)
    model_sh.save(os.path.join(MODEL_DIR, f'{league_name}_model_sh.h5'))
    
    model_winner_svm.fit(X_train_winner, y_train_winner)
    joblib.dump(model_winner_svm, os.path.join(MODEL_DIR, f'{league_name}_model_winner_svm.pkl'))
    
    return {
        'o25_model': model_o25,
        'corners_model': model_corners,
        'winner_nn_model': model_winner_nn,
        'winner_svm_model': model_winner_svm,
        'fh_model': model_fh,
        'sh_model': model_sh,
        'scaler': scaler,
        'features_cols': features_cols 
    }

# Almacenar modelos cargados en memoria
loaded_models = {}

# Cargar y entrenar modelos para cada liga (o cargar si ya existen)
for league, df in dataframes.items():
    if not df.empty:
        scaler_path = os.path.join(MODEL_DIR, f'{league}_scaler.pkl')
        model_o25_path = os.path.join(MODEL_DIR, f'{league}_model_o25.h5')
        model_corners_path = os.path.join(MODEL_DIR, f'{league}_model_corners.h5')
        model_winner_nn_path = os.path.join(MODEL_DIR, f'{league}_model_winner_nn.h5')
        model_winner_svm_path = os.path.join(MODEL_DIR, f'{league}_model_winner_svm.pkl')
        model_fh_path = os.path.join(MODEL_DIR, f'{league}_model_fh.h5')
        model_sh_path = os.path.join(MODEL_DIR, f'{league}_model_sh.h5')

        if (os.path.exists(scaler_path) and
            os.path.exists(model_o25_path) and
            os.path.exists(model_corners_path) and
            os.path.exists(model_winner_nn_path) and
            os.path.exists(model_winner_svm_path) and
            os.path.exists(model_fh_path) and
            os.path.exists(model_sh_path)):
            
            print(f"Cargando modelos pre-entrenados para la liga: {league}")
            try:
                loaded_models[league] = {
                    'scaler': joblib.load(scaler_path),
                    'o25_model': load_model(model_o25_path),
                    'corners_model': load_model(model_corners_path),
                    'winner_nn_model': load_model(model_winner_nn_path),
                    'winner_svm_model': joblib.load(model_winner_svm_path),
                    'fh_model': load_model(model_fh_path),
                    'sh_model': load_model(model_sh_path),
                    'features_cols': [ 
                        'home_team_goals_avg', 'away_team_goals_avg',
                        'home_team_goals_conceded_avg', 'away_team_goals_conceded_avg',
                        'home_team_yellow_cards_avg', 'away_team_yellow_cards_avg',
                        'home_team_corner_count_avg', 'away_team_corner_count_avg',
                        'home_team_possession_avg', 'away_team_possession_avg',
                        'home_attack_strength', 'home_defense_strength',
                        'away_attack_strength', 'away_defense_strength',
                        'possession_diff', 'shots_on_target_diff', 
                        'total_yellow_cards', 'total_red_cards', 
                        'total_shots', 'total_shots_on_target' 
                    ]
                }
            except Exception as e:
                print(f"Error al cargar modelos para {league}: {e}. Intentando re-entrenar.")
                loaded_models[league] = build_and_train_models(df, league)
        else:
            print(f"Modelos no encontrados para {league}. Entrenando...")
            loaded_models[league] = build_and_train_models(df, league)
    else:
        print(f"Advertencia: El DataFrame para la liga '{league}' está vacío. No se entrenarán/cargarán modelos para esta liga.")


# Función para obtener las características de un partido futuro
def get_match_features(home_team_name_input, away_team_name_input, league_name):
    current_league_df = dataframes.get(league_name)
    if current_league_df is None or current_league_df.empty:
        raise ValueError(f"No hay datos históricos cargados para la liga: {league_name}")

    # Normalizar nombres de equipo de entrada
    home_team_name_normalized_input = normalize_team_name(home_team_name_input)
    away_team_name_normalized_input = normalize_team_name(away_team_name_input)

    # Buscar estadísticas usando nombres normalizados
    home_stats_row = current_league_df[current_league_df['home_team_name_normalized'] == home_team_name_normalized_input]
    away_stats_row = current_league_df[current_league_df['away_team_name_normalized'] == away_team_name_normalized_input]

    # --- Manejo de equipos no encontrados ---
    # Si no se encuentran estadísticas para un equipo, usaremos la media de la liga como valor por defecto
    # para sus características, y lanzaremos una advertencia.
    
    # Obtener la media de las características de la liga para usar como fallback
    league_mean_features = current_league_df.mean(numeric_only=True)

    def get_stat_value_robust(df_stats_row, col_name, default_mean_value):
        if not df_stats_row.empty and col_name in df_stats_row.columns and not df_stats_row[col_name].isnull().all():
            return df_stats_row[col_name].iloc[0]
        return default_mean_value 

    match_features_dict = {}
    
    features_cols_template = [ 
        'home_team_goals_avg', 'away_team_goals_avg',
        'home_team_goals_conceded_avg', 'away_team_goals_conceded_avg',
        'home_team_yellow_cards_avg', 'away_team_yellow_cards_avg',
        'home_team_corner_count_avg', 'away_team_corner_count_avg',
        'home_team_possession_avg', 'away_team_possession_avg',
        'home_attack_strength', 'home_defense_strength',
        'away_attack_strength', 'away_defense_strength',
        'possession_diff', 'shots_on_target_diff', 
        'total_yellow_cards', 'total_red_cards', 
        'total_shots', 'total_shots_on_target' 
    ]

    for col in features_cols_template:
        if col.startswith('home_'):
            match_features_dict[col] = get_stat_value_robust(home_stats_row, col, league_mean_features.get(col, 0.0))
        elif col.startswith('away_'):
            match_features_dict[col] = get_stat_value_robust(away_stats_row, col, league_mean_features.get(col, 0.0))
        else: 
            if col == 'possession_diff':
                home_pos = get_stat_value_robust(home_stats_row, 'home_team_possession_avg', league_mean_features.get('home_team_possession_avg', 0.0))
                away_pos = get_stat_value_robust(away_stats_row, 'away_team_possession_avg', league_mean_features.get('away_team_possession_avg', 0.0))
                match_features_dict[col] = home_pos - away_pos
            elif col == 'shots_on_target_diff':
                home_sot = get_stat_value_robust(home_stats_row, 'home_team_shots_on_target', league_mean_features.get('home_team_shots_on_target', 0.0))
                away_sot = get_stat_value_robust(away_stats_row, 'away_team_shots_on_target', league_mean_features.get('away_team_shots_on_target', 0.0))
                match_features_dict[col] = home_sot - away_sot
            elif col == 'total_yellow_cards':
                home_yc = get_stat_value_robust(home_stats_row, 'home_team_yellow_cards', league_mean_features.get('home_team_yellow_cards', 0.0))
                away_yc = get_stat_value_robust(away_stats_row, 'away_team_yellow_cards', league_mean_features.get('away_team_yellow_cards', 0.0))
                match_features_dict[col] = home_yc + away_yc
            elif col == 'total_red_cards':
                home_rc = get_stat_value_robust(home_stats_row, 'home_team_red_cards', league_mean_features.get('home_team_red_cards', 0.0))
                away_rc = get_stat_value_robust(away_stats_row, 'away_team_red_cards', league_mean_features.get('away_team_red_cards', 0.0))
                match_features_dict[col] = home_rc + away_rc
            elif col == 'total_shots':
                home_shots = get_stat_value_robust(home_stats_row, 'home_team_shots', league_mean_features.get('home_team_shots', 0.0))
                away_shots = get_stat_value_robust(away_stats_row, 'away_team_shots', league_mean_features.get('away_team_shots', 0.0))
                match_features_dict[col] = home_shots + away_shots
            elif col == 'total_shots_on_target':
                home_sot = get_stat_value_robust(home_stats_row, 'home_team_shots_on_target', league_mean_features.get('home_team_shots_on_target', 0.0))
                away_sot = get_stat_value_robust(away_stats_row, 'away_team_shots_on_target', league_mean_features.get('away_team_shots_on_target', 0.0))
                match_features_dict[col] = home_sot + away_sot
            else:
                match_features_dict[col] = league_mean_features.get(col, 0.0) 

    for col in features_cols_template: 
        if col not in match_features_dict or pd.isna(match_features_dict[col]):
            print(f"Advertencia final: Característica '{col}' faltante o NaN para la predicción. Usando la media de la liga o 0.")
            match_features_dict[col] = league_mean_features.get(col, 0.0)
            
    return match_features_dict

def predict_for_teams(home_team_name_input, away_team_name_input, league_name):
    # Normalizar los nombres de entrada para la búsqueda
    normalized_home_team_name = normalize_team_name(home_team_name_input)
    normalized_away_team_name = normalize_team_name(away_team_name_input)

    if league_name not in loaded_models or loaded_models[league_name] is None:
        return {"error": f"Modelos de ML no disponibles para la liga: {league_name}. Asegúrate de que la liga exista y tenga datos para el entrenamiento."}

    models = loaded_models[league_name]
    scaler = models['scaler']
    features_cols = models['features_cols']
    
    try:
        # Obtener las características del partido, ahora con normalización y manejo de fallbacks
        match_features_dict = get_match_features(home_team_name_input, away_team_name_input, league_name)
    except ValueError as e:
        return {"error": str(e)}

    match_features_array = np.array([[match_features_dict[col] for col in features_cols]])
    match_features_scaled = scaler.transform(match_features_array)

    results = {}

    nn_o25_prob = models['o25_model'].predict(match_features_scaled)[0][0]
    nn_corners_pred = models['corners_model'].predict(match_features_scaled)[0][0]
    nn_winner_probs = models['winner_nn_model'].predict(match_features_scaled)[0] # [Home, Draw, Away]
    nn_first_half_goals = models['fh_model'].predict(match_features_scaled)[0][0]
    nn_second_half_goals = models['sh_model'].predict(match_features_scaled)[0][0]

    svm_winner_probs = models['winner_svm_model'].predict_proba(match_features_scaled)[0] # [Home, Draw, Away]

    home_goals_avg_poisson = match_features_dict['home_team_goals_avg']
    away_goals_avg_poisson = match_features_dict['away_team_goals_avg']
    poisson_home_win_prob, poisson_draw_prob, poisson_away_win_prob = predict_match_poisson(home_goals_avg_poisson, away_goals_avg_poisson)
    poisson_o25_prob, poisson_btts_prob = predict_goals_poisson(home_goals_avg_poisson, away_goals_avg_poisson)

    # --- SÍNTESIS DE LA PREDICCIÓN (El "toque profesional") ---
    
    # 1. Combinar probabilidades de Ganador (NN y SVM con más peso a NN)
    WEIGHT_NN = 0.6
    WEIGHT_SVM = 0.3
    WEIGHT_POISSON = 0.1

    combined_home_prob = (nn_winner_probs[0] * WEIGHT_NN + svm_winner_probs[0] * WEIGHT_SVM + poisson_home_win_prob * WEIGHT_POISSON)
    combined_draw_prob = (nn_winner_probs[1] * WEIGHT_NN + svm_winner_probs[1] * WEIGHT_SVM + poisson_draw_prob * WEIGHT_POISSON)
    combined_away_prob = (nn_winner_probs[2] * WEIGHT_NN + svm_winner_probs[2] * WEIGHT_SVM + poisson_away_win_prob * WEIGHT_POISSON)

    total_combined_prob_sum = combined_home_prob + combined_draw_prob + combined_away_prob
    if total_combined_prob_sum > 0:
        combined_home_prob /= total_combined_prob_sum
        combined_draw_prob /= total_combined_prob_sum
        combined_away_prob /= total_combined_prob_sum
    else: 
        combined_home_prob, combined_draw_prob, combined_away_prob = 0.33, 0.34, 0.33

    predicted_winner_name = ""
    max_combined_prob = max(combined_home_prob, combined_draw_prob, combined_away_prob)
    
    if max_combined_prob == combined_home_prob:
        predicted_winner_name = home_team_name_input
        confidence_for_pick = combined_home_prob
    elif max_combined_prob == combined_away_prob:
        predicted_winner_name = away_team_name_input
        confidence_for_pick = combined_away_prob
    else:
        predicted_winner_name = "Empate"
        confidence_for_pick = combined_draw_prob

    simulated_odd = 1 / confidence_for_pick if confidence_for_pick > 0 else 1.0 

    # 3. Calcular el Marcador Más Probable (coherente con el ganador y goles esperados)
    # Usar los goles esperados de la NN para mayor coherencia
    estimated_total_goals = nn_first_half_goals + nn_second_half_goals
    
    # Simple heurística para distribuir goles según la probabilidad de ganador
    # Asegurarse de que el divisor no sea cero
    prob_sum_for_score_dist = combined_home_prob + combined_away_prob + combined_draw_prob / 2
    if prob_sum_for_score_dist == 0: 
        prob_sum_for_score_dist = 1.0 

    home_score_raw = estimated_total_goals * ((combined_home_prob + combined_draw_prob / 2) / prob_sum_for_score_dist)
    away_score_raw = estimated_total_goals * ((combined_away_prob + combined_draw_prob / 2) / prob_sum_for_score_dist)

    home_score = int(round(home_score_raw))
    away_score = int(round(away_score_raw))

    # Ajustar para asegurar coherencia con el ganador
    if predicted_winner_name == home_team_name_input:
        if home_score <= away_score: 
            home_score = max(away_score + 1, 1) 
    elif predicted_winner_name == away_team_name_input:
        if away_score <= home_score: 
            away_score = max(home_score + 1, 1) 
    else: # Empate
        if home_score != away_score:
            avg_score = int(round(estimated_total_goals / 2))
            home_score = avg_score
            away_score = avg_score
        if home_score == 0 and away_score == 0 and estimated_total_goals > 0.5: 
            home_score = 1
            away_score = 1

    most_probable_score = f"{int(max(0, home_score))} - {int(max(0, away_score))}" 

    # 4. Determinar BTTS y Over/Under (basado en NN o Poisson si NN no es concluyente)
    # Priorizamos la NN para O2.5
    final_o25_bool = nn_o25_prob > 0.5
    final_o25_display = "+2.5" if final_o25_bool else "-2.5"

    # Para BTTS, si el marcador más probable es 0-X o X-0, entonces BTTS es NO.
    # Si ambos equipos anotan en el marcador probable, entonces BTTS es SÍ.
    final_btts_bool = (home_score > 0 and away_score > 0)
    final_btts_display = "Sí" if final_btts_bool else "No"
    
    # 5. Generar el Consejo (Advice)
    advice_str = f"Nuestro análisis avanzado sugiere que **{predicted_winner_name}** tiene la mayor probabilidad de éxito."
    
    if predicted_winner_name != "Empate":
        advice_str += f" El marcador más probable es **{most_probable_score}**, indicando una victoria {'ajustada' if abs(home_score - away_score) <= 1 else 'clara'}."
    else:
        advice_str += f" El marcador más probable es **{most_probable_score}**, anticipando un duelo muy parejo."

    apuesta_complementaria_text = ""
    combinacion_interesante_text = ""

    # Lógica para la apuesta complementaria y combinación
    # Prioridad: Under 2.5 (si la NN lo apoya fuertemente)
    if nn_o25_prob < 0.40: # Si la NN predice Under 2.5 con alta confianza (menos del 40% para Over)
        apuesta_complementaria_text = "Menos de 2.5 Goles"
        if predicted_winner_name != "Empate":
            combinacion_interesante_text = f"La combinación **'{predicted_winner_name} y Menos de 2.5 Goles'** podría ser interesante."
    # Segunda prioridad: Ambos Anotan: No (si el marcador lo apoya y Poisson/NN lo sugieren)
    elif not final_btts_bool and poisson_btts_prob < 0.45: # Si el marcador es X-0 o 0-X y Poisson BTTS es bajo
        apuesta_complementaria_text = "Ambos Equipos Anotan: No"
        if predicted_winner_name != "Empate":
            combinacion_interesante_text = f"La combinación **'{predicted_winner_name} y Ambos Equipos Anotan: No'** podría ser interesante."
    # Tercera prioridad: Over 2.5 (si la NN lo apoya fuertemente)
    elif nn_o25_prob > 0.60: # Si la NN predice Over 2.5 con alta confianza
        apuesta_complementaria_text = "Más de 2.5 Goles"
        if predicted_winner_name != "Empate":
            combinacion_interesante_text = f"La combinación **'{predicted_winner_name} y Más de 2.5 Goles'** podría ser interesante."
    
    if apuesta_complementaria_text:
        advice_str += f" Como apuesta complementaria, considera **{apuesta_complementaria_text}**."
    if combinacion_interesante_text:
        advice_str += f" {combinacion_interesante_text}"
        
    advice_str += " Analizamos la forma reciente, el historial cara a cara y el rendimiento ofensivo/defensivo de ambos clubes. Recuerda, las predicciones son probabilidades, no certezas. Juega responsablemente."


    # --- Estructura de retorno final ---
    return {
        "main_prediction": {
            "winner": predicted_winner_name,
            "most_probable_score": most_probable_score,
            "pick_description": f"{predicted_winner_name} gana el partido" if predicted_winner_name != "Empate" else "Empate en el partido",
            "confidence_percent": float(f"{confidence_for_pick * 100:.2f}"),
            "simulated_odd": float(f"{simulated_odd:.2f}"),
            "advice": advice_str,
            "btts_display": final_btts_display,
            "under_over_display": final_o25_display,
            "estimated_goals_home": float(f"{home_score_raw:.2f}"), 
            "estimated_goals_away": float(f"{away_score_raw:.2f}"), 
        },
        "detailed_predictions": {
            "nn_winner_probs": [float(p) for p in nn_winner_probs],
            "nn_o25_prob": float(nn_o25_prob),
            "nn_corners_pred": float(nn_corners_pred),
            "nn_first_half_goals": float(nn_first_half_goals),
            "nn_second_half_goals": float(nn_second_half_goals),
            "svm_winner_probs": [float(p) for p in svm_winner_probs],
            "poisson_winner_probs": [float(p) for p in (poisson_home_win_prob, poisson_draw_prob, poisson_away_win_prob)], 
            "poisson_btts_prob": float(poisson_btts_prob),
            "poisson_o25_prob": float(poisson_o25_prob),
            "combined_winner_probs": [float(p) for p in (combined_home_prob, combined_draw_prob, combined_away_prob)], 
        }
    }

# --- Funciones de la GUI (Tkinter) ---
# ... (El código de la GUI de Tkinter se mantiene igual, pero ahora interpretará
#     la nueva estructura de retorno de predict_for_teams) ...

# Funciones de la GUI (movidas al inicio)
def update_teams(event, home_menu, away_menu, league_var_obj, dfs):
    selected_league = league_var_obj.get()
    teams = set(dfs[selected_league]['home_team_name'].unique()).union(set(dfs[selected_league]['away_team_name'].unique()))
    sorted_teams = sorted(teams)
    home_menu.set('')
    away_menu.set('')
    home_menu['values'] = sorted_teams
    away_menu['values'] = sorted_teams
    autocomplete_entry(home_menu, sorted_teams) 
    autocomplete_entry(away_menu, sorted_teams) 

def autocomplete_entry(entry, values):
    def on_keyrelease(event):
        value = entry.get()
        if value == '':
            entry['values'] = values
        else:
            data = [item for item in values if value.lower() in item.lower()]
            entry['values'] = data
    entry.bind('<KeyRelease>', on_keyrelease)

def predict_selected_teams():
    selected_league = league_var.get()
    home_team = home_team_var.get()
    away_team = away_team_var.get()
    
    if not selected_league or not home_team or not away_team:
        result_text.set("Por favor, selecciona liga y ambos equipos.")
        return

    try:
        # Llama a la función predict_for_teams que ahora devuelve la nueva estructura
        results = predict_for_teams(home_team, away_team, selected_league)
        
        if "error" in results:
            result_text.set(results["error"])
            return

        # Mostrar resultados en la interfaz
        # Sección de Predicción Principal
        main_pred = results['main_prediction']
        output_str = (
            f"--- Predicción Principal ---\n"
            f"Ganador Predicho: {main_pred['winner']}\n"
            f"Marcador Más Probable: {main_pred['most_probable_score']}\n"
            f"Tu Selección: {main_pred['pick_description']}\n"
            f"Confianza: {main_pred['confidence_percent']:.2f}%\n"
            f"Cuota Simulada: x{main_pred['simulated_odd']:.2f}\n"
            f"Ambos Anotan: {main_pred['btts_display']}\n"
            f"Total de Goles: {main_pred['under_over_display']}\n"
            f"Goles Esperados (Local): {main_pred['estimated_goals_home']:.2f}\n"
            f"Goles Esperados (Visitante): {main_pred['estimated_goals_away']:.2f}\n"
            f"Consejo del Experto: {main_pred['advice']}\n\n"
        )
        
        # Sección de Detalles de Modelos
        detailed_pred = results['detailed_predictions']
        output_str += (
            f"--- Análisis Detallado de Modelos ---\n"
            f"NN Ganador: Local {detailed_pred['nn_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['nn_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['nn_winner_probs'][2] * 100:.2f}%\n"
            f"NN Más de 2.5 goles: {detailed_pred['nn_o25_prob'] * 100:.2f}%\n"
            f"NN Córners: {detailed_pred['corners_neural_pred']:.2f}\n"
            f"NN Goles 1T: {detailed_pred['nn_first_half_goals']:.2f} goles\n"
            f"NN Goles 2T: {detailed_pred['nn_second_half_goals']:.2f} goles\n\n"
            f"SVM Ganador: Local {detailed_pred['svm_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['svm_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['svm_winner_probs'][2] * 100:.2f}%\n\n"
            f"Poisson Ganador: Local {detailed_pred['poisson_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['poisson_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['poisson_winner_probs'][2] * 100:.2f}%\n"
            f"Poisson BTTS: {detailed_pred['poisson_btts_prob'] * 100:.2f}%\n"
            f"Poisson O2.5: {detailed_pred['poisson_o25_prob'] * 100:.2f}%\n\n"
            f"Probabilidades Combinadas: Local {detailed_pred['combined_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['combined_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['combined_winner_probs'][2] * 100:.2f}%"
        )
        result_text.set(output_str)

    except Exception as e:
        result_text.set(f"Error al predecir: {e}")

# Lista global para almacenar los partidos futuros cargados
loaded_upcoming_matches = []
def get_upcoming_matches(num_matches_per_league=10, days_ahead=3):
    matches = []
    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)

    for league_id, league_name in LEAGUE_ID_TO_NAME.items():
        try:
            params = {
                "league": league_id,
                "season": today.year,
                "from": today.strftime('%Y-%m-%d'),
                "to": end_date.strftime('%Y-%m-%d')
            }
            data = cached_api_call_python("/fixtures", params)
            fixtures = data.get("response", [])[:num_matches_per_league]

            for fixture in fixtures:
                match = {
                    "fixture_id": fixture['fixture']['id'],
                    "fixture_date": fixture['fixture']['date'],
                    "home_team_name": fixture['teams']['home']['name'],
                    "away_team_name": fixture['teams']['away']['name'],
                    "league_name": league_name
                }
                matches.append(match)
        except Exception as e:
            print(f"Error al obtener partidos para {league_name}: {e}")
            continue

    return matches


def get_and_predict_upcoming_matches_gui(): 
    result_text.set("Obteniendo partidos futuros y generando predicciones...")
    
    for i in upcoming_matches_tree.get_children():
        upcoming_matches_tree.delete(i)

    upcoming_matches = get_upcoming_matches(num_matches_per_league=10) 
    global loaded_upcoming_matches 
    loaded_upcoming_matches = upcoming_matches
    
    if not loaded_upcoming_matches:
        result_text.set("No se encontraron partidos futuros en las ligas configuradas con datos disponibles para predicción.")
        return

    for i, match_info in enumerate(loaded_upcoming_matches):
        home_name_display = match_info['home_team_name']
        away_name_display = match_info['away_team_name']
        
        upcoming_matches_tree.insert("", "end", iid=i, values=(
            home_name_display,
            away_name_display,
            match_info['league_name'],
            datetime.strptime(match_info['fixture_date'], '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%d %H:%M')
        ))
    result_text.set(f"Cargados {len(loaded_upcoming_matches)} partidos futuros. Selecciona uno para ver la predicción.")

def predict_from_treeview(event):
    selected_item_id = upcoming_matches_tree.focus()
    if not selected_item_id:
        return
    
    item_index = int(selected_item_id)
    match_info = loaded_upcoming_matches[item_index] 

    try:
        # Llama a la función predict_for_teams que ahora devuelve la nueva estructura
        results = predict_for_teams(
            match_info['home_team_name'], 
            match_info['away_team_name'], 
            match_info['league_name']
        )
        
        if "error" in results:
            result_text.set(f"Error al predecir para {match_info['home_team_name']} vs {match_info['away_team_name']} ({match_info['league_name']}): {results['error']}")
            return

        # Mostrar resultados en la interfaz
        # Sección de Predicción Principal
        main_pred = results['main_prediction']
        output_str = (
            f"--- Predicción Principal ---\n"
            f"Ganador Predicho: {main_pred['winner']}\n"
            f"Marcador Más Probable: {main_pred['most_probable_score']}\n"
            f"Tu Selección: {main_pred['pick_description']}\n"
            f"Confianza: {main_pred['confidence_percent']:.2f}%\n"
            f"Cuota Simulada: x{main_pred['simulated_odd']:.2f}\n"
            f"Ambos Anotan: {main_pred['btts_display']}\n"
            f"Total de Goles: {main_pred['under_over_display']}\n"
            f"Goles Esperados (Local): {main_pred['estimated_goals_home']:.2f}\n"
            f"Goles Esperados (Visitante): {main_pred['estimated_goals_away']:.2f}\n"
            f"Consejo del Experto: {main_pred['advice']}\n\n"
        )
        
        # Sección de Detalles de Modelos
        detailed_pred = results['detailed_predictions']
        output_str += (
            f"--- Análisis Detallado de Modelos ---\n"
            f"NN Ganador: Local {detailed_pred['nn_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['nn_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['nn_winner_probs'][2] * 100:.2f}%\n"
            f"NN Más de 2.5 goles: {detailed_pred['nn_o25_prob'] * 100:.2f}%\n"
            f"NN Córners: {detailed_pred['nn_corners_pred']:.2f}\n"
            f"NN Goles 1T: {detailed_pred['nn_first_half_goals']:.2f} goles\n"
            f"NN Goles 2T: {detailed_pred['nn_second_half_goals']:.2f} goles\n\n"
            f"SVM Ganador: Local {detailed_pred['svm_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['svm_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['svm_winner_probs'][2] * 100:.2f}%\n\n"
            f"Poisson Ganador: Local {detailed_pred['poisson_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['poisson_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['poisson_winner_probs'][2] * 100:.2f}%\n"
            f"Poisson BTTS: {detailed_pred['poisson_btts_prob'] * 100:.2f}%\n"
            f"Poisson O2.5: {detailed_pred['poisson_o25_prob'] * 100:.2f}%\n\n"
            f"Probabilidades Combinadas: Local {detailed_pred['combined_winner_probs'][0] * 100:.2f}%, Empate {detailed_pred['combined_winner_probs'][1] * 100:.2f}%, Visitante {detailed_pred['combined_winner_probs'][2] * 100:.2f}%"
        )
        result_text.set(output_str)

    except Exception as e:
        result_text.set(f"Error al predecir para el partido seleccionado: {e}")


