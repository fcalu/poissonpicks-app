import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz

from PoissonPicksIA import predict_for_teams, get_upcoming_matches

# Zona horaria Guadalajara
TZ_LOCAL = pytz.timezone("America/Mexico_City")

st.set_page_config(page_title="PoissonPicks", layout="wide")

st.markdown("""
    <style>
        .main-header {font-size:2.6em;font-weight:bold;margin-top:30px;}
        .match-card {background:white;border-radius:20px;box-shadow:0 6px 24px #2222; margin:16px 0; padding:22px 24px;}
        .team-logo {height:48px;vertical-align:middle;margin-right:12px;}
        .league-label {font-weight:bold;color:#1a64be;font-size:1.1em;}
        .search-bar input {font-size:1.1em;padding:10px 18px;}
        .stButton button {font-size:1.08em; padding:10px 30px; border-radius:1.5em;}
        .stMarkdown {margin-bottom: 1em;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("PoissonPicks")
st.sidebar.write("Predicciones avanzadas · By Felix")
st.sidebar.markdown("---")
selected_date = st.sidebar.date_input("Filtra por fecha", value=datetime.now(TZ_LOCAL).date())

# --- BUSCADOR EN LA CABECERA ---
search_query = st.text_input("🔎 Busca partido, liga o equipo...", key="search")

st.markdown('<div class="main-header">Próximos Partidos (Zona Guadalajara)</div>', unsafe_allow_html=True)

# --- CARGAR PARTIDOS DE LA API ---
if "partidos" not in st.session_state:
    with st.spinner("Consultando API-Football..."):
        partidos = get_upcoming_matches(num_matches_per_league=30, days_ahead=4)
        st.session_state["partidos"] = partidos
else:
    partidos = st.session_state["partidos"]

# --- FILTRO DE FECHA Y BUSQUEDA ---
matches_list = []
for m in partidos:
    fecha_local = datetime.strptime(m['fixture_date'], "%Y-%m-%dT%H:%M:%S%z")
    if fecha_local.date() == selected_date:
        matches_list.append(m)
if search_query.strip():
    matches_list = [m for m in matches_list if
        search_query.lower() in m["home_team_name"].lower()
        or search_query.lower() in m["away_team_name"].lower()
        or search_query.lower() in m["league_name"].lower()
    ]

if not matches_list:
    st.info("No hay partidos para la fecha/criterio seleccionados.")
else:
    for idx, m in enumerate(matches_list):
        fecha_local = datetime.strptime(m['fixture_date'], "%Y-%m-%dT%H:%M:%S%z")
        st.markdown(f"""
        <div class="match-card">
            <div class="league-label">{m['league_name']} <span style="color:#999;">{fecha_local.strftime('%A, %d %b')} ({fecha_local.strftime('%H:%M')} hrs)</span></div>
            <div style="font-size:1.25em;display:flex;align-items:center;margin:12px 0;">
                <img src="{m.get('home_team_logo','')}" class="team-logo" alt="logo"> <b>{m['home_team_name']}</b>
                <span style="margin:0 10px;color:#222;font-size:1.1em;">vs</span>
                <img src="{m.get('away_team_logo','')}" class="team-logo" alt="logo"> <b>{m['away_team_name']}</b>
            </div>
            <form action="" method="post">
                <input type="hidden" name="match_id" value="{idx}">
                <div>
                    <button id="predict-btn-{idx}" class="stButton">Predecir</button>
                </div>
            </form>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Predecir {m['home_team_name']} vs {m['away_team_name']}", key=f"btn_{idx}"):
            with st.spinner("Calculando predicción..."):
                result = predict_for_teams(
                    m["home_team_name"], m["away_team_name"], m["league_name"]
                )
                if "error" in result:
                    st.error(result["error"])
                else:
                    pred = result["main_prediction"]
                    st.success(f"""
**Ganador Predicho:** {pred['winner']}  
**Marcador Más Probable:** {pred['most_probable_score']}  
**Selección:** {pred['pick_description']}  
**Confianza:** {pred['confidence_percent']:.2f}%  
**Cuota Simulada:** x{pred['simulated_odd']:.2f}  
**Ambos Anotan:** {pred['btts_display']}  
**Total de Goles:** {pred['under_over_display']}  
**Goles Esperados (Local):** {pred['estimated_goals_home']:.2f}  
**Goles Esperados (Visitante):** {pred['estimated_goals_away']:.2f}  
**Consejo del Experto:** {pred['advice']}  
                    """)

st.markdown('<br><center>Desarrollado por Felix Matus &middot; Powered by Poisson & IA &middot; 2025</center>', unsafe_allow_html=True)
