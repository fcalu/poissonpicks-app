import streamlit as st
from datetime import datetime
from PoissonPicksIA import predict_for_teams, get_upcoming_matches

st.set_page_config(page_title="PoissonPicks", layout="wide")
st.title("⚽ PoissonPicks - Predicción de Partidos de Fútbol")

st.header("🌐 Partidos Futuros desde la API (Próximos 4 días)")

# --- Manejo de estado para lista de partidos ---
if "partidos_sorted" not in st.session_state:
    st.session_state["partidos_sorted"] = []

# Botón para cargar partidos futuros
if st.button("Cargar partidos futuros"):
    with st.spinner("Consultando API-Football..."):
        partidos = get_upcoming_matches(num_matches_per_league=20, days_ahead=4)
        if not partidos:
            st.warning("No se encontraron partidos para los próximos 4 días.")
            st.session_state["partidos_sorted"] = []
        else:
            partidos_sorted = sorted(partidos, key=lambda x: x['fixture_date'])
            st.session_state["partidos_sorted"] = partidos_sorted

# Mostrar tabla de partidos si hay partidos cargados
if st.session_state["partidos_sorted"]:
    partidos_df = []
    for m in st.session_state["partidos_sorted"]:
        fecha = datetime.strptime(m['fixture_date'], "%Y-%m-%dT%H:%M:%S%z")
        partidos_df.append({
            "Fecha": fecha.strftime("%d-%m-%Y %H:%M"),
            "Liga": m['league_name'],
            "Local": m['home_team_name'],
            "Visitante": m['away_team_name'],
        })
    st.dataframe(partidos_df, use_container_width=True)

    idx = st.selectbox(
        "Selecciona partido para predecir",
        range(len(partidos_df)),
        format_func=lambda i: f"{partidos_df[i]['Fecha']} | {partidos_df[i]['Liga']} | {partidos_df[i]['Local']} vs {partidos_df[i]['Visitante']}",
        key="pred_index"
    )
    partido_sel = st.session_state["partidos_sorted"][idx]

    if st.button("Predecir este partido"):
        with st.spinner("Calculando predicción para partido seleccionado..."):
            result = predict_for_teams(
                partido_sel["home_team_name"],
                partido_sel["away_team_name"],
                partido_sel["league_name"]
            )
            if "error" in result:
                st.error(result["error"])
            else:
                pred = result["main_prediction"]
                st.markdown(f"""
                ### Predicción Principal
                - **Ganador Predicho:** {pred['winner']}
                - **Marcador Más Probable:** {pred['most_probable_score']}
                - **Selección:** {pred['pick_description']}
                - **Confianza:** {pred['confidence_percent']:.2f}%
                - **Cuota Simulada:** x{pred['simulated_odd']:.2f}
                - **Ambos Anotan:** {pred['btts_display']}
                - **Total de Goles:** {pred['under_over_display']}
                - **Goles Esperados (Local):** {pred['estimated_goals_home']:.2f}
                - **Goles Esperados (Visitante):** {pred['estimated_goals_away']:.2f}
                - **Consejo del Experto:** {pred['advice']}
                """)
                # Si quieres mostrar el análisis detallado, descomenta aquí:
                detailed = result["detailed_predictions"]
                st.json(detailed)
else:
    st.info("Carga los partidos futuros para comenzar.")

