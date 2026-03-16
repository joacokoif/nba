"""
nba_tv_hoy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detecta partidos NBA del día televisados en los canales
principales (ESPN / Prime Video / Max (HBO) / NBA TV),
convierte los horarios a Argentina (UTC-3) y genera el
mensaje listo para copiar y pegar en WhatsApp / Telegram.

Fuente: API pública del CDN de nba.com (sin clave API).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import io
import json
import requests
import urllib3
from datetime import datetime, timezone, timedelta

# ── Forzar UTF-8 en Windows ───────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
urllib3.disable_warnings()

# ── Zona horaria Argentina ─────────────────────────────────
TZ_ARGENTINA = timezone(timedelta(hours=-3))  # ART = UTC-3

# ── Canales que nos interesan (busca en minúsculas) ───────
# Clave = substring a buscar en el nombre del canal
# Valor = etiqueta bonita para el mensaje
CANALES_OBJETIVO = {
    "espn":         "ESPN",
    "abc":          "ESPN / ABC",
    "prime video":  "Prime Video",
    "prime":        "Prime Video",
    "amazon":       "Prime Video",
    "max":          "Max (HBO)",
    "hbo":          "Max (HBO)",
    "nba tv":       "NBA TV",
    "nbatv":        "NBA TV",
    "peacock":      "Peacock",
    "nbc":          "NBC",
}

# ── Headers para las requests ──────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Accept":  "application/json",
}

# ── URL del schedule completo CDN ─────────────────────────
URL_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"


def get_today_date_str() -> str:
    """Devuelve la fecha de hoy en formato YYYY-MM-DD (hora Argentina)."""
    now_arg = datetime.now(TZ_ARGENTINA)
    return now_arg.strftime("%Y-%m-%d")


def fetch_schedule() -> list:
    """
    Descarga el schedule completo del CDN de la NBA y devuelve
    la lista de partidos de hoy. Incluye info de broadcasters.
    """
    try:
        r = requests.get(URL_SCHEDULE, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"⚠️  Error al descargar schedule: {e}")
        return []

    game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
    today_str  = get_today_date_str()  # YYYY-MM-DD

    for gd in game_dates:
        # El campo puede venir como "03/06/2026 00:00:00" o "2026-03-06"
        raw_date = gd.get("gameDate", "")
        # Intentar parsear ambos formatos
        parsed_date = ""
        for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                parsed_date = datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        if parsed_date == today_str:
            return gd.get("games", [])

    # Si no encontramos la fecha exacta, intentar buscar en UTC
    # (los partidos pasadas las 21:00 ART pueden ser del día siguiente en UTC)
    for gd in game_dates:
        raw_date = gd.get("gameDate", "")
        if today_str in raw_date or today_str.replace("-", "/") in raw_date:
            return gd.get("games", [])

    return []


def canal_limpio(nombre_raw: str) -> str | None:
    """
    Recibe el nombre de un broadcaster de la API y devuelve la
    etiqueta limpia si es uno de los canales de interés, o None.
    """
    nombre_lower = nombre_raw.lower().strip()
    for clave, etiqueta in CANALES_OBJETIVO.items():
        if clave in nombre_lower:
            return etiqueta
    return None


def hora_argentina(game_time_utc: str) -> str:
    """
    Convierte un timestamp UTC (ISO 8601, termina en 'Z') a
    la hora local de Argentina (UTC-3, formato HH:MM).
    """
    try:
        # Formato: "2026-03-07T00:00:00Z"
        dt_utc = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        dt_arg = dt_utc.astimezone(TZ_ARGENTINA)
        return dt_arg.strftime("%H:%M")
    except Exception:
        return "??"


def extraer_apodo(team: dict) -> str:
    """
    De un dict de equipo extrae solo el apodo (ej: 'Celtics')
    para que el mensaje sea más corto y legible.
    """
    # teamName ya suele ser solo el apodo ('Celtics', 'Lakers')
    return team.get("teamName", team.get("teamTricode", "??"))


def procesar_partidos(games: list) -> list:
    """
    Procesa la lista de partidos del día y filtra los que tienen
    al menos un broadcaster de interés en TV nacional (o OTT como Prime).
    Devuelve lista ordenada por hora ART.
    """
    resultado = []

    for g in games:
        # ── Hora ──────────────────────────────────────────
        hora_utc = g.get("gameDateTimeUTC", "")
        hora_arg = hora_argentina(hora_utc)

        # ── Equipos ───────────────────────────────────────
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        visitante = extraer_apodo(away)
        local     = extraer_apodo(home)

        # ── Broadcasters ──────────────────────────────────
        bcast = g.get("broadcasters", {})

        # Tomamos: nacionales de TV + OTT nacionales (Prime es OTT)
        fuentes = (
            bcast.get("nationalTvBroadcasters",  []) +
            bcast.get("nationalOttBroadcasters", []) +
            bcast.get("intlTvBroadcasters",      []) +
            bcast.get("intlOttBroadcasters",     [])
        )

        canales_ok = []
        for b in fuentes:
            nombre = b.get("broadcasterDisplay", "")
            c = canal_limpio(nombre)
            if c and c not in canales_ok:
                canales_ok.append(c)

        if canales_ok:
            resultado.append({
                "hora_sort": hora_utc,      # Para ordenar
                "hora_arg":  hora_arg,
                "visitante": visitante,
                "local":     local,
                "canales":   canales_ok,
            })

    # Ordenar por hora UTC
    resultado.sort(key=lambda x: x["hora_sort"])
    return resultado


def generar_mensaje(partidos: list, fecha_str: str) -> str:
    """Genera el mensaje final listo para copiar."""
    dt_fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    fecha_display = dt_fecha.strftime("%d/%m/%Y")

    if not partidos:
        return (
            f"🏀 NBA – Partidos televisados hoy ({fecha_display})\n\n"
            "😔 No hay partidos en ESPN, Prime Video, Max o NBA TV hoy."
        )

    lineas = [f"🏀 NBA – Partidos televisados hoy ({fecha_display})\n"]

    for p in partidos:
        canales_str = " | ".join(p["canales"])
        bloque = (
            f"{p['hora_arg']}\n"
            f"{p['visitante']} vs {p['local']}\n"
            f"📺 {canales_str}"
        )
        lineas.append(bloque)

    return "\n\n".join(lineas)


def main():
    fecha_hoy = get_today_date_str()
    print(f"📅 Buscando partidos NBA para: {fecha_hoy} (hora Argentina)\n")
    print("⏳ Descargando schedule desde nba.com...")

    games = fetch_schedule()

    if not games:
        print("⚠️  No se encontraron partidos para hoy en el schedule.")
        print("    (Puede ser día de descanso o error de red)\n")
        msg = generar_mensaje([], fecha_hoy)
    else:
        print(f"✅  {len(games)} partido(s) en total hoy")
        partidos_tv = procesar_partidos(games)
        print(f"📺  {len(partidos_tv)} partido(s) televisado(s) en canales de interés\n")
        msg = generar_mensaje(partidos_tv, fecha_hoy)

    # ── Mostrar resultado ─────────────────────────────────
    print("═" * 52)
    print(msg)
    print("═" * 52)

    # ── Guardar en archivo ────────────────────────────────
    import os
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "partidos_hoy.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(msg)
    print(f"\n💾 Guardado en: {output_file}")


if __name__ == "__main__":
    main()
