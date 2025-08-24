import json
from io import BytesIO
from datetime import datetime, date, timedelta
from textwrap import fill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==========================
# Configuração geral
# ==========================
st.set_page_config(page_title="Controle de Peso", layout="wide", page_icon="📊")
DATA_FILE = "weights.csv"
SETTINGS_FILE = "settings.json"

# ==========================
# Utilidades: persistência
# ==========================
def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "height_cm": 183.0,           # altura padrão (edite na interface)
            "goal_weight": 80.0,          # meta padrão
            "show_moving_avg_days": 7,    # média móvel para suavizar
        }

def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def load_data():
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=["date"])
        df = df.dropna(subset=["date", "weight"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return df.reset_index(drop=True)
    except FileNotFoundError:
        return pd.DataFrame(columns=["date", "weight"])

def save_data(df):
    df.sort_values("date", inplace=True)
    df.to_csv(DATA_FILE, index=False)

# ==========================
# Cálculos
# ==========================
def bmi(weight_kg, height_cm):
    h = (height_cm or 0) / 100.0
    if h <= 0:
        return None
    return weight_kg / (h * h)

def bmi_category(b):
    if b is None: return "-"
    if b < 18.5: return "Abaixo do peso"
    if b < 25: return "Normal"
    if b < 30: return "Sobrepeso"
    return "Obesidade"

def weekly_series(df):
    # usa o último valor de cada semana (domingo)
    s = df.set_index(pd.to_datetime(df["date"]))["weight"].resample("W-SUN").last().dropna()
    return s

def monthly_series(df):
    # último valor de cada mês
    s = df.set_index(pd.to_datetime(df["date"]))["weight"].resample("M").last().dropna()
    return s

def rate_and_projection(df, goal_weight):
    """Retorna taxa média diária (kg/dia) via regressão linear e data estimada para meta."""
    if len(df) < 2:
        return None, None, None
    x = pd.to_datetime(df["date"]).map(pd.Timestamp.toordinal).to_numpy()
    y = df["weight"].to_numpy()
    # regressão linear simples (y = a*x + b)
    a, b = np.polyfit(x, y, 1)
    rate_per_day = a  # kg por dia (negativo se está caindo)
    # projetar quando y = goal_weight -> goal = a*x + b => x = (goal - b)/a
    eta_date = None
    if rate_per_day != 0:
        x_goal = (goal_weight - b) / rate_per_day
        # só faz sentido se for no futuro e se o sinal levar até a meta
        today_ord = date.today().toordinal()
        if (rate_per_day < 0 and goal_weight < y[-1] and x_goal >= today_ord) or \
           (rate_per_day > 0 and goal_weight > y[-1] and x_goal >= today_ord):
            eta_date = date.fromordinal(int(round(x_goal)))
    # taxa semanal aproximada
    rate_per_week = rate_per_day * 7 if rate_per_day is not None else None
    return rate_per_day, rate_per_week, eta_date

def moving_average(series, window):
    if window is None or window <= 1:
        return None
    s = pd.Series(series)
    ma = s.rolling(window=window, min_periods=1).mean()
    return ma.to_list()

# ==========================
# Motivação
# ==========================
MOTIVATIONAL_QUOTES = [
    "Pequenos passos, grandes resultados.",
    "Consistência vence intensidade.",
    "Você não precisa ser perfeito, só persistente.",
    "A disciplina te leva onde a motivação não alcança.",
    "Hoje melhor que ontem, pior que amanhã.",
    "Seu futuro ‘você’ vai agradecer o que você faz hoje.",
    "Progresso > Perfeição.",
]

def pick_quote(df):
    if df.empty:
        return "Todo começo é um passo na direção certa. Bora!"
    if len(df) >= 2 and df["weight"].iloc[-1] < df["weight"].iloc[-2]:
        return "Mandou bem! O ponteiro desceu — celebre e siga firme! 👏"
    return np.random.choice(MOTIVATIONAL_QUOTES)

# ==========================
# Sidebar: configurações
# ==========================
settings = load_settings()
with st.sidebar:
    st.header("⚙️ Configurações")
    settings["height_cm"] = st.number_input("Altura (cm)", min_value=100.0, max_value=250.0, step=0.5, value=float(settings["height_cm"]))
    settings["goal_weight"] = st.number_input("Meta de peso (kg)", min_value=30.0, max_value=300.0, step=0.1, value=float(settings["goal_weight"]))
    settings["show_moving_avg_days"] = st.slider("Média móvel (dias)", 1, 30, int(settings["show_moving_avg_days"]))
    if st.button("💾 Salvar configurações"):
        save_settings(settings)
        st.success("Configurações salvas!")

# ==========================
# Cabeçalho
# ==========================
st.title("📊 Controle de Peso")
st.caption("Registre seu peso, acompanhe métricas, compare períodos e projete a chegada à sua meta.")

# ==========================
# Entrada de dados
# ==========================
df = load_data()
with st.expander("➕ Adicionar/editar peso", expanded=True):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, step=0.1,
                                 value=(df["weight"].iloc[-1] if not df.empty else 80.0))
    with col2:
        dt = st.date_input("Data", value=date.today())
    with col3:
        if st.button("Salvar registro"):
            new_row = pd.DataFrame({"date": [dt], "weight": [weight]})
            df = pd.concat([df, new_row], ignore_index=True)
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            save_data(df)
            st.success("Peso registrado com sucesso! ✅")

# ==========================
# Lembrete semanal (domingo)
# ==========================
today = date.today()
is_sunday = today.weekday() == 6  # 0=segunda ... 6=domingo
last_entry_days = (today - df["date"].iloc[-1]).days if not df.empty else None

reminder_msg = None
if is_sunday:
    reminder_msg = "Hoje é domingo! Não esqueça de registrar seu peso. 🗓️"
elif last_entry_days is not None and last_entry_days >= 7:
    reminder_msg = f"Você não registra há {last_entry_days} dias. Que tal atualizar agora? ⏰"

if reminder_msg:
    st.info(reminder_msg)

# Oferece download de um lembrete semanal .ics (domingo às 09:00)
with st.expander("🔔 Lembrete semanal (calendário)"):
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//WeightTracker//PT-BR
BEGIN:VEVENT
UID:wt-reminder-{today.isoformat()}
DTSTAMP:{today.strftime("%Y%m%dT090000")}
RRULE:FREQ=WEEKLY;BYDAY=SU
DTSTART:{today.strftime("%Y%m%dT090000")}
DTEND:{today.strftime("%Y%m%dT091500")}
SUMMARY:Registrar peso semanal
DESCRIPTION:Acesse seu app de Controle de Peso e registre o valor de hoje.
END:VEVENT
END:VCALENDAR
"""
    st.download_button(
        "📥 Baixar lembrete .ics",
        data=ics.encode("utf-8"),
        file_name="lembrete_semanal_peso.ics",
        mime="text/calendar"
    )

# ==========================
# Painel principal
# ==========================
if df.empty:
    st.warning("Nenhum registro ainda. Adicione seu primeiro peso acima.")
    st.stop()

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Peso inicial", f"{df['weight'].iloc[0]:.1f} kg")
with colB:
    st.metric("Peso atual", f"{df['weight'].iloc[-1]:.1f} kg")
with colC:
    delta_total = df["weight"].iloc[-1] - df["weight"].iloc[0]
    st.metric("Variação total", f"{delta_total:+.1f} kg")
with colD:
    current_bmi = bmi(df["weight"].iloc[-1], settings["height_cm"])
    st.metric("IMC atual", f"{current_bmi:.1f}" if current_bmi else "-")

st.caption(f"Classificação IMC: **{bmi_category(current_bmi)}**")

# ==========================
# Gráfico principal
# ==========================
st.subheader("📈 Evolução do Peso")
fig, ax = plt.subplots(figsize=(10, 4))
x_dates = pd.to_datetime(df["date"])
ax.plot(x_dates, df["weight"], marker="o", linestyle="-", label="Peso")

# média móvel
ma_days = settings.get("show_moving_avg_days", 7)
ma = moving_average(df["weight"].tolist(), ma_days)
if ma_days > 1 and ma is not None:
    ax.plot(x_dates, ma, linestyle="--", label=f"Média móvel ({ma_days}d)")

# linha de meta
goal = float(settings["goal_weight"])
ax.axhline(y=goal, linestyle=":", label=f"Meta ({goal:.1f} kg)")

ax.set_xlabel("Data")
ax.set_ylabel("Peso (kg)")
ax.set_title("Evolução do Peso")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ==========================
# Comparações semanal e mensal
# ==========================
st.subheader("📊 Comparações")

w = weekly_series(df)
m = monthly_series(df)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Semanal (últimos 8 registros)**")
    if len(w) >= 2:
        w_delta = w.diff()
        st.line_chart(w.tail(8))
        st.dataframe(
            pd.DataFrame({"Peso (kg)": w.round(1), "Δ Semana (kg)": w_delta.round(1)}).tail(8)
        )
    else:
        st.info("Registre ao menos 2 semanas para comparação.")

with col2:
    st.markdown("**Mensal (últimos 12 registros)**")
    if len(m) >= 2:
        m_delta = m.diff()
        st.line_chart(m.tail(12))
        st.dataframe(
            pd.DataFrame({"Peso (kg)": m.round(1), "Δ Mês (kg)": m_delta.round(1)}).tail(12)
        )
    else:
        st.info("Registre ao menos 2 meses para comparação.")

# ==========================
# Projeção até a meta
# ==========================
st.subheader("🧭 Projeção até a Meta")
rate_day, rate_week, eta = rate_and_projection(df, goal)
if rate_day is None:
    st.info("Ainda não há dados suficientes para projetar.")
else:
    trend_txt = f"Taxa média: {rate_week:+.2f} kg/semana ({rate_day:+.3f} kg/dia)."
    if eta:
        st.success(f"{trend_txt} Projeção de atingir **{goal:.1f} kg** em **{eta.strftime('%d/%m/%Y')}** (se mantiver o ritmo).")
    else:
        st.warning(f"{trend_txt} Com a tendência atual, não é possível estimar a data para a meta.")

st.info(pick_quote(df))

# ==========================
# Relatório em PDF (com gráfico + métricas)
# ==========================
st.subheader("📝 Relatório em PDF")

def build_pdf(df, settings, fig) -> bytes:
    # Gera uma imagem do gráfico em buffer
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=180)
    img_buf.seek(0)

    # Monta PDF simples com reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    W, H = A4

    margin = 36
    y = H - margin

    def draw_text(text, size=11, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        wrapped = fill(text, 90)
        for line in wrapped.split("\n"):
            y -= leading
            c.drawString(margin, y, line)
        y -= 6

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Relatório - Controle de Peso")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 18

    # Métricas
    start_w = df['weight'].iloc[0]
    curr_w = df['weight'].iloc[-1]
    delta = curr_w - start_w
    imc_val = bmi(curr_w, settings["height_cm"])
    draw_text(f"Peso inicial: {start_w:.1f} kg")
    draw_text(f"Peso atual: {curr_w:.1f} kg")
    draw_text(f"Variação total: {delta:+.1f} kg")
    draw_text(f"Altura: {settings['height_cm']:.1f} cm | IMC atual: {imc_val:.1f} ({bmi_category(imc_val)})" if imc_val else "IMC: -")
    if rate_week is not None:
        proj_line = f"Taxa média: {rate_week:+.2f} kg/semana."
        if eta:
            proj_line += f" Projeção de atingir {settings['goal_weight']:.1f} kg em {eta.strftime('%d/%m/%Y')}."
        draw_text(proj_line)

    # Espaço
    y -= 12

    # Gráfico
    img = ImageReader(img_buf)
    img_w, img_h = img.getSize()
    # redimensiona para caber na página
    max_w = W - 2 * margin
    scale = min(max_w / img_w, 350 / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
    y -= (draw_h + 12)

    # Tabela resumida (últimos 10)
    tail = df.tail(10).copy()
    table_text = "Últimos registros (data - peso): " + ", ".join(
        [f"{pd.to_datetime(d).strftime('%d/%m/%Y')} - {w:.1f} kg" for d, w in zip(tail['date'], tail['weight'])]
    )
    draw_text(table_text)

    c.showPage()
    c.save()
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes

if st.button("📄 Gerar PDF"):
    pdf_bytes = build_pdf(df, settings, fig)
    st.download_button(
        "📥 Baixar Relatório",
        data=pdf_bytes,
        file_name=f"relatorio_peso_{date.today().isoformat()}.pdf",
        mime="application/pdf"
    )
