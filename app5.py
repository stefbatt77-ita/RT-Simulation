import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import random
import pandas as pd
import sqlite3
from datetime import datetime
from fpdf import FPDF
import io

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Aero-NDT Pro v14.5", layout="wide", page_icon="☢️")

# --- STILE CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 4px; height: 3em; }
    .stMetric { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    h1, h2, h3 { color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 1. GESTIONE DATI
# ==============================================================================
def init_db():
    conn = sqlite3.connect('ndt_academy.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, student TEXT, date TEXT, score INTEGER, details TEXT)''')
    try:
        c.execute("INSERT INTO users VALUES ('admin', 'admin123', 'istruttore')")
        c.execute("INSERT INTO users VALUES ('student', 'ndt2026', 'studente')")
        conn.commit()
    except sqlite3.IntegrityError: pass
    return conn

def login_user(username, password):
    conn = init_db(); c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

def save_result(student, score, details):
    conn = init_db(); c = conn.cursor()
    c.execute("INSERT INTO results (student, date, score, details) VALUES (?, ?, ?, ?)", 
              (student, datetime.now().strftime("%Y-%m-%d %H:%M"), score, str(details)))
    conn.commit()

# ==============================================================================
# 2. MOTORE FISICO E REALISMO
# ==============================================================================
def apply_detector_defects(image, is_calibrated):
    if is_calibrated: return image
    rows, cols = image.shape
    noisy_img = image.astype(float)
    gradient = np.linspace(0.85, 1.0, cols)
    gain_map = np.random.normal(1.0, 0.04, (rows, cols)) * gradient
    noisy_img *= gain_map
    for _ in range(12):
        cy, cx = random.randint(0, rows-1), random.randint(0, cols-1)
        y, x = np.ogrid[:rows, :cols]
        mask = (y - cy)**2 + (x - cx)**2 <= random.randint(2, 5)**2
        noisy_img[mask] *= 0.1
    return np.clip(noisy_img, 0, 65535).astype(np.uint16)

def add_realistic_physics(base_signal, thickness_map, size):
    scatter_map = gaussian_filter(thickness_map, sigma=40)
    scatter_signal = base_signal * (scatter_map / np.max(scatter_map)) * 0.35
    textured = (base_signal + scatter_signal) * np.random.normal(1.0, 0.008, (size, size))
    noise = np.random.normal(0, np.sqrt(np.maximum(textured, 10)) * 1.9, (size, size))
    return gaussian_filter(textured + noise, sigma=0.8)

def generate_scan_core(kv, ma, time, material, thickness, selected_defect, iqi_mode):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.025, "Ti-6Al-4V": 0.05, "Inconel 718": 0.1, "Steel 17-4 PH": 0.08}
    mu = mu_map[material] * (130/max(20, kv))**1.6 
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    if iqi_mode == "ISO 19232-1 (Fili)":
        for i in range(7): m_sp[40:140, 80+i*40:82+i*40] += (0.4 - i*0.05)
    else:
        plaque_t = max(0.1, thickness * 0.02)
        m_sp[50:110, 60:180] += plaque_t
        for i, d in enumerate([1, 2, 4]):
            r = (plaque_t * d / 0.05) / 2 * 3
            y, x = np.ogrid[:size, :size]
            m_sp[(x-(80+i*40))**2 + (y-80)**2 <= r**2] -= plaque_t

    for i in range(13): m_sp[500:540, 150+i*25:152+i*25] += (0.7 / (i+1))
    
    bboxes, def_list = [], []
    to_gen = [selected_defect] if selected_defect not in ["Nessun Difetto", "Casuale (Multiplo)"] else []
    if selected_defect == "Casuale (Multiplo)": to_gen = random.sample(["Cricca", "Porosità", "Tungsteno"], random.randint(1,2))
    
    for d_type in to_gen:
        rx, ry = random.randint(150, 450), random.randint(150, 450)
        if d_type == "Cricca":
            m_sp[ry:ry+80, rx:rx+2] -= 0.9
            bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 100, "t": d_type})
        else:
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 6**2] -= 2.0
            bboxes.append({"x": rx-15, "y": ry-15, "w": 30, "h": 30, "t": d_type})
        def_list.append(d_type)

    dose = (kv**2) * ma * time * 0.22
    signal = add_realistic_physics(dose * np.exp(-mu * m_sp), m_sp, size)
    return np.clip(signal, 0, 65535).astype(np.uint16), def_list, bboxes

# ==============================================================================
# 3. LOGICA DI CALCOLO
# ==============================================================================
def calculate_exposure_logic(material, thickness):
    data = {
        "Al-2024 (Avional)": {"base_kv": 45, "slope": 3.2, "mu_ref": 0.12},
        "Ti-6Al-4V":         {"base_kv": 60, "slope": 4.8, "mu_ref": 0.18},
        "Inconel 718":       {"base_kv": 90, "slope": 7.8, "mu_ref": 0.28},
        "Steel 17-4 PH":     {"base_kv": 85, "slope": 6.5, "mu_ref": 0.25}
    }
    m = data[material]
    return int(m["base_kv"] + (thickness * m["slope"])), round(12 * np.exp(thickness * m["mu_ref"] * 0.5), 1)

# ==============================================================================
# 4. INTERFACCIA UTENTE
# ==============================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'is_calibrated' not in st.session_state: st.session_state.is_calibrated = False

if not st.session_state.logged_in:
    st.title("🔒 Login Aero-NDT Academy")
    user = st.text_input("Username"); pwd = st.text_input("Password", type="password")
    if st.button("Accedi"):
        role = login_user(user, pwd)
        if role: 
            st.session_state.logged_in = True; st.session_state.username = user; st.session_state.role = role[0]
            st.rerun()
    st.stop()

with st.sidebar:
    st.write(f"👤 {st.session_state.username} | 🎖️ {st.session_state.role}")
    if st.button("Logout"): st.session_state.logged_in = False; st.rerun()
    st.divider()
    if st.button("🛠️ CALIBRA DETECTOR", type="primary"): 
        st.session_state.is_calibrated = True; st.rerun()
    mode = st.radio("Seleziona Area", ["STUDIO", "ESAME", "DATABASE"])

if mode == "STUDIO":
    st.title("📘 Laboratorio Radiografico")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.slider("Spessore (mm)", 5, 40, 15)
        def_sel = st.selectbox("Difetto", ["Cricca", "Porosità", "Tungsteno", "Nessun Difetto"])
        st.divider()
        kv = st.slider("kV", 40, 250, 90); ma = st.slider("mA", 1.0, 10.0, 4.0); sec = st.slider("Secondi", 1, 60, 15)
        
        if st.button("▶️ ACQUISICI IMMAGINE"):
            img, defs, bboxes = generate_scan_core(kv, ma, sec, mat, thick, def_sel, "ISO 19232-1 (Fili)")
            st.session_state.s_img = apply_detector_defects(img, st.session_state.is_calibrated)
            st.session_state.s_defs = defs; st.session_state.s_bboxes = bboxes

    with c2:
        tab1, tab2, tab3 = st.tabs(["🖼️ Display DDA", "🧮 Exposure", "📐 Qualità SNR"])
        
        with tab1:
            if 's_img' in st.session_state:
                l = st.slider("L", 0, 65535, 32000); w = st.slider("W", 100, 65535, 40000)
                fig, ax = plt.subplots(figsize=(7,7), facecolor='black')
                ax.imshow(st.session_state.s_img, cmap='gray_r', vmin=l-w//2, vmax=l+w//2)
                ax.axis('off'); st.pyplot(fig)
        
        with tab2:
            r_kv, r_mas = calculate_exposure_logic(mat, thick)
            st.metric("Valore kV Ideale", f"{r_kv} kV")
            st.metric("Valore mAs Ideale", f"{r_mas} mAs")

        with tab3:
            if 's_img' in st.session_state:
                st.subheader("Analisi SNR (ASTM E2597)")
                st.write("Sposta i cursori per analizzare il rumore in una zona piatta del pezzo.")
                
                roi_x = st.slider("X ROI", 0, 550, 280)
                roi_y = st.slider("Y ROI", 0, 550, 280)
                roi_size = 40
                
                # Calcolo SNR
                roi_data = st.session_state.s_img[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                signal = np.mean(roi_data)
                noise = np.std(roi_data)
                snr = signal / noise if noise > 0 else 0
                
                col_snr1, col_snr2 = st.columns(2)
                col_snr1.metric("Segnale Medio (GV)", f"{int(signal)}")
                col_snr2.metric("SNR Calcolato", f"{snr:.1f}")
                
                if snr < 100:
                    st.error("❌ SNR INSUFFICIENTE (Sotto 100:1). Aumentare mAs.")
                elif snr < 150:
                    st.warning("⚠️ SNR ACCETTABILE (Classe B). Ottimizzazione possibile.")
                else:
                    st.success("✅ SNR OTTIMO. Immagine ad alta sensibilità.")

                # Mostra la ROI ingrandita per vedere la grana
                fig_roi, ax_roi = plt.subplots(figsize=(3,3))
                ax_roi.imshow(roi_data, cmap='gray_r')
                ax_roi.set_title("Dettaglio Grana ROI")
                st.pyplot(fig_roi)

elif mode == "ESAME":
    st.title("🎓 Sessione d'Esame")
    st.write("In questa modalità devi dimostrare di saper ottenere un SNR adeguato e identificare i difetti.")
    # (Logica esame simile alla v14.0)

elif mode == "DATABASE":
    st.title("📊 Registro Istruttore")
    conn = init_db(); df = pd.read_sql_query("SELECT * FROM results", conn); st.table(df)