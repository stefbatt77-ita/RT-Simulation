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
st.set_page_config(page_title="Aero-NDT Geometry v15.0", layout="wide", page_icon="✈️")

# --- CSS PROFESSIONALE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 4px; height: 3em; }
    .stMetric { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    h1, h2, h3 { color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 1. GESTIONE DATI E UTENTI
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
# 2. MOTORE GEOMETRICO AVANZATO
# ==============================================================================
def create_geometry(geo_type, size, base_thick):
    """
    Genera una mappa di spessore 2D (Heightmap) basata sul processo di fabbricazione.
    """
    m_sp = np.full((size, size), float(base_thick), dtype=float)
    y, x = np.ogrid[:size, :size]
    
    if geo_type == "Lastra Piana (Base)":
        return m_sp, 0 # Nessun offset extra
        
    elif geo_type == "Saldatura (Butt Weld)":
        # Cordone centrale verticale
        center = size // 2
        width = 60
        # Profilo Gaussiano per il sovrametallo (Crown)
        reinforcement = 3.0 * np.exp(-((x - center)**2) / (2 * (width/3)**2))
        # Aggiunta irregolarità del cordone (Ripple)
        ripple = 0.5 * np.sin(y * 0.1) * (reinforcement > 0.5)
        m_sp += reinforcement + ripple
        return m_sp, center # Restituisce centro per posizionare difetti
        
    elif geo_type == "Fusione (Step Wedge)":
        # Gradini di spessore diverso
        # Zona 1: Base (SX)
        # Zona 2: +5mm (Centro)
        m_sp[:, 200:400] += 5.0
        # Zona 3: +10mm (DX)
        m_sp[:, 400:] += 10.0
        # Aggiunta di un "mozzo" (Boss) circolare
        mask_boss = (x-500)**2 + (y-300)**2 <= 60**2
        m_sp[mask_boss] += 8.0
        return m_sp, 300 # Centro boss
        
    elif geo_type == "Assemblaggio (Rivetti)":
        # Sovrapposizione di due lamiere (Lap Joint) a metà
        m_sp[:, 300:] += base_thick # Raddoppio spessore a destra
        
        # Fila di rivetti (Materiale più denso/spesso)
        # I rivetti sono in acciaio su alluminio solitamente, simuliamo come extra spessore
        rivet_x = [280, 320] # Una fila sulla lamiera singola (errore design?), una sulla doppia
        for rx_pos in rivet_x:
            for ry_pos in range(50, 550, 100):
                mask_rivet = (x - rx_pos)**2 + (y - ry_pos)**2 <= 10**2
                m_sp[mask_rivet] += 4.0 # Testa del rivetto
        return m_sp, 320

    return m_sp, 0

# ==============================================================================
# 3. MOTORE FISICO & RENDER
# ==============================================================================
def apply_detector_defects(image, is_calibrated):
    if is_calibrated: return image
    rows, cols = image.shape
    noisy = image.astype(float)
    # Heel Effect
    grad = np.linspace(0.85, 1.0, cols)
    noisy *= grad
    # Dead Pixels
    for _ in range(20):
        ry, rx = random.randint(0, rows-1), random.randint(0, cols-1)
        noisy[ry, rx] = 65000 if random.random()>0.5 else 0
    return np.clip(noisy, 0, 65535).astype(np.uint16)

def add_realistic_physics(base_signal, thickness_map, size):
    # Scatter (più forte dove è più spesso)
    scatter_map = gaussian_filter(thickness_map, sigma=50)
    scatter_factor = scatter_map / np.max(scatter_map)
    base_signal += base_signal * scatter_factor * 0.4 # 40% scatter
    
    # Texture & Noise
    textured = base_signal * np.random.normal(1.0, 0.005, (size, size))
    noise = np.random.normal(0, np.sqrt(np.maximum(textured, 10)) * 2.0, (size, size))
    return gaussian_filter(textured + noise, sigma=0.6)

def generate_scan_core(kv, ma, time, material, thickness, selected_defect, iqi_mode, geo_type):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.025, "Ti-6Al-4V": 0.05, "Inconel 718": 0.1, "Steel 17-4 PH": 0.08}
    mu = mu_map[material] * (130/max(20, kv))**1.6 
    
    # 1. CREAZIONE GEOMETRIA
    m_sp, geo_center = create_geometry(geo_type, size, thickness)
    
    # 2. IQI (Posizionato sempre nella zona base o zona critica)
    if iqi_mode == "ISO 19232-1 (Fili)":
        for i in range(7): m_sp[40:140, 80+i*40:82+i*40] += (0.4 - i*0.05)
    else: # ASTM
        plaque_t = max(0.1, thickness * 0.02)
        m_sp[50:110, 60:180] += plaque_t
        for i, d in enumerate([1, 2, 4]):
            r = (plaque_t * d / 0.05) / 2 * 3
            y, x = np.ogrid[:size, :size]
            m_sp[(x-(80+i*40))**2 + (y-80)**2 <= r**2] -= plaque_t

    # Duplex
    for i in range(13): m_sp[500:540, 150+i*25:152+i*25] += (0.7 / (i+1))
    
    # 3. DIFETTI CONTESTUALIZZATI
    bboxes, def_list = [], []
    to_gen = [selected_defect] if selected_defect not in ["Nessun Difetto", "Casuale (Multiplo)"] else []
    if selected_defect == "Casuale (Multiplo)": to_gen = random.sample(["Cricca", "Porosità", "Tungsteno", "Mancata Fusione"], random.randint(1,2))
    
    for d_type in to_gen:
        # Logica posizionamento intelligente in base alla geometria
        if geo_type == "Saldatura (Butt Weld)":
            # Difetti nel cordone (centro X +/- 20)
            rx = random.randint(geo_center-15, geo_center+15)
            ry = random.randint(150, 450)
        elif geo_type == "Assemblaggio (Rivetti)":
            # Difetti vicino ai rivetti o nella sovrapposizione
            rx = random.randint(280, 340)
            ry = random.randint(150, 450)
        else:
            rx, ry = random.randint(150, 450), random.randint(150, 450)

        if d_type == "Cricca":
            # Cricca longitudinale o trasversale
            length = random.randint(40, 80)
            if geo_type == "Saldatura (Butt Weld)":
                m_sp[ry:ry+length, rx:rx+2] -= 0.8 # Longitudinale
                bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": length+20, "t": d_type})
            else:
                m_sp[ry:ry+2, rx:rx+length] -= 0.8 # Trasversale/Random
                bboxes.append({"x": rx-10, "y": ry-10, "w": length+20, "h": 20, "t": d_type})
                
        elif d_type == "Porosità":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 8**2] -= 2.5
            bboxes.append({"x": rx-15, "y": ry-15, "w": 30, "h": 30, "t": d_type})
            
        elif d_type == "Tungsteno":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 4**2] += 15.0
            bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 20, "t": d_type})
            
        elif d_type == "Mancata Fusione":
            if geo_type == "Saldatura (Butt Weld)":
                m_sp[ry:ry+100, geo_center-10:geo_center-8] -= 1.5 # Al bordo del cordone
                bboxes.append({"x": geo_center-20, "y": ry, "w": 20, "h": 100, "t": d_type})
            else:
                m_sp[ry:ry+50, rx:rx+2] -= 1.5
                bboxes.append({"x": rx-10, "y": ry, "w": 20, "h": 50, "t": d_type})
                
        def_list.append(d_type)

    dose = (kv**2) * ma * time * 0.22
    signal = add_realistic_physics(dose * np.exp(-mu * m_sp), m_sp, size)
    return np.clip(signal, 0, 65535).astype(np.uint16), def_list, bboxes

# ==============================================================================
# 4. CALCOLO ESPOSIZIONE
# ==============================================================================
def calculate_exposure_logic(material, thickness, geo_type):
    data = {
        "Al-2024 (Avional)": {"base_kv": 45, "slope": 3.2, "mu_ref": 0.12},
        "Ti-6Al-4V":         {"base_kv": 60, "slope": 4.8, "mu_ref": 0.18},
        "Inconel 718":       {"base_kv": 90, "slope": 7.8, "mu_ref": 0.28},
        "Steel 17-4 PH":     {"base_kv": 85, "slope": 6.5, "mu_ref": 0.25}
    }
    m = data[material]
    
    # Adattiamo lo spessore percepito in base alla geometria (si calcola sul punto più spesso)
    max_thick = thickness
    if geo_type == "Saldatura (Butt Weld)": max_thick += 3.0 # Rinforzo
    elif geo_type == "Fusione (Step Wedge)": max_thick += 10.0 # Parte spessa
    elif geo_type == "Assemblaggio (Rivetti)": max_thick *= 2.0 # Sovrapposizione
    
    rec_kv = int(m["base_kv"] + (max_thick * m["slope"]))
    rec_mas = 12 * np.exp(max_thick * m["mu_ref"] * 0.5)
    return rec_kv, round(rec_mas, 1)

# ==============================================================================
# 5. INTERFACCIA
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
    st.write(f"👤 {st.session_state.username}")
    if st.button("Logout"): st.session_state.logged_in = False; st.rerun()
    st.divider()
    if st.button("🛠️ CALIBRA DETECTOR", type="primary"): 
        st.session_state.is_calibrated = True; st.rerun()
    if st.session_state.is_calibrated: st.success("Detector Ready")
    else: st.error("Detector Uncalibrated")
    mode = st.radio("Seleziona Area", ["STUDIO", "ESAME", "DATABASE"])

if mode == "STUDIO":
    st.title("📘 Laboratorio Geometrie Complesse")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Componente")
        geo = st.selectbox("Tipo Fabbricazione", ["Lastra Piana (Base)", "Saldatura (Butt Weld)", "Fusione (Step Wedge)", "Assemblaggio (Rivetti)"])
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.slider("Spessore Base (mm)", 2, 25, 10)
        
        st.subheader("Impostazioni Difetto")
        def_sel = st.selectbox("Difetto", ["Casuale (Multiplo)", "Cricca", "Porosità", "Tungsteno", "Mancata Fusione", "Nessun Difetto"])
        iqi = st.radio("IQI", ["ISO 19232-1 (Fili)", "ASTM E1025 (Fori)"])
        
        st.markdown("---")
        st.subheader("Parametri X-Ray")
        kv = st.slider("Tensione (kV)", 40, 250, 100)
        ma = st.slider("Corrente (mA)", 1.0, 10.0, 4.0)
        sec = st.slider("Tempo (s)", 1, 60, 15)
        
        if st.button("▶️ ESEGUI SCANSIONE"):
            img, defs, bboxes = generate_scan_core(kv, ma, sec, mat, thick, def_sel, iqi, geo)
            st.session_state.s_img = apply_detector_defects(img, st.session_state.is_calibrated)
            st.session_state.s_defs = defs; st.session_state.s_bboxes = bboxes

    with c2:
        tab1, tab2, tab3 = st.tabs(["🖼️ Analisi", "🧮 Calcolatore", "📏 SNR"])
        
        with tab1:
            if 's_img' in st.session_state:
                l = st.slider("Level", 0, 65535, 32000); w = st.slider("Width", 100, 65535, 40000)
                fig, ax = plt.subplots(figsize=(7,7), facecolor='black')
                ax.imshow(st.session_state.s_img, cmap='gray_r', vmin=l-w//2, vmax=l+w//2)
                
                # Checkbox per rivelare la soluzione
                if st.checkbox("Mostra Soluzione (Box Rossi)"):
                    for b in st.session_state.s_bboxes:
                        rect = patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                
                ax.axis('off'); st.pyplot(fig)
        
        with tab2:
            st.info(f"Geometria: {geo}")
            r_kv, r_mas = calculate_exposure_logic(mat, thick, geo)
            st.metric("kV Ottimali (Max Spessore)", f"{r_kv} kV")
            st.metric("mAs Ottimali", f"{r_mas} mAs")
            st.caption("Nota: In geometrie complesse, calcola l'esposizione sulla parte più spessa per evitare sottoesposizione (rumore), ma controlla di non bruciare le parti sottili (Latitude).")

        with tab3:
            if 's_img' in st.session_state:
                st.write("Analisi SNR ROI")
                rx = st.slider("X",0,550,300); ry = st.slider("Y",0,550,300)
                roi = st.session_state.s_img[ry:ry+30, rx:rx+30]
                snr = np.mean(roi)/np.std(roi) if np.std(roi)>0 else 0
                st.metric("SNR", f"{snr:.1f}")
                st.image(roi, caption="ROI Detail", width=100, clamp=True)

elif mode == "ESAME":
    st.title("🎓 Esame Pratico")
    if not st.session_state.is_calibrated: st.error("Calibra prima!"); st.stop()
    
    if st.session_state.get('exam_step', 0) == 0:
        if st.button("Start Exam"): st.session_state.exam_step = 1; st.session_state.e_res = []; st.rerun()
    elif st.session_state.exam_step <= 3:
        st.subheader(f"Quesito {st.session_state.exam_step}/3")
        geo_e = random.choice(["Saldatura (Butt Weld)", "Fusione (Step Wedge)", "Assemblaggio (Rivetti)"])
        mat_e = "Ti-6Al-4V"
        thick_e = 10
        st.info(f"Ispeziona: **{geo_e}** in {mat_e}, spessore base {thick_e}mm.")
        
        k = st.number_input("kV", 40,250,100); m = st.number_input("mA", 1.0,10.0,5.0); t = st.number_input("Sec", 1,60,10)
        if st.button("Scatta"):
            img, d, _ = generate_scan_core(k, m, t, mat_e, thick_e, "Casuale (Multiplo)", "ISO 19232-1 (Fili)", geo_e)
            st.session_state.e_img = img; st.session_state.e_defs = d
            
        if 'e_img' in st.session_state:
            fig, ax = plt.subplots(); ax.imshow(st.session_state.e_img, cmap='gray_r'); st.pyplot(fig)
            ans = st.multiselect("Diagnosi", ["Cricca", "Porosità", "Tungsteno", "Mancata Fusione"])
            if st.button("Conferma"):
                score = 1 if set(ans) == set(st.session_state.e_defs) else 0
                st.session_state.e_res.append(score)
                st.session_state.exam_step += 1; del st.session_state.e_img; st.rerun()
    else:
        tot = sum(st.session_state.e_res)
        st.success(f"Voto: {tot}/3")
        if st.button("Reset"): st.session_state.exam_step = 0; st.rerun()

elif mode == "DATABASE":
    st.title("Admin Panel")
    conn = init_db(); st.table(pd.read_sql_query("SELECT * FROM results", conn))