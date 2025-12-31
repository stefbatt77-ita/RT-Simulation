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
st.set_page_config(page_title="Aero-NDT Photorealistic v13.0", layout="wide", page_icon="✈️")

# --- CSS PROFESSIONALE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 4px; height: 3em; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# SEZIONE 1: GESTIONE DATABASE E REPORTISTICA
# ==============================================================================
def init_db():
    conn = sqlite3.connect('ndt_academy.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, student TEXT, date TEXT, score INTEGER, details TEXT)''')
    try:
        c.execute("INSERT INTO users VALUES ('admin', 'admin123', 'istruttore')")
        c.execute("INSERT INTO users VALUES ('student', 'ndt2025', 'studente')")
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

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15); self.cell(0, 10, 'AERO-NDT CERTIFICATION REPORT', 0, 1, 'C'); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(student, results):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Candidate: {student}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
    pdf.cell(200, 10, txt=f"Standard: NAS 410 / EN4179 (DDA Tech)", ln=1)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(30, 10, "Case", 1); pdf.cell(60, 10, "Material", 1); pdf.cell(30, 10, "kV Rating", 1); pdf.cell(40, 10, "Diagnosis", 1); pdf.ln()
    pdf.set_font("Arial", size=10)
    correct_count = 0
    for res in results:
        is_ok = "PASS" if res['Diagnosi'] == "CORRETTO" else "FAIL"
        if is_ok == "PASS": correct_count += 1
        pdf.cell(30, 10, str(res['Caso']), 1); pdf.cell(60, 10, res['Materiale'][:25], 1); pdf.cell(30, 10, res['kV Voto'], 1)
        if is_ok == "PASS": pdf.set_text_color(0, 150, 0)
        else: pdf.set_text_color(200, 0, 0)
        pdf.cell(40, 10, is_ok, 1); pdf.set_text_color(0, 0, 0); pdf.ln()
    pdf.ln(10); pdf.set_font("Arial", 'B', 14)
    final_score = (correct_count / 5) * 100
    status = "CERTIFIED LEVEL II" if final_score >= 80 else "NOT CERTIFIED"
    pdf.cell(0, 10, f"FINAL SCORE: {final_score}% - {status}", 0, 1, 'C')
    return pdf.output(dest='S').encode('latin-1')

# ==============================================================================
# SEZIONE 2: MOTORE FISICO FOTOREALISTICO
# ==============================================================================
def apply_detector_defects(image, is_calibrated):
    """Simula difetti realistici di un DDA non calibrato."""
    if is_calibrated: return image
    rows, cols = image.shape
    noisy_img = image.astype(float)
    
    # 1. Heel Effect & Gain Variations (Macchie strutturali)
    gradient_x = np.linspace(0.92, 1.0, cols)
    row_gain = np.random.normal(1.0, 0.03, (rows, 1))
    col_gain = np.random.normal(1.0, 0.03, (1, cols))
    gain_map = row_gain * col_gain * gradient_x
    noisy_img = noisy_img * gain_map

    # 2. Bad Pixels (Cluster e singoli)
    for _ in range(8): # Cluster morti
        cy, cx = random.randint(0, rows), random.randint(0, cols)
        y, x = np.ogrid[:rows, :cols]
        mask = (y - cy)**2 + (x - cx)**2 <= random.randint(3, 6)**2
        noisy_img[mask] *= 0.2
        
    # Pixel salt & pepper
    mask_dead = np.random.rand(rows,cols) > 0.9995
    mask_hot = np.random.rand(rows,cols) > 0.9995
    noisy_img[mask_dead] = 1000; noisy_img[mask_hot] = 65000
        
    return np.clip(noisy_img, 0, 65535).astype(np.uint16)

def add_realistic_physics(base_signal, thickness_map, size):
    """Applica scatter, texture dello scintillatore e rumore quantico."""
    # A. Scatter Radiation (Velo di fondo diffuso, proporzionale allo spessore)
    scatter_map = gaussian_filter(thickness_map, sigma=40)
    scatter_signal = base_signal * (scatter_map / np.max(scatter_map)) * 0.35
    signal_with_scatter = base_signal + scatter_signal

    # B. Scintillator Texture (Grana strutturale fine)
    texture_noise = np.random.normal(1.0, 0.007, (size, size)) 
    textured_signal = signal_with_scatter * texture_noise

    # C. Poisson Quantum Noise (Dipendente dal segnale)
    safe_signal = np.maximum(textured_signal, 10.0)
    # Rumore proporzionale alla radice del segnale
    quantum_noise = np.random.normal(0, np.sqrt(safe_signal) * 1.8, (size, size))
    
    final_signal = textured_signal + quantum_noise
    # Leggera penombra geometrica finale
    return gaussian_filter(final_signal, sigma=0.8)

def generate_scan_core(kv, ma, time, material, thickness, selected_defect, iqi_mode):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.025, "Ti-6Al-4V": 0.05, "Inconel 718": 0.1, "Steel 17-4 PH": 0.08}
    # Parametri attenuazione ottimizzati per il nuovo modello di rumore
    safe_kv = max(20, kv)
    mu = mu_map[material] * (130/safe_kv)**1.6 
    
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # --- GEOMETRIA (IQI, Difetti) ---
    if iqi_mode == "ISO 19232-1 (Fili)":
        for i in range(7): m_sp[40:140, 80+i*40:82+i*40] += (0.4 - i*0.05)
        m_sp[145:155, 80:120] -= 0.3 # Label Placeholder
    else: # ASTM
        m_sp[50:110, 60:180] += max(0.1, thickness * 0.02)
        for i, d in enumerate([1, 2, 4]):
            r = (max(0.1, thickness * 0.02) * d / 0.05) / 2 * 3
            y, x = np.ogrid[:size, :size]
            m_sp[(x-(80+i*30))**2 + (y-80)**2 <= r**2] -= max(0.1, thickness * 0.02)
        m_sp[115:125, 60:100] -= 0.3 # Label Placeholder

    for i in range(13): m_sp[500:540, 150+i*25:152+i*25] += (0.7 / (i+1))
    m_sp[545:555, 150:200] -= 0.3 # Label DUPLEX
    
    bboxes, defects_list = [], []
    to_gen = random.sample(["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"], random.randint(1,3)) if selected_defect == "Casuale (Multiplo)" else ([selected_defect] if selected_defect != "Nessun Difetto" else [])
    
    for d_type in to_gen:
        rx, ry = random.randint(100, 500), random.randint(180, 450)
        if d_type == "Cricca":
            px, py, cx, cy = [], [], rx, ry
            for _ in range(80):
                cx += np.cos(random.uniform(0, 6.28)); cy += np.sin(random.uniform(0, 6.28))
                if 0<=int(cy)<size and 0<=int(cx)<size: 
                    m_sp[int(cy), int(cx)] -= 0.8
                    px.append(cx); py.append(cy)
            if px: bboxes.append({"x": min(px)-10, "y": min(py)-10, "w": max(px)-min(px)+20, "h": max(py)-min(py)+20, "t": d_type})
        elif d_type == "Porosità Singola":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 6**2] -= 2.5
            bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 20, "t": d_type})
        elif d_type == "Inclusione Tungsteno":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 4**2] += 15.0
            bboxes.append({"x": rx-10, "y": ry-8, "w": 16, "h": 16, "t": d_type})
        elif d_type == "Cluster Porosità":
            min_x, max_x, min_y, max_y = 600, 0, 600, 0
            for _ in range(12):
                cx, cy = rx + random.randint(-25, 25), ry + random.randint(-25, 25)
                y, x = np.ogrid[:size, :size]
                m_sp[(x-cx)**2 + (y-cy)**2 <= 3**2] -= 1.8
                min_x = min(min_x, cx); max_x = max(max_x, cx); min_y = min(min_y, cy); max_y = max(max_y, cy)
            bboxes.append({"x": min_x-10, "y": min_y-10, "w": max_x-min_x+20, "h": max_y-min_y+20, "t": "Cluster"})
        elif d_type == "Mancata Fusione":
            h = random.randint(60, 150)
            m_sp[ry:ry+h, rx:rx+3] -= 1.6
            bboxes.append({"x": rx-5, "y": ry, "w": 15, "h": h, "t": "LoF"})
        defects_list.append(d_type)

    # --- GENERAZIONE SEGNALE FOTOREALISTICO ---
    # Dose di base aumentata per gestire il nuovo modello di rumore
    dose = (kv**2) * ma * time * 0.25 
    base_signal = dose * np.exp(-mu * m_sp)
    
    # Applicazione fisica avanzata
    realistic_signal = add_realistic_physics(base_signal, m_sp, size)
    
    raw = np.clip(realistic_signal, 0, 65535).astype(np.uint16)
    return raw, defects_list, bboxes

def get_ideal_params(material, thickness):
    props = {"Al-2024 (Avional)": {"b": 45, "k": 3.2}, "Ti-6Al-4V": {"b": 65, "k": 4.8}, "Inconel 718": {"b": 95, "k": 7.8}, "Steel 17-4 PH": {"b": 85, "k": 6.5}}
    p = props[material]
    id_kv = max(40, min(250, int(p["b"] + (thickness * p["k"]))))
    id_mas = round(15 * np.exp(thickness * 0.06), 1)
    return id_kv, id_mas

def grade_param(u, i):
    if i == 0: return "N/A"
    p = (abs(u - i) / i) * 100
    if p <= 10: return "OTTIMO"; 
    if p <= 30: return "BUONO"; 
    if p <= 50: return "SUFFICIENTE"; 
    return "INSUFFICIENTE"

# ==============================================================================
# SEZIONE 3: INTERFACCIA UTENTE E LOGICA
# ==============================================================================
# Inizializzazione Stato
keys = ['logged_in', 'is_calibrated', 'exam_progress', 'exam_results', 's_img', 's_eval', 'line_profile']
for k in keys:
    if k not in st.session_state:
        if k == 'exam_results': st.session_state[k] = []
        elif k == 'exam_progress': st.session_state[k] = 0
        elif k == 'logged_in' or k == 'is_calibrated' or k == 's_eval' or k == 'line_profile': st.session_state[k] = False
        else: st.session_state[k] = None

# --- LOGIN ---
if not st.session_state.logged_in:
    st.title("🔒 Aero-NDT Enterprise Login")
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("LOGIN"):
            role = login_user(user, pwd)
            if role:
                st.session_state.logged_in = True; st.session_state.username = user; st.session_state.role = role[0]
                st.rerun()
            else: st.error("Credenziali Errate.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.info(f"Utente: {st.session_state.username} ({st.session_state.role})")
    if st.button("LOGOUT"):
        for k in st.session_state.keys(): del st.session_state[k]
        st.rerun()
    
    st.divider(); st.title("🎛️ Sistema DDA")
    st.subheader("🛠️ Stato Rivelatore")
    if st.session_state.is_calibrated:
        st.success("✅ CALIBRATO")
        if st.button("Reset Calibrazione"): st.session_state.is_calibrated = False; st.rerun()
    else:
        st.error("❌ NON CALIBRATO (Artefatti Attivi)")
        if st.button("ESEGUI CALIBRAZIONE (Gain/Offset)"):
            with st.spinner("Acquisizione dark/white fields..."): st.session_state.is_calibrated = True
            st.rerun()
            
    st.divider()
    mode = st.radio("Modalità", ["STUDIO", "ESAME", "DASHBOARD (Istruttore)"])

# === DASHBOARD ISTRUTTORE ===
if mode == "DASHBOARD (Istruttore)":
    if st.session_state.role != "istruttore": st.error("Accesso negato.")
    else:
        st.title("📊 Dashboard Risultati")
        conn = init_db(); df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(), "ndt_db.csv")

# === MODALITÀ STUDIO ===
elif mode == "STUDIO":
    st.title("📘 Studio Avanzato & Analisi")
    if not st.session_state.is_calibrated: st.warning("⚠️ Rivelatore non calibrato. Rumore strutturale e pixel morti attivi.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Setup Tecnica")
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 5, 30, 10)
        def_c = st.selectbox("Difetto", ["Casuale (Multiplo)", "Cricca", "Porosità Singola", "Inclusione Tungsteno", "Mancata Fusione", "Nessun Difetto"])
        iqi_m = st.radio("IQI Standard", ["ISO 19232-1 (Fili)", "ASTM E1025 (Fori)"])
        st.divider()
        kv = st.slider("kV", 40, 250, 90); ma = st.slider("mA", 1.0, 15.0, 5.0); ti = st.slider("Tempo (s)", 1, 120, 25)
        if st.button("ACQUISICI"):
            raw, defs, bbox = generate_scan_core(kv, ma, ti, mat, thick, def_c, iqi_m)
            st.session_state.s_img = apply_detector_defects(raw, st.session_state.is_calibrated)
            st.session_state.s_bboxes = bbox; st.session_state.s_defs = defs; st.session_state.s_eval = False
            st.session_state.s_iqi_type = iqi_m

    with c2:
        if st.session_state.s_img is not None:
            lev = st.slider("L", 0, 65535, 32768); wid = st.slider("W", 100, 65535, 45000)
            fig, ax = plt.subplots(facecolor='black', figsize=(8,8))
            ax.imshow(st.session_state.s_img, cmap='gray_r', vmin=lev-wid//2, vmax=lev+wid//2)
            
            # Etichette IQI
            lbl_iqi = "ISO W10" if st.session_state.s_iqi_type == "ISO 19232-1 (Fili)" else "ASTM E1025"
            ax.text(85, 140, lbl_iqi, color='white', alpha=0.6, fontsize=8, weight='bold')
            ax.text(155, 540, "DUPLEX EN", color='white', alpha=0.6, fontsize=8, weight='bold')

            if st.session_state.s_eval:
                for b in st.session_state.s_bboxes:
                    ax.add_patch(patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=2, edgecolor='red', facecolor='none'))
                    ax.text(b['x'], b['y']-5, b['t'], color='red', fontsize=9, fontweight='bold')
            
            if st.session_state.line_profile:
                row = st.slider("Riga Profilo Y", 0, 599, 300)
                ax.axhline(row, color='cyan', linestyle='--', linewidth=1)

            ax.axis('off'); st.pyplot(fig)
            
            tabs = st.tabs(["🔎 Verifica", "📈 Line Profile", "📏 SNR Tool"])
            with tabs[0]:
                if st.button("VERIFICA DIAGNOSI E PARAMETRI"): st.session_state.s_eval = True
                if st.session_state.s_eval:
                    id_k, id_m = get_ideal_params(mat, thick)
                    st.info(f"Difetti Reali: {', '.join(st.session_state.s_defs) if st.session_state.s_defs else 'Sano'}")
                    res_data = {"Param": ["kV", "mAs"], "User": [f"{kv}", f"{ma*ti:.1f}"], "Ideal": [f"{id_k}", f"{id_m}"], "Grade": [grade_param(kv, id_k), grade_param(ma*ti, id_m)]}
                    st.table(pd.DataFrame(res_data))
            with tabs[1]:
                st.checkbox("Attiva Cursore Profilo", key="line_profile")
                if st.session_state.line_profile:
                    prof = st.session_state.s_img[row, :]
                    fig_p, ax_p = plt.subplots(figsize=(6, 2), facecolor='#1e1e1e')
                    ax_p.plot(prof, color='cyan', linewidth=1); ax_p.set_facecolor('black'); ax_p.grid(True, alpha=0.2, color='white')
                    ax_p.tick_params(colors='white'); st.pyplot(fig_p)
            with tabs[2]:
                st.write("Calcolo SNR (Regione 30x30px)")
                c_snr = st.columns(2)
                x1 = c_snr[0].number_input("X Pos", 0, 570, 300); y1 = c_snr[1].number_input("Y Pos", 0, 570, 300)
                roi = st.session_state.s_img[y1:y1+30, x1:x1+30]
                mean, std = np.mean(roi), np.std(roi)
                snr = mean/std if std > 0 else 0
                st.metric("SNR (Signal-to-Noise)", f"{snr:.1f}", delta=f"Mean GV: {int(mean)}")
                fig_r, ax_r = plt.subplots(figsize=(2,2)); ax_r.imshow(roi, cmap='gray_r'); ax_r.axis('off'); st.pyplot(fig_r)

# === MODALITÀ ESAME ===
elif mode == "ESAME":
    st.title("🎓 Certificazione Ufficiale NAS 410")
    if not st.session_state.is_calibrated: st.error("⛔ Calibrazione richiesta per procedere all'esame."); st.stop()
    
    if st.session_state.exam_progress == 0:
        if st.button("INIZIA SESSIONE D'ESAME (5 Casi)"):
            st.session_state.exam_cases = [{"mat": random.choice(["Ti-6Al-4V", "Steel 17-4 PH", "Inconel 718"]), "thick": random.randint(8, 25), "defect": "Casuale (Multiplo)", "iqi": random.choice(["ISO 19232-1 (Fili)", "ASTM E1025 (Fori)"])} for _ in range(5)]
            st.session_state.exam_progress = 1; st.rerun()
            
    elif st.session_state.exam_progress > 5:
        st.success("ESAME COMPLETATO. Generazione report in corso...")
        df = pd.DataFrame(st.session_state.exam_results)
        st.table(df)
        score = (df[df["Diagnosi"] == "CORRETTO"].shape[0] / 5) * 100
        st.metric("Punteggio Finale", f"{score}%")
        
        if st.button("SALVA RISULTATO E SCARICA PDF"):
            save_result(st.session_state.username, int(score), st.session_state.exam_results)
            pdf_bytes = generate_pdf(st.session_state.username, st.session_state.exam_results)
            st.download_button("⬇️ Scarica Certificato PDF", pdf_bytes, "certificato_ndt.pdf", "application/pdf")
            
        if st.button("Chiudi Sessione"):
            st.session_state.exam_progress = 0; st.session_state.exam_results = []; st.rerun()
            
    else:
        idx = st.session_state.exam_progress; case = st.session_state.exam_cases[idx-1]
        st.subheader(f"Caso {idx}/5 - {case['mat']} {case['thick']}mm | {case['iqi']}")
        c1, c2 = st.columns([1, 2])
        with c1:
            k = st.slider("kV", 40, 250, 100, key=f"k{idx}"); m = st.slider("mA", 1.0, 15.0, 5.0, key=f"m{idx}"); t = st.slider("Sec", 1, 120, 20, key=f"t{idx}")
            if st.button("SCATTA"):
                raw, d, b = generate_scan_core(k, m, t, case['mat'], case['thick'], case['defect'], case['iqi'])
                st.session_state.e_img = apply_detector_defects(raw, True)
                st.session_state.e_defs = d
        with c2:
            if st.session_state.get('e_img') is not None:
                l, w = st.slider("L", 0, 65535, 32768, key=f"l{idx}"), st.slider("W", 100, 65535, 40000, key=f"w{idx}")
                fig, ax = plt.subplots(facecolor='black')
                ax.imshow(st.session_state.e_img, cmap='gray_r', vmin=l-w//2, vmax=l+w//2)
                lbl_iqi = "ISO W10" if case['iqi'] == "ISO 19232-1 (Fili)" else "ASTM E1025"
                ax.text(85, 140, lbl_iqi, color='white', alpha=0.6, fontsize=8, weight='bold')
                ax.text(155, 540, "DUPLEX EN", color='white', alpha=0.6, fontsize=8, weight='bold')
                ax.axis('off'); st.pyplot(fig)
                
                ans = st.multiselect("Difetti Identificati:", ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"], key=f"a{idx}")
                if st.button("CONFERMA DIAGNOSI"):
                    ok = set(ans) == set(st.session_state.e_defs)
                    ik, im = get_ideal_params(case['mat'], case['thick'])
                    st.session_state.exam_results.append({"Caso": idx, "Materiale": case['mat'], "kV Voto": grade_param(k, ik), "Diagnosi": "CORRETTO" if ok else "ERRATO"})
                    st.session_state.exam_progress += 1; st.session_state.e_img = None; st.rerun()