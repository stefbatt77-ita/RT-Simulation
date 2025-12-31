import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, laplace
import random
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
from fpdf import FPDF
import io
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Aero-NDT Enterprise v12.0", layout="wide", page_icon="✈️")

# --- CSS PROFESSIONALE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 4px; height: 3em; }
    .reportview-container .main .block-container{ max-width: 95%; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. GESTIONE DATABASE (SQLITE) ---
def init_db():
    conn = sqlite3.connect('ndt_academy.db')
    c = conn.cursor()
    # Tabella Utenti
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    # Tabella Risultati
    c.execute('''CREATE TABLE IF NOT EXISTS results 
                 (id INTEGER PRIMARY KEY, student TEXT, date TEXT, score INTEGER, details TEXT)''')
    
    # Utenti Default (Se non esistono)
    try:
        c.execute("INSERT INTO users VALUES ('admin', 'admin123', 'istruttore')")
        c.execute("INSERT INTO users VALUES ('student', 'ndt2025', 'studente')")
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    return conn

def login_user(username, password):
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

def save_result(student, score, details):
    conn = init_db()
    c = conn.cursor()
    c.execute("INSERT INTO results (student, date, score, details) VALUES (?, ?, ?, ?)", 
              (student, datetime.now().strftime("%Y-%m-%d %H:%M"), score, str(details)))
    conn.commit()

# --- 2. SISTEMA DI REPORTISTICA PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AERO-NDT CERTIFICATION REPORT', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(student, results):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Info Studente
    pdf.cell(200, 10, txt=f"Candidate: {student}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
    pdf.cell(200, 10, txt=f"Standard: NAS 410 / EN4179", ln=1)
    pdf.ln(10)
    
    # Tabella Risultati
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(30, 10, "Case", 1)
    pdf.cell(60, 10, "Material", 1)
    pdf.cell(30, 10, "Technique", 1)
    pdf.cell(40, 10, "Diagnosis", 1)
    pdf.ln()
    
    pdf.set_font("Arial", size=10)
    correct_count = 0
    for res in results:
        is_ok = "PASS" if res['Diagnosi'] == "CORRETTO" else "FAIL"
        if is_ok == "PASS": correct_count += 1
        
        pdf.cell(30, 10, str(res['Caso']), 1)
        pdf.cell(60, 10, res['Materiale'][:25], 1) # Troncato
        pdf.cell(30, 10, res['kV Voto'], 1)
        
        # Colore per esito
        if is_ok == "PASS": pdf.set_text_color(0, 150, 0)
        else: pdf.set_text_color(200, 0, 0)
        pdf.cell(40, 10, is_ok, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    final_score = (correct_count / 5) * 100
    status = "CERTIFIED LEVEL II" if final_score >= 80 else "NOT CERTIFIED"
    pdf.cell(0, 10, f"FINAL SCORE: {final_score}% - {status}", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 3. MOTORE FISICO CON DIFETTI RIVELATORE ---
def apply_detector_defects(image, is_calibrated):
    """Simula difetti hardware se non calibrato."""
    if is_calibrated:
        return image
    
    noisy_img = image.copy().astype(float)
    rows, cols = image.shape
    
    # 1. Heel Effect (Gradiente di intensità)
    gradient = np.linspace(0.85, 1.0, cols)
    noisy_img = noisy_img * gradient
    
    # 2. Bad Pixels (Dead & Hot pixels)
    # Aggiungiamo rumore sale e pepe fisso
    num_bad = 500
    for _ in range(num_bad):
        ry, rx = random.randint(0, rows-1), random.randint(0, cols-1)
        noisy_img[ry, rx] = 0 if random.random() < 0.5 else 65535
        
    # 3. Fixed Pattern Noise (Righe verticali)
    for i in range(0, cols, 50):
        noisy_img[:, i] = noisy_img[:, i] * 0.9
        
    return np.clip(noisy_img, 0, 65535).astype(np.uint16)

# ... (Funzioni get_ideal_params, generate_scan_final, grade_param identiche alla v10.7/10.8) ...
# Riporto qui le funzioni core per completezza
def get_ideal_params(material, thickness):
    props = {"Al-2024 (Avional)": {"b": 45, "k": 3.2}, "Ti-6Al-4V": {"b": 65, "k": 4.8}, "Inconel 718": {"b": 95, "k": 7.8}, "Steel 17-4 PH": {"b": 85, "k": 6.5}}
    p = props[material]
    id_kv = max(40, min(250, int(p["b"] + (thickness * p["k"]))))
    id_mas = round(12 * np.exp(thickness * 0.06), 1)
    return id_kv, id_mas

def generate_scan_core(kv, ma, time, material, thickness, selected_defect, iqi_mode):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[material] * (120/max(10, kv))**1.5
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # IQI
    if iqi_mode == "ISO 19232-1 (Fili)":
        for i in range(7): m_sp[40:140, 80+i*40:82+i*40] += (0.4 - i*0.05)
    else: # ASTM
        m_sp[50:110, 60:180] += max(0.1, thickness * 0.02)
        for i, d in enumerate([1, 2, 4]):
            r = (max(0.1, thickness * 0.02) * d / 0.05) / 2 * 3
            y, x = np.ogrid[:size, :size]
            m_sp[(x-(80+i*30))**2 + (y-80)**2 <= r**2] -= max(0.1, thickness * 0.02)

    # Duplex
    for i in range(13): m_sp[500:540, 150+i*25:152+i*25] += (0.7 / (i+1))
    
    # Defects
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
            bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 20, "t": d_type})
        
        defects_list.append(d_type)

    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    raw = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    return raw, defects_list, bboxes

def grade_param(u, i):
    p = (abs(u - i) / i) * 100
    if p <= 10: return "OTTIMO"
    if p <= 30: return "BUONO"
    if p <= 50: return "SUFFICIENTE"
    return "INSUFFICIENTE"

# --- INIZIALIZZAZIONE STATO ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'is_calibrated' not in st.session_state: st.session_state.is_calibrated = False
if 'exam_results' not in st.session_state: st.session_state.exam_results = []
if 'line_profile' not in st.session_state: st.session_state.line_profile = False

# --- LOGICA DI LOGIN ---
if not st.session_state.logged_in:
    st.title("🔒 Aero-NDT Secure Login")
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("LOGIN"):
            role = login_user(user, pwd)
            if role:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.session_state.role = role[0]
                st.rerun()
            else:
                st.error("Credenziali Errate. (Prova: admin/admin123 o student/ndt2025)")
    st.stop()

# --- APP PRINCIPALE ---
with st.sidebar:
    st.success(f"Utente: {st.session_state.username} ({st.session_state.role})")
    if st.button("LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.divider()
    st.title("🎛️ Controlli Sistema")
    
    # CALIBRATION WIZARD
    st.subheader("🛠️ Detector Status")
    if st.session_state.is_calibrated:
        st.success("✅ CALIBRATED (Gain/Offset OK)")
        if st.button("Reset Calibration"):
            st.session_state.is_calibrated = False
            st.rerun()
    else:
        st.error("❌ UNCALIBRATED (Artifacts Active)")
        if st.button("PERFORM CALIBRATION"):
            with st.spinner("Acquiring Dark/White fields..."):
                st.session_state.is_calibrated = True
            st.rerun()
            
    st.divider()
    mode = st.radio("Modalità", ["STUDIO", "ESAME", "DASHBOARD (Istruttore)"])

# === MODALITÀ ISTRUTTORE ===
if mode == "DASHBOARD (Istruttore)":
    if st.session_state.role != "istruttore":
        st.error("Accesso negato. Solo per Istruttori.")
    else:
        st.title("📊 Dashboard Istruttore")
        conn = init_db()
        df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
        st.dataframe(df)
        st.download_button("Scarica Database CSV", df.to_csv(), "ndt_db.csv")

# === MODALITÀ STUDIO ===
elif mode == "STUDIO":
    st.title("📘 Studio Avanzato & Analisi")
    
    if not st.session_state.is_calibrated:
        st.warning("⚠️ ATTENZIONE: Il pannello non è calibrato. Potresti vedere pixel morti o gradienti. Calibra dalla sidebar.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Setup")
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 5, 30, 10)
        def_c = st.selectbox("Difetto", ["Casuale (Multiplo)", "Cricca", "Porosità Singola", "Inclusione Tungsteno"])
        iqi_m = st.radio("IQI", ["ISO 19232-1 (Fili)", "ASTM E1025 (Fori)"])
        
        st.markdown("---")
        kv = st.slider("kV", 40, 250, 90)
        ma = st.slider("mA", 1.0, 15.0, 5.0)
        ti = st.slider("Tempo", 1, 120, 25)
        
        if st.button("ACQUISICI"):
            raw, defs, bbox = generate_scan_core(kv, ma, ti, mat, thick, def_c, iqi_m)
            # APPLICAZIONE DIFETTI RIVELATORE
            final_img = apply_detector_defects(raw, st.session_state.is_calibrated)
            
            st.session_state.s_img = final_img
            st.session_state.s_bboxes = bbox
            st.session_state.s_defs = defs
            st.session_state.s_eval = False

    with c2:
        if 's_img' in st.session_state:
            lev = st.slider("L", 0, 65535, 32768)
            wid = st.slider("W", 100, 65535, 40000)
            
            fig, ax = plt.subplots(facecolor='black')
            ax.imshow(st.session_state.s_img, cmap='gray_r', vmin=lev-wid//2, vmax=lev+wid//2)
            
            # Show Bbox if evaluated
            if st.session_state.s_eval:
                for b in st.session_state.s_bboxes:
                    ax.add_patch(patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=2, edgecolor='red', facecolor='none'))
            
            # Line Profile Overlay
            if st.session_state.line_profile:
                row = st.slider("Riga Profilo (Y)", 0, 600, 300)
                ax.axhline(row, color='cyan', linestyle='--')
            
            ax.axis('off')
            st.pyplot(fig)
            
            # STRUMENTI ANALISI
            tabs = st.tabs(["🔎 Analisi", "📈 Line Profile", "📏 SNR"])
            
            with tabs[0]:
                if st.button("VERIFICA"): st.session_state.s_eval = True
                if st.session_state.s_eval:
                    st.info(f"Difetti: {st.session_state.s_defs}")
            
            with tabs[1]:
                st.checkbox("Mostra riga su immagine", key="line_profile")
                if st.session_state.line_profile:
                    prof_data = st.session_state.s_img[row, :]
                    fig_p, ax_p = plt.subplots(figsize=(6, 2))
                    ax_p.plot(prof_data, color='cyan')
                    ax_p.set_title("Density Profile")
                    ax_p.grid(True, alpha=0.3)
                    st.pyplot(fig_p)
            
            with tabs[2]:
                st.write("Calcolo SNR (ASTM E2597)")
                c_snr = st.columns(2)
                x1 = c_snr[0].number_input("X ROI", 0, 600, 300)
                y1 = c_snr[1].number_input("Y ROI", 0, 600, 300)
                roi = st.session_state.s_img[y1:y1+20, x1:x1+20]
                mean, std = np.mean(roi), np.std(roi)
                snr = mean/std if std > 0 else 0
                st.metric("SNR Basic", f"{snr:.2f}")
                st.caption("ROI size: 20x20 px")

# === MODALITÀ ESAME ===
elif mode == "ESAME":
    st.title("🎓 Certificazione Ufficiale")
    
    if not st.session_state.is_calibrated:
        st.error("⛔ IMPOSSIBILE PROCEDERE: Il sistema non è calibrato. Esegui la calibrazione prima dell'esame.")
        st.stop()

    if 'exam_progress' not in st.session_state: st.session_state.exam_progress = 0
    
    if st.session_state.exam_progress == 0:
        if st.button("INIZIA ESAME (5 Casi)"):
            st.session_state.exam_cases = [{"mat": random.choice(["Ti-6Al-4V", "Steel 17-4 PH"]), "thick": random.randint(8, 25), "defect": "Casuale (Multiplo)"} for _ in range(5)]
            st.session_state.exam_progress = 1
            st.rerun()
            
    elif st.session_state.exam_progress > 5:
        st.success("ESAME COMPLETATO")
        df = pd.DataFrame(st.session_state.exam_results)
        st.table(df)
        
        # Calcolo Score
        score = (df[df["Diagnosi"] == "CORRETTO"].shape[0] / 5) * 100
        st.metric("Punteggio Finale", f"{score}%")
        
        # Salvataggio DB
        if st.button("SALVA E GENERA PDF"):
            save_result(st.session_state.username, int(score), st.session_state.exam_results)
            pdf_bytes = generate_pdf(st.session_state.username, st.session_state.exam_results)
            st.download_button("⬇️ Scarica Certificato PDF", pdf_bytes, "certificato.pdf", "application/pdf")
            
        if st.button("Nuova Sessione"):
            st.session_state.exam_progress = 0
            st.session_state.exam_results = []
            st.rerun()
            
    else:
        # Codice esecuzione esame (semplificato per brevità, usa la logica v10.7 integrando calibrazione)
        idx = st.session_state.exam_progress
        case = st.session_state.exam_cases[idx-1]
        st.subheader(f"Caso {idx}/5 - {case['mat']} {case['thick']}mm")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            k = st.slider("kV", 40, 250, 100, key=f"k{idx}")
            m = st.slider("mA", 1.0, 15.0, 5.0, key=f"m{idx}")
            t = st.slider("Sec", 1, 120, 20, key=f"t{idx}")
            if st.button("SCATTA"):
                raw, d, b = generate_scan_core(k, m, t, case['mat'], case['thick'], case['defect'], "ISO 19232-1 (Fili)")
                st.session_state.e_img = apply_detector_defects(raw, True) # In esame assumiamo calibrazione fatta
                st.session_state.e_defs = d
        
        with c2:
            if 'e_img' in st.session_state and st.session_state.e_img is not None:
                l, w = st.slider("L", 0, 65535, 32768, key=f"l{idx}"), st.slider("W", 100, 65535, 40000, key=f"w{idx}")
                fig, ax = plt.subplots(facecolor='black')
                ax.imshow(st.session_state.e_img, cmap='gray_r', vmin=l-w//2, vmax=l+w//2)
                ax.axis('off')
                st.pyplot(fig)
                
                ans = st.multiselect("Difetti:", ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"], key=f"a{idx}")
                if st.button("CONFERMA"):
                    ok = set(ans) == set(st.session_state.e_defs)
                    ik, im = get_ideal_params(case['mat'], case['thick'])
                    st.session_state.exam_results.append({
                        "Caso": idx, "Materiale": case['mat'],
                        "kV Voto": grade_param(k, ik), "Diagnosi": "CORRETTO" if ok else "ERRATO"
                    })
                    st.session_state.exam_progress += 1
                    st.session_state.e_img = None
                    st.rerun()