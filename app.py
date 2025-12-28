import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, laplace
import random
import io
import datetime

# --- CONFIGURAZIONE INIZIALE ---
st.set_page_config(page_title="Aero-NDT Exam System v9.0", layout="wide", page_icon="✈️")

# CSS per rendere l'interfaccia simile a un software industriale
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 4px; height: 3em; }
    /* Pulsante Acquisizione */
    div.stButton > button:first-child { background-color: #d32f2f; color: white; border: none; }
    /* Pulsante Conferma Esame */
    div[data-testid="stVerticalBlock"] > div > div > div > div.stButton > button { background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- GESTIONE STATO SESSIONE (Persistenza dati) ---
if 'scan_data' not in st.session_state:
    st.session_state['scan_data'] = None  # Contiene l'immagine raw
if 'exam_state' not in st.session_state:
    st.session_state['exam_state'] = "SETUP" # SETUP -> ACQUIRED -> REVIEWED
if 'ground_truth' not in st.session_state:
    st.session_state['ground_truth'] = {} # Contiene tipo difetto e bbox

# --- MOTORE FISICO (EN4179/NAS 410) ---
def generate_xray(kv, ma, time, mat, thick, iqi_opt):
    size = 800
    # Fisica attenuazione (Simulata)
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[mat] * (120/kv)**1.5
    
    # Matrice base
    m_sp = np.full((size, size), float(thick), dtype=float)
    
    # GENERAZIONE DIFETTI RANDOM
    defects_list = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione", "Nessun Difetto"]
    chosen_defect = random.choice(defects_list)
    
    # Coordinate Bounding Box (x, y, w, h) per la verifica
    bbox = None 
    
    cx, cy = 400, 400
    
    if chosen_defect == "Cricca":
        pts_x, pts_y = [], []
        curr_x = cx
        for y in range(300, 500):
            m_sp[y, int(curr_x)] -= 0.6
            pts_x.append(curr_x); pts_y.append(y)
            curr_x += random.uniform(-0.6, 0.6)
        bbox = {"x": min(pts_x)-5, "y": 300, "w": (max(pts_x)-min(pts_x))+10, "h": 200}
        
    elif chosen_defect == "Porosità Singola":
        y, x = np.ogrid[:size, :size]
        mask = (x-400)**2 + (y-400)**2 <= 6**2
        m_sp[mask] -= 2.0
        bbox = {"x": 390, "y": 390, "w": 20, "h": 20}
        
    elif chosen_defect == "Cluster Porosità":
        min_x, max_x, min_y, max_y = 800, 0, 800, 0
        for _ in range(12):
            rx, ry = random.randint(360, 440), random.randint(360, 440)
            y, x = np.ogrid[:size, :size]
            mask = (x-rx)**2 + (y-ry)**2 <= 3**2
            m_sp[mask] -= 1.5
            min_x, max_x = min(min_x, rx), max(max_x, rx)
            min_y, max_y = min(min_y, ry), max(max_y, ry)
        bbox = {"x": min_x-5, "y": min_y-5, "w": max_x-min_x+10, "h": max_y-min_y+10}
        
    elif chosen_defect == "Inclusione Tungsteno":
        y, x = np.ogrid[:size, :size]
        mask = (x-400)**2 + (y-400)**2 <= 4**2
        m_sp[mask] += 15.0 # Alta densità
        bbox = {"x": 394, "y": 394, "w": 12, "h": 12}
        
    elif chosen_defect == "Incisione Marginale":
        m_sp[200:600, 430:433] -= 1.2
        bbox = {"x": 425, "y": 200, "w": 15, "h": 400}
        
    elif chosen_defect == "Mancata Fusione":
        m_sp[200:600, 398:401] -= 1.8
        bbox = {"x": 395, "y": 200, "w": 10, "h": 400}
        
    elif chosen_defect == "Nessun Difetto":
        bbox = None

    # IQI Duplex (Sempre presente per SRb)
    for i in range(13): 
        m_sp[700:750, 50 + i*25 : 50 + i*25 + 2] += (0.8 / (i+1))
        
    # IQI Selezionato
    if iqi_opt == "ISO 19232-1 (Fili)":
        for i in range(7): m_sp[100:250, 50+i*30:52+i*30] += (0.4 - i*0.05)
    else: # ASTM Holes
        m_sp[100:150, 50:180] += 0.2
        for i, r in enumerate([2, 4, 8]):
            y, x = np.ogrid[:size, :size]
            m_sp[(x-(75+i*35))**2 + (y-125)**2 <= r**2] -= 0.2

    # Engine 16-bit
    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    
    raw = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    return raw, chosen_defect, bbox

# --- INTERFACCIA UTENTE ---
st.title("✈️ Aero-NDT Certification Simulator")
st.markdown("Sistema di addestramento radiografico conforme a **EN4179 / NAS 410**.")

# Layout a due colonne
col_setup, col_work = st.columns([1, 3])

# --- 1. SIDEBAR: SETUP & ACQUISIZIONE ---
with col_setup:
    st.header("1. Setup Acquisizione")
    kv = st.slider("Tensione (kV)", 40, 250, 110)
    ma = st.slider("Corrente (mA)", 0.5, 15.0, 5.0)
    time = st.slider("Tempo (s)", 1, 120, 25)
    
    st.subheader("Campione")
    mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
    thick = st.number_input("Spessore (mm)", 1, 30, 10)
    iqi = st.radio("IQI Standard", ["ISO 19232-1 (Fili)", "ASTM E1025 (Fori)"])
    
    st.markdown("---")
    if st.button("ACQUISICI SCANSIONE (Start Exam)"):
        # Reset Stato
        raw_img, true_def, true_bbox = generate_xray(kv, ma, time, mat, thick, iqi)
        st.session_state['scan_data'] = raw_img
        st.session_state['ground_truth'] = {'type': true_def, 'bbox': true_bbox}
        st.session_state['exam_state'] = "ACQUIRED"
        st.rerun()

    # DICONDE EXPORT (Solo se acquisito)
    if st.session_state['exam_state'] in ["ACQUIRED", "REVIEWED"]:
        st.markdown("---")
        buf = io.BytesIO()
        plt.imsave(buf, st.session_state['scan_data'], cmap='gray_r', format='png')
        st.download_button("💾 Esporta DICONDE (Raw)", buf.getvalue(), "aero_test.dcm", "application/octet-stream")

# --- 2. WORKSPACE: ANALISI & ESAME ---
with col_work:
    if st.session_state['scan_data'] is None:
        st.info("👈 Configura i parametri e premi 'ACQUISICI SCANSIONE' per iniziare la sessione.")
    else:
        # --- TOOLBAR IMMAGINE ---
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: level = st.slider("Livello (Brightness)", 0, 65535, 32768)
        with c2: width = st.slider("Contrasto (Window)", 100, 65535, 65535)
        with c3: sharpen = st.checkbox("Filtro Sharpen")

        # --- RENDERING PLOT ---
        data = st.session_state['scan_data'].astype(float)
        if sharpen:
            data = data + 2.0 * laplace(data)
        
        vmin, vmax = max(0, level - width//2), min(65535, level + width//2)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        ax.imshow(data, cmap='gray_r', vmin=vmin, vmax=vmax)
        
        # --- LOGICA VISUALIZZAZIONE SOLUZIONE (Verifica Finale) ---
        if st.session_state['exam_state'] == "REVIEWED":
            gt = st.session_state['ground_truth']
            if gt['bbox']: # Se c'è un difetto
                # Disegna Box Rosso
                rect = patches.Rectangle((gt['bbox']['x'], gt['bbox']['y']), 
                                         gt['bbox']['w'], gt['bbox']['h'], 
                                         linewidth=2, edgecolor='#ff0000', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(gt['bbox']['x'], gt['bbox']['y']-10, f"POSIZIONE REALE: {gt['type']}", 
                        color='red', fontsize=10, fontweight='bold', backgroundcolor='black')
            else:
                ax.text(400, 400, "PEZZO SANO (Nessun Difetto)", color='#00ff00', 
                        fontsize=15, fontweight='bold', ha='center', backgroundcolor='black')

        ax.axis('off')
        st.pyplot(fig)

        # --- SEZIONE DENSITOMETRIA ---
        center_px = st.session_state['scan_data'][400, 400]
        st.caption(f"Info Sistema: 16-bit DDA | SNR: {np.mean(data)/np.std(data):.2f} | Grigio Centrale: {center_px}")

        # --- SEZIONE ESAME E VERIFICA ---
        st.divider()
        st.header("2. Refertazione Esame")
        
        # Layout modulo esame
        ex_col1, ex_col2 = st.columns([3, 1])
        
        with ex_col1:
            user_choice = st.selectbox("Classificazione Indicazione:", 
                                     ["Seleziona diagnosi...", "Nessun Difetto", "Cricca", "Porosità Singola", 
                                      "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"],
                                     disabled=(st.session_state['exam_state'] == "REVIEWED"))
        
        with ex_col2:
            st.write("") # Spacer
            st.write("") 
            # Pulsante Conferma
            if st.session_state['exam_state'] == "ACQUIRED":
                if st.button("CONFERMA DIAGNOSI"):
                    if user_choice == "Seleziona diagnosi...":
                        st.warning("Devi selezionare un tipo di difetto!")
                    else:
                        st.session_state['user_answer'] = user_choice
                        st.session_state['exam_state'] = "REVIEWED"
                        st.rerun()
            else:
                if st.button("NUOVO ESAME"):
                    st.session_state['exam_state'] = "SETUP"
                    st.session_state['scan_data'] = None
                    st.rerun()

        # --- REPORT FINALE (Appare solo dopo la conferma) ---
        if st.session_state['exam_state'] == "REVIEWED":
            real_defect = st.session_state['ground_truth']['type']
            user_ans = st.session_state['user_answer']
            
            st.markdown("### 📋 Risultato Verifica")
            
            if user_ans == real_defect:
                st.success(f"**ESITO: SUPERATO** ✅\n\nHai identificato correttamente: **{real_defect}**.")
            else:
                st.error(f"**ESITO: NON SUPERATO** ❌\n\nLa tua diagnosi: **{user_ans}**\nRealtà: **{real_defect}**")
                
            st.info("💡 **Feedback Istruttore:** Il riquadro rosso sull'immagine mostra l'esatta posizione e dimensione del difetto simulato. Verifica se la tua indicazione coincideva con l'area evidenziata.")