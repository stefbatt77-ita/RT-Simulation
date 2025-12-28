def check_password():
    """Restituisce True se l'utente ha inserito la password corretta."""
    def password_entered():
        if st.session_state["password"] == "Aero2025!": # Modifica la tua password qui
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password di Accesso NDT Suite", type="password", 
                      on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password Errata. Riprova:", type="password", 
                      on_change=password_entered, key="password")
        st.error("Accesso Negato")
        return False
    return True

if not check_password():
    st.stop()  # Ferma l'esecuzione se la password è sbagliata
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace
import random
import datetime
import io

# Configurazione Pagina
st.set_page_config(page_title="Aero-NDT Web Sim", layout="wide", page_icon="☢️")

# --- STILE CSS PERSONALIZZATO (Dark Mode Professionale) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSlider [data-baseweb="slider"] { margin-bottom: 20px; }
    .stButton button { width: 100%; border-radius: 5px; height: 3em; background-color: #d32f2f; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNZIONI CORE (FISICA E GENERAZIONE) ---
def generate_scan(kv, ma, time, material, thickness, iqi_type):
    size = 800
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[material] * (120/kv)**1.5
    
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # Generazione Difetto
    defects = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"]
    chosen_defect = random.choice(defects)
    
    cx, cy = 400, 400
    if chosen_defect == "Cricca":
        for y in range(300, 500):
            m_sp[y, int(cx)] -= 0.6; cx += random.uniform(-0.6, 0.6)
    elif chosen_defect == "Porosità Singola":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-400)**2 + (y-400)**2 <= 5**2] -= 2.0
    elif chosen_defect == "Inclusione Tungsteno":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-400)**2 + (y-400)**2 <= 4**2] += 12.0
    elif chosen_defect == "Incisione Marginale":
        m_sp[200:600, 430:433] -= 1.2
    elif chosen_defect == "Mancata Fusione":
        m_sp[200:600, 398:401] -= 1.8

    # IQI & Duplex
    for i in range(13): m_sp[700:750, 50 + i*25 : 50 + i*25 + 2] += (0.8 / (i+1))
    if iqi_type == "ISO 19232-1 (Wires)":
        for i in range(6): m_sp[100:250, 50+i*30:52+i*30] += (0.4 - i*0.05)
    else:
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
    return raw, chosen_defect

# --- INTERFACCIA WEB ---
st.title("☢️ Aero-NDT Digital Imaging Suite")
st.caption("Conforme NAS 410 / EN4179 - Simulazione DDA 16-bit")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("⚙️ Acquisizione")
    kv = st.slider("Tensione (kV)", 40, 250, 110)
    ma = st.slider("Corrente (mA)", 0.5, 15.0, 5.0)
    time = st.slider("Esposizione (s)", 1, 120, 25)
    
    mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
    thick = st.number_input("Spessore (mm)", 1, 25, 10)
    iqi = st.radio("Tipo IQI", ["ISO 19232-1 (Wires)", "ASTM E1025 (Holes)"])
    
    if st.button("ACQUISISCI IMMAGINE"):
        raw, defect = generate_scan(kv, ma, time, mat, thick, iqi)
        st.session_state['raw_data'] = raw
        st.session_state['true_defect'] = defect
        st.session_state['meta'] = f"KV:{kv} MA:{ma} T:{time}s Mat:{mat}"

with col2:
    if 'raw_data' in st.session_state:
        st.header("🖥️ Monitor DDA")
        
        # Post-Processing integrato nel layout
        p1, p2, p3 = st.columns(3)
        with p1: level = st.slider("Livello (L)", 0, 65535, 32768)
        with p2: width = st.slider("Larghezza (W)", 100, 65535, 65535)
        with p3: sharpen = st.checkbox("Filtro Sharpening")

        # Rendering Immagine
        data = st.session_state['raw_data'].astype(float)
        if sharpen:
            data = data + 1.8 * laplace(data)
        
        vmin, vmax = max(0, level - width//2), min(65535, level + width//2)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        img_disp = ax.imshow(data, cmap='gray_r', vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        # Densitometro (valore medio centrale per il web)
        center_val = st.session_state['raw_data'][400, 400]
        st.metric("Grigio al Centro (16-bit)", center_val)
        
        st.pyplot(fig)
        
        # --- MODULO ESAME ---
        st.divider()
        st.subheader("📝 Training Exam Mode")
        user_choice = st.selectbox("Identifica il difetto:", 
                                  ["Scegli...", "Cricca", "Porosità Singola", "Cluster Porosità", 
                                   "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"])
        
        if st.button("Invia Risposta"):
            if user_choice == st.session_state['true_defect']:
                st.success(f"CORRETTO! È un/una {st.session_state['true_defect']}")
            else:
                st.error(f"ERRATO. Il difetto era: {st.session_state['true_defect']}")
                
        # Export (Simulazione DICONDE)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        st.download_button("Esporta DICONDE (PNG 16-bit)", buf.getvalue(), "aero_scan.png", "image/png")
    else:
        st.info("Configura i parametri e premi 'ACQUISISCI IMMAGINE' per iniziare.")