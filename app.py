import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, laplace
import random
import io

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Aero-NDT Master Suite v8.0", layout="wide", page_icon="☢️")

# CSS per Dark Mode Professionale
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    /* Colore bottoni specifici */
    div.stButton > button:first-child { background-color: #d32f2f; color: white; } /* Acquisisci */
    </style>
    """, unsafe_allow_html=True)

# --- INIZIALIZZAZIONE STATO ---
if 'show_boundary' not in st.session_state:
    st.session_state['show_boundary'] = False
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None

# --- MOTORE FISICO E GENERATORE ---
def generate_scan(kv, ma, time, material, thickness, iqi_type):
    size = 800
    # Fisica Attenuazione (Coefficienti approssimati per simulazione)
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[material] * (120/kv)**1.5
    
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # Selezione Random Difetto
    defects = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"]
    chosen_defect = random.choice(defects)
    
    # Coordinate Difetto per evidenziazione (x, y, larghezza, altezza)
    # Default (centro)
    bbox = {"x": 400, "y": 400, "w": 50, "h": 50}
    
    cx, cy = 400, 400
    
    if chosen_defect == "Cricca":
        # Generazione random walk verticale
        points_x = []
        points_y = []
        curr_x = cx
        for y in range(300, 500):
            m_sp[y, int(curr_x)] -= 0.6
            points_x.append(curr_x)
            points_y.append(y)
            curr_x += random.uniform(-0.6, 0.6)
        # Calcolo Bounding Box dinamico
        min_x, max_x = min(points_x), max(points_x)
        bbox = {"x": min_x, "y": 300, "w": (max_x - min_x) + 10, "h": 200}

    elif chosen_defect == "Porosità Singola":
        y, x = np.ogrid[:size, :size]
        r = 6
        mask = (x-400)**2 + (y-400)**2 <= r**2
        m_sp[mask] -= 2.0
        bbox = {"x": 390, "y": 390, "w": 20, "h": 20}

    elif chosen_defect == "Cluster Porosità":
        min_x, max_x, min_y, max_y = 800, 0, 800, 0
        for _ in range(10):
            rx, ry = random.randint(360, 440), random.randint(360, 440)
            r = 3
            y, x = np.ogrid[:size, :size]
            mask = (x-rx)**2 + (y-ry)**2 <= r**2
            m_sp[mask] -= 1.5
            # Aggiorna estensione box
            min_x, max_x = min(min_x, rx), max(max_x, rx)
            min_y, max_y = min(min_y, ry), max(max_y, ry)
        bbox = {"x": min_x-5, "y": min_y-5, "w": (max_x-min_x)+10, "h": (max_y-min_y)+10}

    elif chosen_defect == "Inclusione Tungsteno":
        y, x = np.ogrid[:size, :size]
        mask = (x-400)**2 + (y-400)**2 <= 4**2
        m_sp[mask] += 15.0 # Alta densità
        bbox = {"x": 392, "y": 392, "w": 16, "h": 16}

    elif chosen_defect == "Incisione Marginale":
        m_sp[200:600, 430:433] -= 1.2
        bbox = {"x": 425, "y": 200, "w": 15, "h": 400}

    elif chosen_defect == "Mancata Fusione":
        m_sp[200:600, 398:401] -= 1.8
        bbox = {"x": 395, "y": 200, "w": 10, "h": 400}

    # Inserimento IQI Duplex (Sempre presente per EN4179)
    for i in range(13): 
        m_sp[700:750, 50 + i*25 : 50 + i*25 + 2] += (0.8 / (i+1))
    
    # Inserimento IQI Selezionato
    if iqi_type == "ISO 19232-1 (Wires)":
        for i in range(7): m_sp[100:250, 50+i*30:52+i*30] += (0.4 - i*0.05)
    else: # ASTM Holes
        m_sp[100:150, 50:180] += 0.2 # Placca
        for i, r in enumerate([2, 4, 8]): # Fori 1T, 2T, 4T
            y, x = np.ogrid[:size, :size]
            m_sp[(x-(75+i*35))**2 + (y-125)**2 <= r**2] -= 0.2

    # Calcolo Intensità Radiazione (16-bit)
    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    
    # Sfocatura e Rumore
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    
    raw = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    return raw, chosen_defect, bbox

# --- LAYOUT INTERFACCIA ---
st.title("☢️ Aero-NDT Master Suite v8.0")
st.markdown("**Simulatore Radiografico NAS 410 / EN4179 - Digital Detector Array 16-bit**")

col_ctrl, col_view = st.columns([1, 3])

# === SIDEBAR (CONTROLLI) ===
with col_ctrl:
    st.header("⚙️ Setup Generatore")
    kv = st.slider("Tensione (kV)", 40, 250, 110)
    ma = st.slider("Corrente (mA)", 0.5, 15.0, 5.0)
    time = st.slider("Tempo (s)", 1, 120, 25)
    
    st.divider()
    st.subheader("🛠️ Pezzo & IQI")
    mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
    thick = st.number_input("Spessore (mm)", 1, 30, 10)
    iqi = st.radio("Tipo Penetrametro", ["ISO 19232-1 (Wires)", "ASTM E1025 (Holes)"])
    
    st.divider()
    # TASTO ACQUISIZIONE
    if st.button("ACQUISICI NUOVA SCANSIONE", help="Genera una nuova immagine con un difetto casuale"):
        raw, defect, bbox = generate_scan(kv, ma, time, mat, thick, iqi)
        st.session_state['raw_data'] = raw
        st.session_state['true_defect'] = defect
        st.session_state['bbox'] = bbox
        st.session_state['show_boundary'] = False # Nascondi soluzione
        st.session_state['scan_params'] = f"{mat} {thick}mm | {kv}kV {ma}mA"

    # DICONDE EXPORT
    if st.session_state['raw_data'] is not None:
        st.divider()
        buf = io.BytesIO()
        plt.imsave(buf, st.session_state['raw_data'], cmap='gray_r', format='png')
        st.download_button("💾 Esporta DICONDE (PNG 16-bit)", buf.getvalue(), "scan_aero.png", "image/png")

# === MAIN VIEW (VISUALIZZAZIONE) ===
with col_view:
    if st.session_state['raw_data'] is not None:
        # Toolbar Post-Processing
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: level = st.slider("Livello (Brightness)", 0, 65535, 32768)
        with c2: width = st.slider("Finestra (Contrast)", 100, 65535, 65535)
        with c3: sharpen = st.checkbox("Filtro Sharpening")

        # Elaborazione Immagine
        data = st.session_state['raw_data'].astype(float)
        if sharpen:
            data = data + 2.0 * laplace(data) # Filtro bordi
        
        # Window / Level logic
        vmin, vmax = max(0, level - width//2), min(65535, level + width//2)

        # Plotting con Matplotlib
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        # Mappa colori 'gray_r' -> Negativo (Bianco=Denso, Nero=Vuoto/Cricca)
        ax.imshow(data, cmap='gray_r', vmin=vmin, vmax=vmax)
        
        # LOGICA DISEGNO CONTORNO (SOLUZIONE)
        if st.session_state['show_boundary']:
            b = st.session_state['bbox']
            # Crea rettangolo rosso
            rect = patches.Rectangle((b['x'], b['y']), b['w'], b['h'], 
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(b['x'], b['y']-10, "DIFETTO", color='red', fontsize=12, fontweight='bold')

        ax.axis('off')
        st.pyplot(fig)
        
        # Densitometro e Info
        st.info(f"Parametri Scansione: {st.session_state['scan_params']}")
        
        # --- MODULO ESAME ---
        st.divider()
        st.subheader("📝 Modulo Esame / Valutazione")
        
        col_ex1, col_ex2 = st.columns([3, 1])
        with col_ex1:
            user_ans = st.selectbox("Identifica il difetto rilevato:", 
                                   ["Seleziona...", "Cricca", "Porosità Singola", "Cluster Porosità", 
                                    "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione", "Nessun Difetto"])
        
        with col_ex2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("VALUTA ESAME"):
                if user_ans == "Seleziona...":
                    st.warning("Seleziona un difetto prima di valutare.")
                else:
                    st.session_state['show_boundary'] = True # MOSTRA IL RIQUADRO
                    if user_ans == st.session_state['true_defect']:
                        st.success(f"✅ ESATTO! Difetto confermato: {st.session_state['true_defect']}")
                        st.balloons()
                    else:
                        st.error(f"❌ ERRATO. Il difetto era: {st.session_state['true_defect']}")
                    st.rerun() # Ricarica pagina per mostrare il rettangolo

    else:
        # Schermata iniziale
        st.markdown("""
        ### 👋 Benvenuto nel Simulatore Aero-NDT
        
        Questo software simula un sistema radiografico digitale (DDA) per l'addestramento secondo **NAS 410 / EN4179**.
        
        **Funzionalità incluse:**
        * Simulazione fisica raggi X a 16-bit.
        * Difettologia aeronautica reale (Cricche, Tungsteno, Lack of Fusion).
        * Strumenti IQI ISO (Fili) e ASTM (Fori/Placchette).
        * Penetrametro Duplex per verifica risoluzione spaziale (SRb).
        * **Training Mode:** Identifica il difetto e ricevi feedback visivo immediato.
        
        👈 **Configura i parametri a sinistra e clicca "ACQUISICI" per iniziare.**
        """)