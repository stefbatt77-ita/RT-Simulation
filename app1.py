import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import random
import pandas as pd

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Aero-NDT Advanced v10.5", layout="wide", page_icon="☢️")

# --- INIZIALIZZAZIONE SICURA DELLO STATO ---
# Usiamo il metodo .get() o l'assegnazione esplicita per evitare AttributeError
if 'exam_progress' not in st.session_state:
    st.session_state.exam_progress = 0
if 'exam_results' not in st.session_state:
    st.session_state.exam_results = []
if 'exam_cases' not in st.session_state:
    st.session_state.exam_cases = []
if 's_eval' not in st.session_state:
    st.session_state.s_eval = False

# --- MOTORE DI GENERAZIONE ---
def generate_scan_v10_5(kv, ma, time, material, thickness, selected_defect=None):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[material] * (120/max(10, kv))**1.5
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # IQI & DUPLEX CON ETICHETTE
    for i in range(7): m_sp[40:140, 80 + i*40 : 80 + i*40 + 2] += (0.4 - i*0.05)
    for i in range(13): m_sp[500:540, 150 + i*25 : 150 + i*25 + 2] += (0.7 / (i+1))
    m_sp[145:160, 80:150] -= 0.5  # Label ISO
    m_sp[545:560, 150:250] -= 0.5 # Label DUPLEX

    all_bboxes = []
    defects_to_gen = []
    
    if selected_defect == "Casuale (Multiplo)":
        num = random.randint(1, 3)
        possible = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"]
        defects_to_gen = random.sample(possible, num)
    elif selected_defect == "Nessun Difetto":
        defects_to_gen = []
    else:
        defects_to_gen = [selected_defect]

    for d_type in defects_to_gen:
        rx, ry = random.randint(100, 500), random.randint(180, 450)
        if d_type == "Cricca":
            angle = random.randint(0, 360)
            length = random.randint(60, 130)
            curr_x, curr_y = rx, ry
            for _ in range(length):
                rad = np.radians(angle)
                curr_x += np.cos(rad) + random.uniform(-0.4, 0.4)
                curr_y += np.sin(rad) + random.uniform(-0.4, 0.4)
                if 0 <= int(curr_y) < size and 0 <= int(curr_x) < size:
                    m_sp[int(curr_y), int(curr_x)] -= 0.9
            all_bboxes.append({"x": rx-25, "y": ry-25, "w": 50, "h": 50, "t": "Cricca"})
        elif d_type == "Porosità Singola":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 5**2] -= 2.5
            all_bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 20, "t": "Porosità"})
        elif d_type == "Cluster Porosità":
            for _ in range(12):
                cx, cy = rx + random.randint(-30, 30), ry + random.randint(-30, 30)
                y, x = np.ogrid[:size, :size]
                m_sp[(x-cx)**2 + (y-cy)**2 <= 3**2] -= 1.8
            all_bboxes.append({"x": rx-40, "y": ry-40, "w": 80, "h": 80, "t": "Cluster"})
        elif d_type == "Inclusione Tungsteno":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 4**2] += 15.0
            all_bboxes.append({"x": rx-8, "y": ry-8, "w": 16, "h": 16, "t": "Tungsteno"})
        elif d_type == "Mancata Fusione":
            h = random.randint(80, 200)
            m_sp[ry:ry+h, rx:rx+3] -= 1.6
            all_bboxes.append({"x": rx-5, "y": ry, "w": 15, "h": h, "t": "Mancata Fusione"})

    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    return np.clip(signal + noise, 0, 65535).astype(np.uint16), defects_to_gen, all_bboxes

def get_ideal_params(material, thickness):
    base_kv = {"Al-2024 (Avional)": 45, "Ti-6Al-4V": 65, "Inconel 718": 95, "Steel 17-4 PH": 85}
    k_kv = {"Al-2024 (Avional)": 3.2, "Ti-6Al-4V": 4.8, "Inconel 718": 7.8, "Steel 17-4 PH": 6.5}
    id_kv = int(base_kv[material] + (thickness * k_kv[material]))
    id_mas = round(12 * np.exp(thickness * 0.06), 1)
    return id_kv, id_mas

def grade_param(u, i):
    p = (abs(u - i) / i) * 100
    if p <= 10: return "OTTIMO"
    if p <= 30: return "BUONO"
    if p <= 50: return "SUFFICIENTE"
    return "INSUFFICIENTE"

# --- INTERFACCIA ---
mode = st.sidebar.radio("Scegli Modalità", ["STUDIO", "ESAME"])

if mode == "STUDIO":
    st.title("📘 Ambiente di Studio")
    c1, c2 = st.columns([1, 2])
    with c1:
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 5, 30, 10)
        def_choice = st.selectbox("Difetto da simulare", ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione", "Casuale (Multiplo)", "Nessun Difetto"])
        st.divider()
        kv = st.slider("kV", 40, 250, 90)
        ma = st.slider("mA", 1.0, 15.0, 5.0)
        ti = st.slider("Tempo (s)", 1, 120, 25)
        if st.button("GENERA RADIOGRAFIA"):
            img, defs, bboxes = generate_scan_v10_5(kv, ma, ti, mat, thick, def_choice)
            st.session_state.s_img = img
            st.session_state.s_bboxes = bboxes
            st.session_state.s_defs = defs
            st.session_state.s_eval = False

    with c2:
        if 's_img' in st.session_state:
            l, w = st.slider("Level", 0, 65535, 32768), st.slider("Width", 100, 65535, 45000)
            fig, ax = plt.subplots(figsize=(8,8), facecolor='black')
            ax.imshow(st.session_state.s_img, cmap='gray_r', vmin=l-w//2, vmax=l+w//2)
            if st.session_state.s_eval:
                for b in st.session_state.s_bboxes:
                    ax.add_patch(patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=1.5, edgecolor='red', facecolor='none'))
                    ax.text(b['x'], b['y']-5, b['t'], color='red', fontsize=9, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            if st.button("VERIFICA E MOSTRA PARAMETRI IDEALI"):
                st.session_state.s_eval = True
                st.rerun()
            
            if st.session_state.s_eval:
                id_k, id_m = get_ideal_params(mat, thick)
                st.subheader("📊 Analisi Tecnica")
                res_df = pd.DataFrame({
                    "Parametro": ["Tensione (kV)", "Esposizione (mAs)"],
                    "Valore Utente": [f"{kv}", f"{ma*ti:.1f}"],
                    "Valore Ideale": [f"{id_k}", f"{id_m}"],
                    "Valutazione": [grade_param(kv, id_k), grade_param(ma*ti, id_m)]
                })
                st.table(res_df)
                st.info(f"**Difetti Presenti:** {', '.join(st.session_state.s_defs) if st.session_state.s_defs else 'Pezzo Sano'}")

elif mode == "ESAME":
    st.title("🎓 Esame di Certificazione")
    if st.session_state.exam_progress == 0:
        if st.button("INIZIA ESAME"):
            st.session_state.exam_cases = [{"mat": random.choice(["Ti-6Al-4V", "Steel 17-4 PH", "Inconel 718"]), "thick": random.randint(8, 25)} for _ in range(5)]
            st.session_state.exam_progress = 1
            st.session_state.exam_results = []
            st.rerun()
    elif st.session_state.exam_progress > 5:
        st.header("🏁 Report Finale")
        st.table(pd.DataFrame(st.session_state.exam_results))
        if st.button("RESET"):
            st.session_state.exam_progress = 0
            st.session_state.exam_results = []
            st.rerun()
    else:
        case = st.session_state.exam_cases[st.session_state.exam_progress-1]
        st.subheader(f"Caso {st.session_state.exam_progress} di 5")
        st.info(f"Specimen: **{case['mat']}**, **{case['thick']} mm**")