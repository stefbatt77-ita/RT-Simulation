import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, laplace
import random
import pandas as pd
import io

# --- CONFIGURAZIONE E CSS ---
st.set_page_config(page_title="Aero-NDT Certification Authority v10.1", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 5px; height: 3em; }
    .stTable { background-color: #1e1e1e; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTORE FISICO E CALCOLO IDEALE ---
def get_ideal_parameters(material, thickness):
    mat_props = {
        "Al-2024 (Avional)": {"base_kv": 45, "k_kv": 3.2, "mu_factor": 0.02},
        "Ti-6Al-4V":         {"base_kv": 65, "k_kv": 4.8, "mu_factor": 0.045},
        "Steel 17-4 PH":     {"base_kv": 85, "k_kv": 6.5, "mu_factor": 0.075},
        "Inconel 718":       {"base_kv": 95, "k_kv": 7.8, "mu_factor": 0.09}
    }
    props = mat_props[material]
    ideal_kv = int(props["base_kv"] + (thickness * props["k_kv"]))
    ideal_kv = max(40, min(250, ideal_kv))
    
    # Target mAs per grigio 32000
    ideal_mas = round(12 * np.exp(thickness * 0.06), 1)
    ideal_level = 32768
    ideal_width = 25000
    
    return ideal_kv, ideal_mas, ideal_level, ideal_width

def generate_scan_v10(kv, ma, time, material, thickness, chosen_defect=None):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    safe_kv = max(10, kv)
    mu = mu_map[material] * (120/safe_kv)**1.5
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    if not chosen_defect:
        defects = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"]
        chosen_defect = random.choice(defects)
    
    bbox = {"x": 300, "y": 300, "w": 50, "h": 50}
    if chosen_defect == "Cricca":
        curr_x = 300
        for y in range(200, 400):
            m_sp[y, int(curr_x)] -= 0.6
            curr_x += random.uniform(-0.6, 0.6)
        bbox = {"x": 280, "y": 200, "w": 40, "h": 200}
    elif chosen_defect == "Porosità Singola":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-300)**2 + (y-300)**2 <= 6**2] -= 2.5
        bbox = {"x": 290, "y": 290, "w": 20, "h": 20}
    elif chosen_defect == "Inclusione Tungsteno":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-300)**2 + (y-300)**2 <= 5**2] += 15.0
        bbox = {"x": 295, "y": 295, "w": 10, "h": 10}
    elif chosen_defect == "Nessun Difetto": bbox = None

    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    return np.clip(signal + noise, 0, 65535).astype(np.uint16), chosen_defect, bbox

def grade_parameter(user_val, ideal_val):
    delta = abs(user_val - ideal_val)
    perc = (delta / ideal_val) * 100
    if perc <= 10: return "OTTIMO"
    elif perc <= 30: return "BUONO"
    elif perc <= 50: return "SUFFICIENTE"
    else: return "INSUFFICIENTE"

# --- INIZIALIZZAZIONE SESSIONE ---
for key in ['mode', 'exam_progress', 'exam_results', 'current_case', 'study_evaluated']:
    if key not in st.session_state:
        if key == 'mode': st.session_state[key] = "STUDIO (Training)"
        elif key == 'exam_results': st.session_state[key] = []
        elif key == 'exam_progress': st.session_state[key] = 0
        else: st.session_state[key] = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Pannello NDT")
    new_mode = st.radio("Modalità Operativa", ["STUDIO (Training)", "ESAME (Certificazione)"])
    if new_mode != st.session_state['mode']:
        st.session_state['mode'] = new_mode
        st.session_state['exam_progress'] = 0
        st.session_state['exam_results'] = []
        st.session_state['current_case'] = None
        st.rerun()

# ==============================================================================
# SEZIONE 1: MODALITÀ STUDIO
# ==============================================================================
if st.session_state['mode'] == "STUDIO (Training)":
    st.title("📘 Ambiente di Studio")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 1, 30, 10)
        kv = st.slider("kV", 40, 250, 100)
        ma = st.slider("mA", 0.5, 15.0, 5.0)
        time = st.slider("Tempo (s)", 1, 120, 20)
        
        if st.button("ACQUISICI SCANSIONE"):
            raw, defect, bbox = generate_scan_v10(kv, ma, time, mat, thick)
            st.session_state['study_data'] = raw
            st.session_state['study_truth'] = {'defect': defect, 'bbox': bbox, 'params': (mat, thick, kv, ma, time)}
            st.session_state['study_evaluated'] = False

    with col2:
        if 'study_data' in st.session_state:
            lev = st.slider("Level", 0, 65535, 32768)
            wid = st.slider("Width", 100, 65535, 60000)
            
            fig, ax = plt.subplots(facecolor='black')
            ax.imshow(st.session_state['study_data'], cmap='gray_r', vmin=lev-wid//2, vmax=lev+wid//2)
            if st.session_state.get('study_evaluated'):
                b = st.session_state['study_truth']['bbox']
                if b: ax.add_patch(patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
            ax.axis('off')
            st.pyplot(fig)
            
            ans = st.selectbox("Diagnosi:", ["Scegli...", "Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"])
            if st.button("VERIFICA E MOSTRA PARAMETRI IDEALI"):
                st.session_state['study_evaluated'] = True
                st.rerun()

            if st.session_state.get('study_evaluated'):
                real_def = st.session_state['study_truth']['defect']
                mat_s, th_s, kv_s, ma_s, ti_s = st.session_state['study_truth']['params']
                id_kv, id_mas, id_lev, id_wid = get_ideal_parameters(mat_s, th_s)
                
                st.subheader("📊 Risultati e Parametri Ideali")
                st.write(f"**Diagnosi Corretta:** {real_def}")
                
                comparison = pd.DataFrame({
                    "Parametro": ["Tensione (kV)", "Esposizione (mAs)", "Level", "Width"],
                    "Tuo Valore": [f"{kv_s}", f"{ma_s*ti_s:.1f}", f"{lev}", f"{wid}"],
                    "Ideale": [f"{id_kv}", f"{id_mas}", f"{id_lev}", f"{id_wid}"],
                    "Valutazione": [grade_parameter(kv_s, id_kv), grade_parameter(ma_s*ti_s, id_mas), "-", "-"]
                })
                st.table(comparison)

# ==============================================================================
# SEZIONE 2: MODALITÀ ESAME
# ==============================================================================
elif st.session_state['mode'] == "ESAME (Certificazione)":
    st.title("🎓 Sessione d'Esame")
    
    if st.session_state['exam_progress'] == 0:
        if st.button("INIZIA ESAME (5 Casi)"):
            st.session_state['exam_cases'] = [{"mat": random.choice(["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718"]), "thick": random.randint(5, 25), "defect": random.choice(["Cricca", "Porosità Singola", "Inclusione Tungsteno"])} for _ in range(5)]
            st.session_state['exam_progress'] = 1
            st.session_state['current_case'] = st.session_state['exam_cases'][0]
            st.rerun()

    elif st.session_state['exam_progress'] > 5:
        st.header("🏁 Report Finale Certificazione")
        df_res = pd.DataFrame(st.session_state['exam_results'])
        if not df_res.empty:
            st.table(df_res)
            # Verifica sicura della colonna Diagnosi
            if "Diagnosi" in df_res.columns:
                punteggio = df_res[df_res["Diagnosi"] == "CORRETTO"].shape[0]
                st.metric("Difetti Identificati", f"{punteggio} / 5")
        if st.button("Ricomincia Esame"):
            st.session_state['exam_progress'] = 0
            st.session_state['exam_results'] = []
            st.rerun()

    else:
        idx = st.session_state['exam_progress']
        case = st.session_state['current_case']
        st.subheader(f"Caso {idx} di 5")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info(f"**Specimen:** {case['mat']} - {case['thick']}mm")
            kv_e = st.slider("kV", 40, 250, 100, key=f"k{idx}")
            ma_e = st.slider("mA", 0.5, 15.0, 5.0, key=f"m{idx}")
            ti_e = st.slider("Tempo (s)", 1, 120, 20, key=f"t{idx}")
            if st.button("SCATTA"):
                img, d_t, bb = generate_scan_v10(kv_e, ma_e, ti_e, case['mat'], case['thick'], case['defect'])
                st.session_state['e_img'] = img
        
        with c2:
            if 'e_img' in st.session_state:
                l_e = st.slider("L", 0, 65535, 32768, key=f"l{idx}")
                w_e = st.slider("W", 100, 65535, 60000, key=f"w{idx}")
                fig, ax = plt.subplots(facecolor='black')
                ax.imshow(st.session_state['e_img'], cmap='gray_r', vmin=l_e-w_e//2, vmax=l_e+w_e//2)
                ax.axis('off')
                st.pyplot(fig)
                
                diag = st.selectbox("Identifica:", ["Scegli...", "Cricca", "Porosità Singola", "Inclusione Tungsteno", "Nessun Difetto"], key=f"d{idx}")
                if st.button("SALVA E PROSSIMO"):
                    id_kv, id_mas, _, _ = get_ideal_parameters(case['mat'], case['thick'])
                    res = {
                        "Caso": idx,
                        "Materiale": f"{case['mat']} ({case['thick']}mm)",
                        "Voto kV": grade_parameter(kv_e, id_kv),
                        "Voto mAs": grade_parameter(ma_e*ti_e, id_mas),
                        "Diagnosi": "CORRETTO" if diag == case['defect'] else "ERRATO"
                    }
                    st.session_state['exam_results'].append(res)
                    st.session_state['exam_progress'] += 1
                    if st.session_state['exam_progress'] <= 5:
                        st.session_state['current_case'] = st.session_state['exam_cases'][st.session_state['exam_progress']-1]
                        if 'e_img' in st.session_state: del st.session_state['e_img']
                    st.rerun()