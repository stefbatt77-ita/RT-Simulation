import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, laplace
import random
import pandas as pd

# --- CONFIGURAZIONE E CSS ---
st.set_page_config(page_title="Aero-NDT Certification Authority v10", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #e0e0e0; }
    div.stButton > button { font-weight: bold; border-radius: 5px; height: 3em; }
    .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTORE FISICO E CALCOLO IDEALE ---
def get_ideal_parameters(material, thickness):
    """
    Calcola i parametri radiografici teorici ideali per ottenere 
    un'immagine bilanciata (Grigio ~32000) con buon contrasto.
    """
    # Coefficienti empirici (K factor) per materiale
    # kV ideali = Base + (Thick * K)
    mat_props = {
        "Al-2024 (Avional)": {"base_kv": 40, "k_kv": 3.0, "mu_factor": 0.02},
        "Ti-6Al-4V":         {"base_kv": 60, "k_kv": 4.5, "mu_factor": 0.045},
        "Steel 17-4 PH":     {"base_kv": 80, "k_kv": 6.0, "mu_factor": 0.075},
        "Inconel 718":       {"base_kv": 90, "k_kv": 7.5, "mu_factor": 0.09}
    }
    
    props = mat_props[material]
    
    # 1. Calcolo kV Ideali (Regola del pollice per contrasto ottimale)
    ideal_kv = props["base_kv"] + (thickness * props["k_kv"])
    # Limiti macchina
    ideal_kv = max(40, min(250, ideal_kv))
    
    # 2. Calcolo mAs Ideali per avere densità media
    # Target Intensity al detector ~ 32000 (metà dinamica 16-bit)
    # I = I0 * e^(-mu * x)  -> Vogliamo I costante
    # I0 dipende da mAs e kV^2
    
    # Semplificazione per simulazione:
    # mAs ideali crescono esponenzialmente con lo spessore
    ideal_mas = 10 * np.exp(thickness * 0.05) 
    
    # Split mAs in mA e Tempo (cerchiamo mA alti per ridurre tempo, ma max 15)
    if ideal_mas > 150: # Se servono tanti mAs, abbassiamo mA e alziamo tempo
        ideal_ma = 5.0
        ideal_time = ideal_mas / ideal_ma
    else:
        ideal_ma = 5.0
        ideal_time = ideal_mas / ideal_ma

    # Window / Level ideali
    ideal_level = 32768
    ideal_width = 30000 # Un po' di contrasto ma non troppo
    
    return int(ideal_kv), round(ideal_ma, 1), int(ideal_time), ideal_level, ideal_width

def generate_scan_v10(kv, ma, time, material, thickness, chosen_defect=None):
    size = 600 # Ottimizzato per velocità
    
    # Fisica
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    # Se i kV sono troppo bassi, mu schizza alle stelle (immagine nera)
    safe_kv = max(10, kv)
    mu = mu_map[material] * (120/safe_kv)**1.5
    
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # Difetto (Se non passato, ne sceglie uno a caso)
    if not chosen_defect:
        defects = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"]
        chosen_defect = random.choice(defects)
    
    # Logica generazione difetto e BBOX (semplificata per stabilità)
    bbox = {"x": 300, "y": 300, "w": 50, "h": 50}
    
    # Applicazione difetto alla matrice spessore
    cx, cy = 300, 300
    if chosen_defect == "Cricca":
        curr_x = cx
        points_x = []
        for y in range(200, 400):
            m_sp[y, int(curr_x)] -= 0.6
            points_x.append(curr_x); curr_x += random.uniform(-0.6, 0.6)
        bbox = {"x": min(points_x)-10, "y": 200, "w": max(points_x)-min(points_x)+20, "h": 200}
    elif chosen_defect == "Porosità Singola":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-300)**2 + (y-300)**2 <= 6**2] -= 2.0
        bbox = {"x": 290, "y": 290, "w": 20, "h": 20}
    elif chosen_defect == "Cluster Porosità":
        for _ in range(8):
            rx, ry = random.randint(280, 320), random.randint(280, 320)
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 3**2] -= 1.5
        bbox = {"x": 270, "y": 270, "w": 60, "h": 60}
    elif chosen_defect == "Inclusione Tungsteno":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-300)**2 + (y-300)**2 <= 4**2] += 15.0
        bbox = {"x": 295, "y": 295, "w": 10, "h": 10}
    elif chosen_defect == "Incisione Marginale":
        m_sp[100:500, 330:333] -= 1.2
        bbox = {"x": 325, "y": 100, "w": 15, "h": 400}
    elif chosen_defect == "Mancata Fusione":
        m_sp[100:500, 298:301] -= 1.8
        bbox = {"x": 295, "y": 100, "w": 10, "h": 400}

    # IQI Duplex
    for i in range(13): m_sp[550:580, 50 + i*25 : 50 + i*25 + 2] += (0.8 / (i+1))

    # Engine 16-bit
    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    raw = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    
    return raw, chosen_defect, bbox

# --- FUNZIONE DI VALUTAZIONE ---
def grade_parameter(user_val, ideal_val):
    """Calcola il voto basato sulla deviazione percentuale."""
    if ideal_val == 0: return "N/A", "gray"
    delta = abs(user_val - ideal_val)
    perc = (delta / ideal_val) * 100
    
    if perc <= 10: return "OTTIMO", "green"
    elif perc <= 30: return "BUONO", "blue"
    elif perc <= 50: return "SUFFICIENTE", "orange"
    else: return "INSUFFICIENTE", "red"

# --- INIZIALIZZAZIONE SESSIONE ---
if 'mode' not in st.session_state: st.session_state['mode'] = "STUDIO"
if 'exam_progress' not in st.session_state: st.session_state['exam_progress'] = 0
if 'exam_score' not in st.session_state: st.session_state['exam_results'] = []
if 'current_case' not in st.session_state: st.session_state['current_case'] = None

# --- SIDEBAR NAVIGAZIONE ---
with st.sidebar:
    st.title("🎛️ Pannello Controllo")
    mode = st.radio("Modalità Operativa", ["STUDIO (Training)", "ESAME (Certificazione)"])
    
    if mode != st.session_state['mode']:
        st.session_state['mode'] = mode
        # Reset esame se si cambia modalità
        st.session_state['exam_progress'] = 0
        st.session_state['exam_results'] = []
        st.session_state['current_case'] = None
        st.rerun()

# ==============================================================================
# SEZIONE 1: MODALITÀ STUDIO
# ==============================================================================
if st.session_state['mode'] == "STUDIO (Training)":
    st.title("📘 Ambiente di Studio e Calibrazione")
    st.info("In questa modalità puoi sperimentare liberamente. Dopo la valutazione, il sistema ti mostrerà i parametri ideali.")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Parametri")
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 1, 30, 10)
        st.divider()
        kv = st.slider("kV", 40, 250, 100)
        ma = st.slider("mA", 0.5, 15.0, 5.0)
        time = st.slider("Tempo (s)", 1, 120, 20)
        
        if st.button("ACQUISISCI SCANSIONE"):
            raw, defect, bbox = generate_scan_v10(kv, ma, time, mat, thick)
            st.session_state['study_data'] = raw
            st.session_state['study_truth'] = {'defect': defect, 'bbox': bbox, 'params': (mat, thick, kv, ma, time)}
            st.session_state['study_evaluated'] = False
    
    with col2:
        if 'study_data' in st.session_state:
            # Toolbar
            c1, c2, c3 = st.columns([2,2,1])
            with c1: lev = st.slider("Level", 0, 65535, 32768)
            with c2: wid = st.slider("Width", 100, 65535, 60000)
            
            vmin, vmax = max(0, lev - wid//2), min(65535, lev + wid//2)
            fig, ax = plt.subplots(facecolor='black', figsize=(8,6))
            ax.imshow(st.session_state['study_data'], cmap='gray_r', vmin=vmin, vmax=vmax)
            
            # Mostra soluzione se valutato
            if st.session_state.get('study_evaluated'):
                b = st.session_state['study_truth']['bbox']
                rect = patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
            ax.axis('off')
            st.pyplot(fig)
            
            # Valutazione
            st.divider()
            ans = st.selectbox("Diagnosi:", ["Scegli...", "Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"])
            
            if st.button("VERIFICA E MOSTRA IDEALI"):
                st.session_state['study_evaluated'] = True
                real = st.session_state['study_truth']['defect']
                
                # Feedback Diagnosi
                if ans == real: st.success(f"Diagnosi Corretta: {real}")
                else: st.error(f"Diagnosi Errata. Era: {real}")
                
                # Calcolo Parametri Ideali
                mat_t, th_t, kv_u, ma_u, time_u = st.session_state['study_truth']['params']
                id_kv, id_ma, id_time, id_lev, id_wid = get_ideal_parameters(mat_t, th_t)
                id_mas = id_ma * id_time
                user_mas = ma_u * time_u
                
                # Tabella Comparativa
                st.subheader("📊 Analisi Tecnica")
                res_df = pd.DataFrame({
                    "Parametro": ["Tensione (kV)", "Esposizione (mAs)", "Level", "Width"],
                    "Tuo Set": [f"{kv_u}", f"{user_mas:.1f}", f"{lev}", f"{wid}"],
                    "Ideale (Calc)": [f"{id_kv}", f"{id_mas:.1f}", f"{id_lev}", f"{id_wid}"],
                    "Voto": [grade_parameter(kv_u, id_kv)[0], grade_parameter(user_mas, id_mas)[0], "N/A", "N/A"]
                })
                st.table(res_df)
                st.rerun()

# ==============================================================================
# SEZIONE 2: MODALITÀ ESAME (CERTIFICAZIONE)
# ==============================================================================
elif st.session_state['mode'] == "ESAME (Certificazione)":
    st.title("🎓 Sessione d'Esame Ufficiale")
    
    # SETUP INIZIALE DEI 5 CASI
    if st.session_state['exam_progress'] == 0 and st.session_state['current_case'] is None:
        st.markdown("""
        Benvenuto all'esame.
        - Dovrai analizzare **5 casi**.
        - Il materiale e lo spessore sono imposti dal sistema.
        - Devi scegliere kV, mA e Tempo corretti.
        - La valutazione include: **Correttezza Parametri** e **Identificazione Difetto**.
        """)
        if st.button("INIZIA ESAME"):
            # Genera 5 casi random
            cases = []
            mats = ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"]
            for _ in range(5):
                cases.append({
                    "mat": random.choice(mats),
                    "thick": random.randint(5, 25),
                    "defect": random.choice(["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno"])
                })
            st.session_state['exam_cases'] = cases
            st.session_state['exam_progress'] = 1
            st.session_state['current_case'] = cases[0]
            st.rerun()

    # REPORT FINALE
    elif st.session_state['exam_progress'] > 5:
        st.success("ESAME COMPLETATO!")
        st.subheader("Pagella Finale")
        
        results = st.session_state['exam_results']
        df_res = pd.DataFrame(results)
        st.dataframe(df_res)
        
        # Calcolo Media
        passed = df_res[df_res["Diagnosi"] == "OK"].shape[0]
        st.metric("Difetti Identificati", f"{passed}/5")
        
        if st.button("NUOVA SESSIONE"):
            st.session_state['exam_progress'] = 0
            st.session_state['exam_results'] = []
            st.session_state['current_case'] = None
            st.rerun()

    # SVOLGIMENTO CASO X
    else:
        case_idx = st.session_state['exam_progress']
        case = st.session_state['current_case']
        
        st.progress(case_idx / 5)
        st.subheader(f"Caso {case_idx}/5")
        
        # Area Operativa
        c_p, c_v = st.columns([1, 2])
        
        with c_p:
            st.info(f"📋 **SPECIMEN:**\n\n**Materiale:** {case['mat']}\n\n**Spessore:** {case['thick']} mm")
            st.write("Imposta parametri:")
            kv = st.slider("kV", 40, 250, 100, key="ex_kv")
            ma = st.slider("mA", 0.5, 15.0, 5.0, key="ex_ma")
            time = st.slider("Tempo (s)", 1, 100, 20, key="ex_t")
            
            if st.button("SCATTA RADIOGRAFIA"):
                raw, def_type, bbox = generate_scan_v10(kv, ma, time, case['mat'], case['thick'], case['defect'])
                st.session_state['exam_img'] = raw
                st.session_state['exam_bbox'] = bbox
        
        with c_v:
            if 'exam_img' in st.session_state:
                # Visualizzazione
                lev = st.slider("L", 0, 65535, 32768, key="ex_l")
                wid = st.slider("W", 100, 65535, 65535, key="ex_w")
                
                vmin, vmax = max(0, lev - wid//2), min(65535, lev + wid//2)
                fig, ax = plt.subplots(facecolor='black', figsize=(6,6))
                ax.imshow(st.session_state['exam_img'], cmap='gray_r', vmin=vmin, vmax=vmax)
                ax.axis('off')
                st.pyplot(fig)
                
                st.divider()
                ans = st.selectbox("Identifica Difetto:", ["Scegli...", "Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno"])
                
                if st.button("CONFERMA E PROCEDI"):
                    if ans == "Scegli...":
                        st.warning("Seleziona una diagnosi.")
                    else:
                        # CALCOLO VOTI
                        id_kv, id_ma, id_time, _, _ = get_ideal_parameters(case['mat'], case['thick'])
                        id_mas = id_ma * id_time
                        user_mas = ma * time
                        
                        grade_kv, _ = grade_parameter(kv, id_kv)
                        grade_mas, _ = grade_parameter(user_mas, id_mas)
                        diag_res = "OK" if ans == case['defect'] else "FAIL"
                        
                        # Salva risultati
                        st.session_state['exam_results'].append({
                            "Caso": case_idx,
                            "Materiale": case['mat'],
                            "kV Voto": grade_kv,
                            "mAs Voto": grade_mas,
                            "Diagnosi": diag_res,
                            "Difetto Reale": case['defect']
                        })
                        
                        # Passa al prossimo
                        st.session_state['exam_progress'] += 1
                        if st.session_state['exam_progress'] <= 5:
                            st.session_state['current_case'] = st.session_state['exam_cases'][st.session_state['exam_progress']-1]
                            del st.session_state['exam_img'] # Pulisci immagine vecchia
                        st.rerun()