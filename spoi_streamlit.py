"""
================================================================================
SPOI WAREHOUSE OPTIMIZATION - STREAMLIT APP
================================================================================
Streamlit aplikacija koja koristi warehouse_model.py

Potrebni fajlovi u istom folderu:
- spoi_streamlit.py (ovaj fajl)
- warehouse_model.py (model)
- SPOI_DATA (1) new.xlsx (podaci)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Import modela
import warehouse_model as model

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SPOI Warehouse Optimization",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">🏭 SPOI Warehouse Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Optimizacija rasporeda artikala korištenjem ILP solvera</p>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Parametri")
    
    COST_A = st.slider("α - Horizontalna (H)", 0.5, 5.0, 2.0, 0.5)
    COST_B = st.slider("β - Vertikalna (V>2)", 1.0, 10.0, 5.0, 0.5)
    COST_C = st.slider("γ - Dubina (E)", 0.5, 5.0, 2.0, 0.5)
    
    st.markdown("---")
    
    TAU = st.slider("τ - Temperatura", 2.0, 20.0, 8.0, 1.0)
    DEMAND_MULTIPLIER = st.slider("λ - Multiplikator", 5, 30, 15, 1)
    N_PICKS = st.number_input("Pickova", 100, 2000, 500, 100)
    
    st.markdown("---")
    st.latex(r"C = \alpha H + \beta \max(0,V-2) + \gamma(4-E)")
    st.latex(r"U = e^{-C/\tau} \cdot (1 + \lambda \frac{d}{d_{max}})")

# Parametri dict
params = {
    'COST_A': COST_A, 'COST_B': COST_B, 'COST_C': COST_C,
    'TAU': TAU, 'DEMAND_MULTIPLIER': DEMAND_MULTIPLIER, 'N_PICKS': N_PICKS
}

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📂 Učitaj & Optimiziraj", "📊 Rezultati", "📈 Grafici"])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.header("📂 Učitavanje Podataka")
    
    FILE_PATH = "SPOI_DATA (1) new.xlsx"
    
    st.info(f"📁 Tražim fajl: **{FILE_PATH}**")
    st.warning("⚠️ Stavi fajl u isti folder kao aplikaciju!")
    
    if st.button("📂 UČITAJ PODATKE", use_container_width=True):
        try:
            df = pd.read_excel(FILE_PATH)
            
            # Validacija
            missing = [c for c in ['H', 'V', 'E', 'izlaz'] if c not in df.columns]
            if missing:
                st.error(f"❌ Nedostaju kolone: {missing}")
                st.stop()
            
            # Pripremi podatke koristeći model
            df = model.prepare_data(df)
            n = len(df)
            
            st.success(f"✅ Učitano **{n}** artikala")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("H", f"{int(df['H'].min())} - {int(df['H'].max())}")
            col2.metric("V", f"{int(df['V'].min())} - {int(df['V'].max())}")
            col3.metric("E", f"{int(df['E'].min())} - {int(df['E'].max())}")
            col4.metric("Izlaz", f"{int(df['izlaz'].min())} - {int(df['izlaz'].max())}")
            
            st.session_state['df_raw'] = df
            st.session_state['data_loaded'] = True
            
            with st.expander("👀 Pregled"):
                st.dataframe(df.head(15), use_container_width=True)
                
        except FileNotFoundError:
            st.error(f"❌ **{FILE_PATH}** nije pronađen!")
        except Exception as e:
            st.error(f"❌ Greška: {e}")
    
    st.markdown("---")
    st.header("🚀 Optimizacija")
    
    if st.button("🎯 OPTIMIZIRAJ", type="primary", use_container_width=True):
        if 'data_loaded' not in st.session_state:
            st.warning("⚠️ Prvo učitaj podatke!")
            st.stop()
        
        progress = st.progress(0)
        status = st.empty()
        
        status.text("⏳ Pokrećem optimizaciju...")
        progress.progress(20)
        
        # Pozovi model
        try:
            results = model.optimize(st.session_state['df_raw'], params)
            progress.progress(90)
            
            # Spremi rezultate
            st.session_state['results'] = results
            st.session_state['optimized'] = True
            
            progress.progress(100)
            status.text("✅ Završeno!")
            
            st.balloons()
            st.success(f"🎯 Poboljšanje: **+{results['improvement']:.2f}%**")
            st.info("👉 Idi na tabove **Rezultati** i **Grafici**")
            
        except Exception as e:
            st.error(f"❌ Greška: {e}")

# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.header("📊 Rezultati")
    
    if 'optimized' not in st.session_state:
        st.warning("⚠️ Prvo pokreni optimizaciju!")
    else:
        r = st.session_state['results']
        
        # Metrike
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Utility", f"{r['opt_utils'].sum():.2f}", f"+{r['improvement']:.1f}%")
        col2.metric("Sim. Cost", f"{r['opt_sim']:,.0f}", f"-{r['cost_reduction']:.1f}%", delta_color="inverse")
        col3.metric("Weighted H", f"{r['opt_wH']:.2f}", f"-{r['h_reduction']:.1f}%", delta_color="inverse")
        col4.metric("Weighted V", f"{r['opt_wV']:.2f}", f"-{r['v_reduction']:.1f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # Tabela
        st.subheader("📋 Usporedba")
        comp = pd.DataFrame({
            'Metrika': ['Total Utility', 'Simulation Cost', 'Weighted H', 'Weighted V', 'Premješteno'],
            'Početno': [f"{r['init_utils'].sum():.2f}", f"{r['init_sim']:,.0f}", 
                       f"{r['init_wH']:.2f}", f"{r['init_wV']:.2f}", "-"],
            'Optimizirano': [f"{r['opt_utils'].sum():.2f}", f"{r['opt_sim']:,.0f}",
                            f"{r['opt_wH']:.2f}", f"{r['opt_wV']:.2f}", f"{r['moved']}"],
            'Promjena': [f"+{r['improvement']:.2f}%", f"-{r['cost_reduction']:.2f}%",
                        f"-{r['h_reduction']:.2f}%", f"-{r['v_reduction']:.2f}%", 
                        f"{r['moved']/r['n_items']*100:.1f}%"]
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Top 10
        st.subheader("📦 Top 10 Promjena")
        df = r['df']
        df_pos = r['df_positions']
        opt = r['opt_assign']
        
        top10 = []
        for idx in df.nlargest(10, 'izlaz').index:
            j = opt[idx]
            naziv = df.iloc[idx].get('Naziv artikla', f'#{idx}')
            if pd.isna(naziv): naziv = f'#{idx}'
            top10.append({
                'Naziv': str(naziv)[:25],
                'Potražnja': int(df.iloc[idx]['izlaz']),
                'H': f"{int(df.iloc[idx]['H'])} → {int(df_pos.iloc[j]['H'])}",
                'V': f"{int(df.iloc[idx]['V'])} → {int(df_pos.iloc[j]['V'])}",
                'E': f"{int(df.iloc[idx]['E'])} → {int(df_pos.iloc[j]['E'])}"
            })
        st.dataframe(pd.DataFrame(top10), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Download
        st.subheader("💾 Download")
        
        output_df = model.create_output_dataframe(r)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='Optimized', index=False)
            comp.to_excel(writer, sheet_name='Results', index=False)
        buffer.seek(0)
        
        st.download_button(
            "📥 Download Excel",
            data=buffer,
            file_name=f"SPOI_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.header("📈 Grafici")
    
    if 'optimized' not in st.session_state:
        st.warning("⚠️ Prvo pokreni optimizaciju!")
    else:
        r = st.session_state['results']
        df = r['df']
        df_pos = r['df_positions']
        n = r['n_items']
        opt = r['opt_assign']
        
        C_I, C_O, C_A = '#E74C3C', '#27AE60', '#3498DB'
        izlaz = df['izlaz'].values
        sidx = np.argsort(izlaz)[::-1]
        xi = np.arange(n)
        
        # 1. Summary
        st.subheader("1️⃣ Summary")
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for a, y, t, c in [(ax[0,0], [r['init_utils'].sum(), r['opt_utils'].sum()], 'Total Utility', C_O),
                          (ax[0,1], [r['init_sim'], r['opt_sim']], 'Simulation Cost', C_I),
                          (ax[1,0], [r['init_wH'], r['opt_wH']], 'Weighted H', C_A),
                          (ax[1,1], [r['init_wV'], r['opt_wV']], 'Weighted V', '#9B59B6')]:
            a.plot(['Početno', 'Optim.'], y, 'o-', lw=3, ms=12, c=c)
            a.fill_between(['Početno', 'Optim.'], 0, y, alpha=0.3, color=c)
            a.set_title(t, fontweight='bold')
            a.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # 2. Utility
        st.subheader("2️⃣ Utility po Artiklima")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(xi, r['init_utils'][sidx], c=C_I, label='Početno')
        ax.plot(xi, r['opt_utils'][sidx], c=C_O, label='Optimizirano')
        ax.fill_between(xi, r['init_utils'][sidx], r['opt_utils'][sidx],
                       where=r['opt_utils'][sidx] > r['init_utils'][sidx], alpha=0.3, color=C_O)
        ax.set_xlabel('Artikli (po potražnji)')
        ax.set_ylabel('Utility')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # 3 & 4. H i V
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("3️⃣ H po Artiklima")
            fig, ax = plt.subplots(figsize=(7, 4))
            h_i = np.array([df.iloc[i]['H'] for i in sidx])
            h_o = np.array([df_pos.iloc[opt[i]]['H'] for i in sidx])
            ax.plot(xi, h_i, c=C_I, label='Početno')
            ax.plot(xi, h_o, c=C_O, label='Optimizirano')
            ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 16)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("4️⃣ V po Artiklima")
            fig, ax = plt.subplots(figsize=(7, 4))
            v_i = np.array([df.iloc[i]['V'] for i in sidx])
            v_o = np.array([df_pos.iloc[opt[i]]['V'] for i in sidx])
            ax.plot(xi, v_i, c=C_I, label='Početno')
            ax.plot(xi, v_o, c=C_O, label='Optimizirano')
            ax.axhline(2, c='gold', ls='--', lw=2, label='Zlatna zona')
            ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 5. Heatmap
        st.subheader("5️⃣ Heatmap Potražnje")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        mh, mv = int(df['H'].max()), 5
        hm_b, hm_a = np.zeros((mv, mh)), np.zeros((mv, mh))
        for i in range(n):
            hm_b[int(df.iloc[i]['V'])-1, int(df.iloc[i]['H'])-1] += izlaz[i]
            j = opt[i]
            hm_a[int(df_pos.iloc[j]['V'])-1, int(df_pos.iloc[j]['H'])-1] += izlaz[i]
        axes[0].imshow(hm_b, cmap='Reds', aspect='auto')
        axes[0].set_title('POČETNO', fontweight='bold')
        axes[1].imshow(hm_a, cmap='Greens', aspect='auto')
        axes[1].set_title('OPTIMIZIRANO', fontweight='bold')
        for a in axes: a.set_xlabel('H'); a.set_ylabel('V')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # 6. Cumulative
        st.subheader("6️⃣ Kumulativni Utility")
        fig, ax = plt.subplots(figsize=(14, 5))
        cum_i = np.cumsum(r['init_utils'][sidx])
        cum_o = np.cumsum(r['opt_utils'][sidx])
        ax.plot(xi, cum_i, c=C_I, label=f'Početno ({cum_i[-1]:.1f})')
        ax.plot(xi, cum_o, c=C_O, label=f'Optim. ({cum_o[-1]:.1f})')
        ax.fill_between(xi, cum_i, cum_o, alpha=0.3, color=C_O)
        ax.set_xlabel('Artikli'); ax.set_ylabel('Kumulativni Utility')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6B7280'>🏭 SPOI Warehouse Optimization</div>", unsafe_allow_html=True)
