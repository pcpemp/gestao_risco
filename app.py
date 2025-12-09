import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Gestão de Risco Global (BRL)", page_icon="🌍")

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 10px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: black;
    }
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-weight: 600;
        font-size: 13px;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700;
        font-size: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #262730;
        color: white;
    }
    div[data-testid="column"] { display: flex; align-items: center; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES ---

@st.cache_data
def get_selic_bcb():
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados/ultimos/1?formato=json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data[0]['valor'])
        return 10.50
    except: return 10.50

@st.cache_data
def get_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    tickers_to_fetch = list(set(tickers + ['BRL=X']))
    try:
        data = yf.download(tickers_to_fetch, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if 'Adj Close' in data: df = data['Adj Close']
        elif 'Close' in data: df = data['Close']
        else: return pd.DataFrame()
        if isinstance(df, pd.Series):
            df = df.to_frame()
            if len(tickers_to_fetch) == 1: df.columns = tickers_to_fetch
        if 'BRL=X' in df.columns:
            usd_brl = df['BRL=X'].fillna(method='ffill')
            for col in df.columns:
                if col == 'BRL=X': continue
                if not col.endswith('.SA') and col != '^BVSP':
                    df[col] = df[col] * usd_brl
            df = df.drop(columns=['BRL=X'])
        return df
    except Exception: return pd.DataFrame()

def apply_fee(returns_series, annual_fee_pct):
    """Aplica taxa de administração como provisão diária"""
    if annual_fee_pct == 0: return returns_series
    daily_factor = (1 - annual_fee_pct/100.0) ** (1/252)
    net_returns = (1 + returns_series) * daily_factor - 1
    return net_returns

def calculate_portfolio_series_with_rebalancing(weights_dict, df_returns, rf_daily, rebal_freq):
    """
    Calcula a série de retornos do portfólio considerando a frequência de rebalanceamento.
    rebal_freq: 'D' (Diário), 'M' (Mensal), 'Q' (Trimestral), '6M' (Semestral), 'Y' (Anual), 'None' (Hold)
    """
    df_calc = df_returns.copy()
    df_calc['CAIXA'] = rf_daily
    
    w_target = pd.Series(weights_dict)
    w_target['CAIXA'] = 1.0 - w_target.sum()
    w_target = w_target.reindex(df_calc.columns).fillna(0.0)
    
    if rebal_freq == 'D':
        return df_calc.dot(w_target)
    
    if rebal_freq == 'None':
        # Buy & Hold
        cum_returns = (1 + df_calc).cumprod()
        portfolio_idx = cum_returns.dot(w_target)
        portfolio_daily_ret = portfolio_idx.pct_change().fillna(0.0)
        portfolio_daily_ret.iloc[0] = df_calc.iloc[0].dot(w_target) 
        return portfolio_daily_ret

    # Rebalanceamento Periódico
    freq_map = {'M': 'ME', 'Q': 'QE', '6M': '6ME', 'Y': 'YE'}
    period_alias = freq_map.get(rebal_freq, 'ME')
    
    groups = df_calc.groupby(pd.Grouper(freq=period_alias))
    portfolio_rets = []
    
    for _, block in groups:
        if block.empty: continue
        cum_local = (1 + block).cumprod()
        val_local = cum_local.dot(w_target)
        ret_local = val_local.pct_change().fillna(0.0)
        ret_local.iloc[0] = block.iloc[0].dot(w_target)
        portfolio_rets.append(ret_local)
        
    if not portfolio_rets:
        return pd.Series(0.0, index=df_calc.index)
        
    return pd.concat(portfolio_rets)

def calculate_portfolio_metrics(weights_dict, returns_df, benchmark_returns, risk_free_rate, annual_fee=0.0, rebal_freq='D'):
    
    keys = ['ret_ann', 'vol_ann', 'var_95', 'var_99', 'cvar_95', 'cvar_99', 'beta', 'max_dd', 'sharpe', 'sortino']
    if returns_df.empty and not weights_dict:
        return {k: 0.0 for k in keys}

    rf_daily = (1 + risk_free_rate) ** (1/252) - 1
    common_tickers = [t for t in returns_df.columns if t in weights_dict]
    
    if not common_tickers:
        dates = returns_df.index if not returns_df.empty else pd.date_range(end=datetime.now(), periods=252)
        s_rf = pd.Series(rf_daily, index=dates)
        s_net = apply_fee(s_rf, annual_fee)
        ret_ann = (1 + s_net.mean())**252 - 1
        return {k: 0.0 if k != 'ret_ann' else ret_ann for k in keys}

    df_active = returns_df[common_tickers]
    weights_active = {k: v for k,v in weights_dict.items() if k in common_tickers}
    
    gross_daily_returns = calculate_portfolio_series_with_rebalancing(weights_active, df_active, rf_daily, rebal_freq)
    portfolio_daily_returns = apply_fee(gross_daily_returns, annual_fee)
    
    total_return_ann = (1 + portfolio_daily_returns.mean()) ** 252 - 1
    volatility_ann = portfolio_daily_returns.std() * np.sqrt(252)
    
    var_95 = np.percentile(portfolio_daily_returns, 5) * -1 
    var_99 = np.percentile(portfolio_daily_returns, 1) * -1
    
    tail_95 = portfolio_daily_returns[portfolio_daily_returns <= -var_95]
    tail_99 = portfolio_daily_returns[portfolio_daily_returns <= -var_99]
    cvar_95 = tail_95.mean() * -1 if not tail_95.empty else var_95
    cvar_99 = tail_99.mean() * -1 if not tail_99.empty else var_99

    wealth_index = (1 + portfolio_daily_returns).cumprod()
    peaks = wealth_index.cummax()
    drawdown = (wealth_index - peaks) / peaks
    max_dd = drawdown.min()

    beta = 0.0
    if not benchmark_returns.empty:
        if benchmark_returns.std() == 0:
            beta = 0.0
        else:
            combined = pd.concat([portfolio_daily_returns, benchmark_returns], axis=1).dropna()
            if not combined.empty:
                cov_matrix = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
                cov_pb = cov_matrix[0, 1]
                var_b = cov_matrix[1, 1]
                beta = cov_pb / var_b if var_b != 0 else 0.0

    sharpe = (total_return_ann - risk_free_rate) / volatility_ann if volatility_ann > 0 else 0.0
    
    neg_ret = portfolio_daily_returns[portfolio_daily_returns < 0]
    downside_std = neg_ret.std() * np.sqrt(252)
    sortino = (total_return_ann - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    return {
        'ret_ann': total_return_ann, 'vol_ann': volatility_ann,
        'var_95': var_95, 'var_99': var_99,
        'cvar_95': cvar_95, 'cvar_99': cvar_99,
        'max_dd': max_dd, 'beta': beta,
        'sharpe': sharpe, 'sortino': sortino,
        'series': portfolio_daily_returns
    }

# --- INICIALIZAÇÃO ---
if 'tickers' not in st.session_state: st.session_state.tickers = [] 
if 'weights_curr' not in st.session_state: st.session_state.weights_curr = {}
if 'weights_sim' not in st.session_state: st.session_state.weights_sim = {}
if 'rf_default' not in st.session_state: st.session_state.rf_default = get_selic_bcb()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Configurações")
    periodo = st.selectbox("Janela Histórica", ["Últimos 1 Ano", "Últimos 2 Anos", "Últimos 5 Anos", "Últimos 10 Anos"])
    days_map = {"Últimos 1 Ano": 365, "Últimos 2 Anos": 730, "Últimos 5 Anos": 1825, "Últimos 10 Anos": 3650}
    start_date = datetime.now() - timedelta(days=days_map[periodo])
    
    rf_input = st.number_input("Taxa Livre de Risco (Anual %)", value=st.session_state.rf_default, step=0.1) / 100.0
    
    bench_options = {"Ibovespa (B3)": "^BVSP", "S&P 500 (EUA - em Reais)": "^GSPC", "CDI (Taxa Livre de Risco)": "CDI"}
    selected_bench_label = st.selectbox("Benchmark Comparativo", list(bench_options.keys()))
    bench_ticker_value = bench_options[selected_bench_label]

    st.markdown("---")
    
    # --- CUSTOS E REBALANCEAMENTO ---
    with st.expander("Custos & Rebalanceamento", expanded=True):
        st.caption("Taxa de Administração (Anual %)")
        col_fee1, col_fee2 = st.columns(2)
        fee_curr = col_fee1.number_input("Tx. Atual", 0.0, 20.0, 0.0, step=0.1)
        fee_sim = col_fee2.number_input("Tx. Sim.", 0.0, 20.0, 0.0, step=0.1)
        
        st.caption("Frequência de Rebalanceamento")
        rebal_opts = {
            "Diário": "D",
            "Mensal": "M",
            "Trimestral": "Q",
            "Semestral": "6M",
            "Anual": "Y",
            "Nunca (Buy & Hold)": "None"
        }
        rebal_keys = list(rebal_opts.keys())
        # Define índice padrão para "Nunca" (5)
        default_rebal_idx = rebal_keys.index("Nunca (Buy & Hold)")
        
        col_reb1, col_reb2 = st.columns(2)
        lbl_reb_curr = col_reb1.selectbox("Rebal. Atual", rebal_keys, index=default_rebal_idx)
        lbl_reb_sim = col_reb2.selectbox("Rebal. Sim.", rebal_keys, index=default_rebal_idx)
        
        reb_curr_val = rebal_opts[lbl_reb_curr]
        reb_sim_val = rebal_opts[lbl_reb_sim]
    # ------------------------------------

    st.markdown("---")
    # CSV
    st.subheader("Gerenciar Dados (CSV)")
    uploaded_file = st.file_uploader("Carregar Arquivo (.csv)", type=["csv"], help="Ticker, Peso Atual, Peso Simulado")
    if uploaded_file is not None:
        try:
            try: uploaded_file.seek(0); df_import = pd.read_csv(uploaded_file, sep=';')
            except: uploaded_file.seek(0); df_import = pd.read_csv(uploaded_file, sep=',')
            df_import.columns = [str(c).strip().lower() for c in df_import.columns]
            col_map = {'ticker': 'ticker', 'peso atual': 'peso atual', 'peso simulado': 'peso simulado'}
            found_cols = {k: c for k, c in col_map.items() for col in df_import.columns if k in col}
            
            if len(found_cols) == 3:
                if st.button("🔄 Confirmar Importação"):
                    st.session_state.tickers = []; st.session_state.weights_curr = {}; st.session_state.weights_sim = {}
                    for _, row in df_import.iterrows():
                        tk = str(row[found_cols['ticker']]).strip().upper()
                        if "CAIXA" in tk or tk == "" or tk == "NAN": continue
                        try: w_curr = float(str(row[found_cols['peso atual']]).replace(',', '.').replace('%', ''))
                        except: w_curr = 0.0
                        try: w_sim = float(str(row[found_cols['peso simulado']]).replace(',', '.').replace('%', ''))
                        except: w_sim = 0.0
                        if tk not in st.session_state.tickers:
                            st.session_state.tickers.append(tk); st.session_state.weights_curr[tk]=w_curr; st.session_state.weights_sim[tk]=w_sim
                    st.rerun()
            else: st.error("Colunas inválidas.")
        except Exception as e: st.error(str(e))

    sum_curr = sum(st.session_state.weights_curr.values()); sum_sim = sum(st.session_state.weights_sim.values())
    export_data = [{"Ticker": t, "Peso Atual": st.session_state.weights_curr.get(t,0), "Peso Simulado": st.session_state.weights_sim.get(t,0)} for t in st.session_state.tickers]
    export_data.append({"Ticker": "CAIXA", "Peso Atual": round(100-sum_curr,2), "Peso Simulado": round(100-sum_sim,2)})
    st.download_button("📥 Baixar CSV", pd.DataFrame(export_data).to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'), "portfolio.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns([3, 1])
    with c1: new_ticker = st.text_input("Ticker", placeholder="ex: AAPL", label_visibility="collapsed").upper().strip()
    with c2: btn_add = st.button("➕")
    if (btn_add or new_ticker) and new_ticker and new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker); st.session_state.weights_curr[new_ticker] = 0.0; st.session_state.weights_sim[new_ticker] = 0.0; st.rerun()

    st.markdown("---")
    if st.session_state.tickers:
        st.subheader("Editar Pesos")
        with st.expander("Ferramentas", expanded=False):
            c_eq1, c_eq2 = st.columns(2)
            if c_eq1.button("⚖️ Igualar Atual"):
                if len(st.session_state.tickers)>0:
                    for t in st.session_state.tickers: st.session_state.weights_curr[t] = 100.0/len(st.session_state.tickers)
                    st.rerun()
            if c_eq2.button("⚖️ Igualar Sim."):
                if len(st.session_state.tickers)>0:
                    for t in st.session_state.tickers: st.session_state.weights_sim[t] = 100.0/len(st.session_state.tickers)
                    st.rerun()
            st.divider()
            c_cx1, c_cx2 = st.columns(2)
            with c_cx1:
                tc_c = st.number_input("Cx Meta (At.)", 0.0, 100.0, 0.0, key="tg_c")
                if st.button("Aplicar (A)"):
                    cs = sum(st.session_state.weights_curr.values())
                    if cs > 0:
                        f = (100.0 - tc_c) / cs
                        for t in st.session_state.tickers: st.session_state.weights_curr[t] *= f
                    st.rerun()
            with c_cx2:
                tc_s = st.number_input("Cx Meta (Si.)", 0.0, 100.0, 0.0, key="tg_s")
                if st.button("Aplicar (S)"):
                    cs = sum(st.session_state.weights_sim.values())
                    if cs > 0:
                        f = (100.0 - tc_s) / cs
                        for t in st.session_state.tickers: st.session_state.weights_sim[t] *= f
                    st.rerun()

        cols = st.columns([2.5, 2.5, 2.5, 1]); cols[0].markdown("**ATIVO**"); cols[1].markdown("**ATUAL**"); cols[2].markdown("**SIM.**")
        sum_c, sum_s = 0.0, 0.0; to_remove = []
        for t in st.session_state.tickers:
            r = st.columns([2.5, 2.5, 2.5, 1])
            with r[0]: st.markdown(f"<span style='font-size:0.9em'>{t}</span>", unsafe_allow_html=True)
            with r[1]: v = st.number_input(f"c_{t}", 0.0, 100.0, step=1.0, value=st.session_state.weights_curr.get(t,0.0), label_visibility="collapsed"); st.session_state.weights_curr[t]=v; sum_c+=v
            with r[2]: v = st.number_input(f"s_{t}", 0.0, 100.0, step=1.0, value=st.session_state.weights_sim.get(t,0.0), label_visibility="collapsed"); st.session_state.weights_sim[t]=v; sum_s+=v
            with r[3]: 
                if st.button("✕", key=f"del_{t}"): to_remove.append(t)
        if to_remove:
            for t in to_remove: st.session_state.tickers.remove(t); del st.session_state.weights_curr[t]; del st.session_state.weights_sim[t]
            st.rerun()
        st.markdown("---")
        cc = 100-sum_c; cs = 100-sum_s
        c = st.columns([3,3,3]); c[0].markdown("**CAIXA**")
        c[1].markdown(f"<span style='color:{'#4CAF50' if cc>=0 else '#F44336'};font-weight:bold'>{cc:.1f}%</span>", unsafe_allow_html=True)
        c[2].markdown(f"<span style='color:{'#4CAF50' if cs>=0 else '#F44336'};font-weight:bold'>{cs:.1f}%</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Limpar Tudo", type="primary", use_container_width=True):
        st.session_state.tickers=[]; st.session_state.weights_curr={}; st.session_state.weights_sim={}; st.rerun()

# ==========================================
# ÁREA PRINCIPAL
# ==========================================
tickers_dl = st.session_state.tickers.copy()
real_bench_ticker = None
if bench_ticker_value != "CDI":
    real_bench_ticker = bench_ticker_value
    if real_bench_ticker not in tickers_dl: tickers_dl.append(real_bench_ticker)

df_all = get_data(tickers_dl, start_date, datetime.now())

if not st.session_state.tickers:
    st.info("👈 Adicione ativos na barra lateral.")
elif df_all.empty:
    st.warning("Aguardando dados...")
else:
    bench_returns = pd.Series()
    if bench_ticker_value == "CDI":
        valid_cols = [c for c in df_all.columns if c in st.session_state.tickers]
        df_assets = df_all[valid_cols]
    else:
        if real_bench_ticker and real_bench_ticker in df_all.columns:
            bench_returns = df_all[real_bench_ticker].pct_change().dropna()
        valid_cols = [c for c in df_all.columns if c in st.session_state.tickers]
        df_assets = df_all[valid_cols]

    df_returns = df_assets.pct_change().dropna()

    if bench_ticker_value == "CDI" and not df_returns.empty:
        rf_daily = (1 + rf_input) ** (1/252) - 1
        bench_returns = pd.Series(rf_daily, index=df_returns.index)

    if not bench_returns.empty and not df_returns.empty:
        idx = df_returns.index.intersection(bench_returns.index)
        if not idx.empty: df_returns = df_returns.loc[idx]; bench_returns = bench_returns.loc[idx]

    fw_c = {k: v/100.0 for k, v in st.session_state.weights_curr.items()}
    fw_s = {k: v/100.0 for k, v in st.session_state.weights_sim.items()}

    mc = calculate_portfolio_metrics(fw_c, df_returns, bench_returns, rf_input, fee_curr, reb_curr_val)
    ms = calculate_portfolio_metrics(fw_s, df_returns, bench_returns, rf_input, fee_sim, reb_sim_val)

    col_L, col_R = st.columns([1, 3.5])

    with col_L:
        st.subheader("Métricas (em BRL)")
        def d_card(l1, v1c, v1s, l2, v2c, v2s, i1=True, i2=True):
            c1, c2 = st.columns(2)
            c1.metric(l1, f"{v1s:.2%}", f"{(v1s-v1c)*100:+.2f} p.p.", delta_color="inverse" if i1 else "normal")
            c2.metric(l2, f"{v2s:.2%}", f"{(v2s-v2c)*100:+.2f} p.p.", delta_color="inverse" if i2 else "normal")
        
        d_card("Volatilidade Anual", mc['vol_ann'], ms['vol_ann'], "Retorno Esp. (Anual)", mc['ret_ann'], ms['ret_ann'], True, False)
        
        cb, cd = st.columns(2)
        if bench_ticker_value == "CDI": cb.metric("Beta (vs CDI)", "0.00", "0.00", delta_color="off")
        else: cb.metric("Beta Atual", f"{ms['beta']:.2f}", f"{ms['beta']-mc['beta']:+.2f}", delta_color="inverse")
        cd.metric("Máximo Drawdown", f"{ms['max_dd']:.2%}", f"{(ms['max_dd']-mc['max_dd'])*100:+.2f} p.p.", delta_color="inverse")

        d_card("Sharpe Ratio", mc['sharpe'], ms['sharpe'], "Sortino Ratio", mc['sortino'], ms['sortino'], False, False)
        d_card("VaR 95% (1D)", mc['var_95'], ms['var_95'], "VaR 99% (1D)", mc['var_99'], ms['var_99'])
        d_card("CVaR 95%", mc['cvar_95'], ms['cvar_95'], "CVaR 99%", mc['cvar_99'], ms['cvar_99'])

    with col_R:
        t1, t2, t3, t4 = st.tabs(["Fronteira Eficiente", "Matriz de Correlação", "Drawdown Histórico", "Retorno Acumulado"])
        valid_cols = [c for c in df_returns.columns if c in st.session_state.tickers]
        
        with t1:
            if valid_cols:
                dfa = df_returns[valid_cols]
                mu_daily = dfa.mean(); cov_daily = dfa.cov()
                
                rand_r, rand_v = [], []
                if not dfa.empty:
                    for _ in range(800): 
                        w = np.random.random(len(valid_cols)); w /= np.sum(w)
                        eq = np.random.random() 
                        vol_st = np.sqrt(np.dot(w.T, np.dot(cov_daily, w)))
                        p_vol = vol_st * eq * np.sqrt(252)
                        ret_st_daily = np.sum(w * mu_daily)
                        rf_d_sim = (1 + rf_input)**(1/252) - 1
                        ret_port_daily = (ret_st_daily * eq) + (rf_d_sim * (1 - eq))
                        p_ret = (1 + ret_port_daily)**252 - 1
                        rand_r.append(p_ret); rand_v.append(p_vol)
                
                fig = go.Figure()
                if not bench_returns.empty:
                    b_ret_geo = (1 + bench_returns.mean()) ** 252 - 1
                    b_vol_ann = bench_returns.std() * np.sqrt(252)
                    if b_vol_ann > 0:
                        slope = (b_ret_geo - rf_input) / b_vol_ann
                        max_x = max(max(rand_v) if rand_v else 0, b_vol_ann, 0.4) * 1.3
                        end_y = rf_input + slope * max_x
                        fig.add_trace(go.Scatter(x=[0, max_x * 100], y=[rf_input * 100, end_y * 100], mode='lines', line=dict(color='#888', dash='dot', width=1), name='CML'))
                        fig.add_trace(go.Scatter(x=[b_vol_ann * 100], y=[b_ret_geo * 100], mode='markers+text', text=[selected_bench_label], textposition="bottom right", marker=dict(symbol='diamond', size=10, color='purple'), name='Benchmark'))

                sharpes = [0.5, 1.0, 2.0, 3.0]
                max_vol_plot = max(rand_v) if rand_v else 0.4
                max_vol_plot = max(max_vol_plot, 0.4) * 1.2
                for s in sharpes:
                    y_end = rf_input + (s * max_vol_plot)
                    fig.add_trace(go.Scatter(x=[0, max_vol_plot * 100], y=[rf_input * 100, y_end * 100], mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dashdot'), name=f'Sharpe {s}', hoverinfo='skip'))
                    fig.add_annotation(x=max_vol_plot*100, y=y_end*100, text=f"S={s}", showarrow=False, font=dict(size=10, color="gray"), xanchor="left")

                if rand_v: fig.add_trace(go.Scatter(x=np.array(rand_v)*100, y=np.array(rand_r)*100, mode='markers', marker=dict(color='#999', size=3, opacity=0.2), name='Simulações', hoverinfo='none'))

                ar, av, an = [], [], []
                for c in valid_cols:
                    r = (1 + dfa[c].mean()) ** 252 - 1; v = dfa[c].std() * np.sqrt(252)
                    ar.append(r*100); av.append(v*100); an.append(c)
                ar.append(rf_input*100); av.append(0.0); an.append("CAIXA")
                
                fig.add_trace(go.Scatter(x=av, y=ar, mode='markers+text', text=an, textposition="top center", marker=dict(color='#F1C40F', size=8, line=dict(width=1, color='black')), name='Ativos (BRL)'))
                fig.add_trace(go.Scatter(x=[mc['vol_ann']*100], y=[mc['ret_ann']*100], mode='markers', marker=dict(color='#00CC96', size=18, symbol='square'), name='Atual'))
                fig.add_trace(go.Scatter(x=[ms['vol_ann']*100], y=[ms['ret_ann']*100], mode='markers', marker=dict(color='#EF553B', size=18, symbol='triangle-up'), name='Simulado'))
                fig.add_annotation(x=ms['vol_ann']*100, y=ms['ret_ann']*100, ax=mc['vol_ann']*100, ay=mc['ret_ann']*100, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#666", opacity=0.5)
                fig.update_layout(xaxis_title="Volatilidade Anualizada (%)", yaxis_title="Retorno Anualizado (%)", template="plotly_white", height=650, margin=dict(l=0,r=0,t=20,b=0), legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center'))
                st.plotly_chart(fig, use_container_width=True)

        with t2:
            if valid_cols:
                fig_c = px.imshow(df_returns[valid_cols].corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1, origin='lower')
                fig_c.update_layout(height=650, template="plotly_white", margin=dict(l=0,r=0,t=40,b=0)); st.plotly_chart(fig_c, use_container_width=True)
            else: st.info("N/A")

        with t3:
            if valid_cols:
                sc = mc['series']; ss = ms['series']
                def calc_dd(s): w=(1+s).cumprod(); return (w-w.cummax())/w.cummax()
                dc = calc_dd(sc); ds = calc_dd(ss)
                fd = go.Figure()
                if not bench_returns.empty: db = calc_dd(bench_returns); fd.add_trace(go.Scatter(x=db.index, y=db, mode='lines', name=selected_bench_label, line=dict(color='#555', width=1.5, dash='dash')))
                fd.add_trace(go.Scatter(x=dc.index, y=dc, mode='lines', fill='tozeroy', name='Atual', line=dict(color='#00CC96', width=1)))
                fd.add_trace(go.Scatter(x=ds.index, y=ds, mode='lines', fill='tozeroy', name='Simulado', line=dict(color='#EF553B', width=1)))
                fd.update_layout(title="Drawdown (%) - Líquido de Taxas", yaxis_tickformat=".1%", template="plotly_white", height=650, margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center'), hovermode="x unified"); st.plotly_chart(fd, use_container_width=True)
            else: st.info("N/A")

        with t4:
            if valid_cols:
                sc = mc['series']; ss = ms['series']
                cc = 100*(1+sc).cumprod(); cs = 100*(1+ss).cumprod()
                fr = go.Figure()
                if not bench_returns.empty: cb = 100*(1+bench_returns).cumprod(); fr.add_trace(go.Scatter(x=cb.index, y=cb, mode='lines', name=selected_bench_label, line=dict(color='#555', width=1.5, dash='dash')))
                fr.add_trace(go.Scatter(x=cc.index, y=cc, mode='lines', name='Atual', line=dict(color='#00CC96', width=2)))
                fr.add_trace(go.Scatter(x=cs.index, y=cs, mode='lines', name='Simulado', line=dict(color='#EF553B', width=2)))
                fr.update_layout(title="Retorno Base 100 (BRL) - Líquido de Taxas", yaxis_title="R$", template="plotly_white", height=650, margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center'), hovermode="x unified"); st.plotly_chart(fr, use_container_width=True)
            else: st.info("N/A")