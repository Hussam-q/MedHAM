"""
MedHAM Interactive Research Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal
from itertools import combinations

st.set_page_config(
    page_title="MedHAM Research Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #ffffff; color: #1a1a2e; }
[data-testid="stSidebar"] { background: #f6f8fa; border-right: 1px solid #d0d7de; }
[data-testid="stSidebar"] * { color: #24292f !important; }
.metric-card { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 12px; padding: 20px 24px; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.metric-card.red::before   { background: linear-gradient(90deg, #C4687A, #d4889a); }
.metric-card.green::before { background: linear-gradient(90deg, #2D6A4F, #4d9a7f); }
.metric-card.blue::before  { background: linear-gradient(90deg, #1B3A5C, #3b6a9c); }
.metric-card.amber::before { background: linear-gradient(90deg, #C4A882, #d4c0a0); }
.metric-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #57606a; margin-bottom: 8px; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 36px; color: #1a1a2e; line-height: 1; margin-bottom: 4px; }
.metric-sub { font-size: 12px; color: #57606a; }
.section-header { font-family: 'DM Serif Display', serif; font-size: 26px; color: #1a1a2e; margin-bottom: 4px; }
.section-sub { font-size: 13px; color: #57606a; margin-bottom: 24px; line-height: 1.6; }
.badge-supported { background: #dafbe1; color: #1a7f37; border: 1px solid #4ac26b; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-partial { background: #fff8c5; color: #9a6700; border: 1px solid #d4a72c; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-descriptive { background: #ddf4ff; color: #0969da; border: 1px solid #54aeff; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-notsupported { background: #ffebe9; color: #cf222e; border: 1px solid #ff8182; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { background: #f6f8fa; border-radius: 8px; padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #57606a; border-radius: 6px; font-weight: 500; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #ffffff !important; color: #1a1a2e !important; }
.info-box { background: #f6f8fa; border: 1px solid #d0d7de; border-left: 3px solid #1B3A5C; border-radius: 8px; padding: 14px 18px; font-size: 13px; color: #24292f; margin-bottom: 16px; }
.warning-box { background: #fff8c5; border: 1px solid #d0d7de; border-left: 3px solid #C4A882; border-radius: 8px; padding: 14px 18px; font-size: 13px; color: #24292f; margin-bottom: 16px; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
hr { border-color: #d0d7de !important; }
.stat-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.stat-table th { background: #f6f8fa; color: #57606a; font-weight: 600; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; padding: 10px 14px; text-align: left; }
.stat-table td { padding: 10px 14px; border-bottom: 1px solid #d0d7de; color: #24292f; }
.stat-table tr:hover td { background: #eaeef2; }
code { font-family: 'JetBrains Mono', monospace; background: #f6f8fa; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #1B3A5C; }
[data-baseweb="tag"] { background: #f0f0ee !important; border: 0.5px solid #d3d1c7 !important; border-radius: 20px !important; }
[data-baseweb="tag"] span { color: #444441 !important; font-size: 12px !important; }
[data-baseweb="tag"] button { color: #5F5E5A !important; }
</style>
""", unsafe_allow_html=True)

MODEL_ORDER  = ['GPT-4o', 'Claude Sonnet', 'Gemini 2.5 Flash', 'Llama 3.1 8B']
COND_ORDER   = ['ZS_NoRAG', 'ZS_RAG', 'CIT_NoRAG', 'CIT_RAG']

MODEL_COLORS = {
    'GPT-4o':           '#1B3A5C',
    'Claude Sonnet':    '#C4A882',
    'Gemini 2.5 Flash': '#C4687A',
    'Llama 3.1 8B':     '#2D6A4F',
}
COND_COLORS = {
    'ZS_NoRAG':  '#C4A882',
    'ZS_RAG':    '#C4687A',
    'CIT_NoRAG': '#2D6A4F',
    'CIT_RAG':   '#1B3A5C',
}
# Outcome colors
COL_HALLU   = '#C4687A'
COL_MISINFO = '#C4A882'
COL_ACC     = '#1B3A5C'
COL_ACC0    = '#C4687A'
COL_ACC1    = '#C4A882'
COL_ACC2    = '#1B3A5C'

BASE_LAYOUT = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#f6f8fa',
    font=dict(family='DM Sans', color='#24292f', size=12),
    xaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickcolor='#57606a'),
    legend=dict(bgcolor='#ffffff', bordercolor='#d0d7de', borderwidth=1),
    margin=dict(l=50, r=30, t=60, b=50),
)

def apply_layout(fig, height=400, title=None, yaxis_title=None,
                 yaxis_tickformat=None, yaxis_range=None,
                 xaxis_title=None, barmode=None, showlegend=None):
    yaxis_cfg = dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickcolor='#57606a')
    if yaxis_title:      yaxis_cfg['title']     = yaxis_title
    if yaxis_tickformat: yaxis_cfg['tickformat'] = yaxis_tickformat
    if yaxis_range:      yaxis_cfg['range']      = yaxis_range
    xaxis_cfg = dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickcolor='#57606a')
    if xaxis_title:      xaxis_cfg['title']      = xaxis_title
    updates = dict(
        paper_bgcolor='#ffffff', plot_bgcolor='#f6f8fa',
        font=dict(family='DM Sans', color='#24292f', size=12),
        legend=dict(bgcolor='#ffffff', bordercolor='#d0d7de', borderwidth=1),
        margin=dict(l=50, r=30, t=60, b=50),
        height=height, yaxis=yaxis_cfg, xaxis=xaxis_cfg,
    )
    if title:    updates['title']    = dict(text=title, font_size=15)
    if barmode:  updates['barmode']  = barmode
    if showlegend is not None: updates['showlegend'] = showlegend
    fig.update_layout(**updates)
    return fig

def rank_biserial(u, n1, n2): return 1 - (2 * u) / (n1 * n2)
def cramers_v(chi2, n, k):    return np.sqrt(chi2 / (n * (k - 1)))
def interpret_r(r):
    r = abs(r)
    if r < 0.1:  return "negligible"
    elif r < 0.3: return "small"
    elif r < 0.5: return "moderate"
    else:         return "large"
def pvalue_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"

@st.cache_data(show_spinner=False)
def load_data():
    from datasets import load_dataset
    REPO = "Hussam-q/MedHAM"
    def _load(config):
        return load_dataset(REPO, config, split="train", trust_remote_code=False).to_pandas()
    items     = _load("benchmark_items")
    models_df = _load("models")
    strats    = _load("prompt_strategies")
    rag_ctx   = _load("rag_contexts")
    responses = _load("model_responses")
    signals   = _load("evaluation_signals")
    judges    = _load("judge_evaluations")
    results   = _load("evaluation_results")
    name_map  = {
        'GPT-4o': 'GPT-4o', 'Claude Sonnet 4.6': 'Claude Sonnet',
        'Gemini 2.5 Flash': 'Gemini 2.5 Flash', 'Llama 3.1 8B': 'Llama 3.1 8B',
    }
    df = (
        results
        .merge(responses[['response_id','item_id','model_id','strategy_id']], on='response_id')
        .merge(models_df[['model_id','model_name','model_type']], on='model_id')
        .merge(strats[['strategy_id','strategy_name','rag_enabled']], on='strategy_id')
        .merge(items[['item_id','hallucination_category','difficulty']], on='item_id')
    )
    df['model_label']      = df['model_name'].map(name_map)
    df['rag']              = df['rag_enabled'].fillna(0).astype(int).astype(bool)
    df['citation']         = df['strategy_name'].str.startswith('CIT')
    df['model_type_label'] = df['model_type'].map({
        'commercial_large': 'Commercial Large', 'open_source_small': 'Open Source Small'})
    df = df[df['model_label'].notna()].copy()
    df['hallucination_label']  = df['hallucination_label'].astype(int)
    df['misinformation_label'] = df['misinformation_label'].astype(int)
    df['accuracy_score']       = df['accuracy_score'].astype(int)
    df_sig = df.merge(
        signals[['response_id','biobert_f1','factscore_f1','citation_fake_count','citation_real_count']],
        on='response_id', how='left')
    return df, df_sig, judges, responses, items, rag_ctx

with st.sidebar:
    st.markdown("### 🏥 MedHAM Dashboard")
    st.markdown("---")
    st.markdown("**Filters**")
    selected_models       = st.multiselect("Models", MODEL_ORDER, default=MODEL_ORDER)
    selected_conditions   = st.multiselect("Conditions", COND_ORDER, default=COND_ORDER)
    selected_difficulties = st.multiselect("Difficulty", ['Easy','Medium','Hard'], default=['Easy','Medium','Hard'])
    st.markdown("---")
    st.markdown("""<div style='font-size:11px;color:#57606a;line-height:1.6'>
    <b>Dataset:</b> <a href='https://huggingface.co/datasets/Hussam-q/MedHAM' style='color:#1B3A5C'>Hussam-q/MedHAM</a><br>
    <b>N:</b> 16,000 responses<br><b>Models:</b> 4 LLMs<br>
    <b>Questions:</b> 1,000<br><b>Judges:</b> 4 blind LLM judges<br>
    <b>Design:</b> 2×2 factorial</div>""", unsafe_allow_html=True)

with st.spinner("Loading MedHAM dataset from HuggingFace..."):
    df_full, df_sig_full, judges_df, responses_df, items_df, rag_ctx_df = load_data()

def apply_filters(d):
    return d[
        d['model_label'].isin(selected_models) &
        d['strategy_name'].isin(selected_conditions) &
        d['difficulty'].str.capitalize().isin([x.capitalize() for x in selected_difficulties])
    ].copy()

df = apply_filters(df_full)
if len(df) == 0:
    st.warning("No data matches current filters.")
    st.stop()

st.markdown("""
<div style='padding:32px 0 16px'>
  <div style='font-family:DM Serif Display,serif;font-size:38px;color:#1a1a2e;line-height:1.1;margin-bottom:8px'>
    MedHAM Research Dashboard
  </div>
  <div style='font-size:15px;color:#57606a;max-width:700px'>
    Evaluating hallucination, accuracy, and misinformation in LLMs for medical question answering —
    comparative study of RAG and citation prompting across a 2×2 factorial design.
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview", "🔬  RQ1 — Model Comparison",
    "⚗️  RQ2 — Treatment Effects", "🧪  Hypothesis Tests", "🗃️  Data Explorer",
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Study Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Key metrics across all selected models and conditions</div>', unsafe_allow_html=True)

    h_rate   = df['hallucination_label'].mean()
    m_rate   = df['misinformation_label'].mean()
    acc_mean = df['accuracy_consensus'].mean()
    n_resp   = len(df)
    # Best model per metric
    model_hallu  = df.groupby('model_label')['hallucination_label'].mean()
    model_misinfo= df.groupby('model_label')['misinformation_label'].mean()
    model_acc    = df.groupby('model_label')['accuracy_consensus'].mean()
    best_hallu_model  = model_hallu.idxmin()
    best_hallu_rate   = model_hallu.min()
    best_misinfo_model= model_misinfo.idxmin()
    best_misinfo_rate = model_misinfo.min()
    best_acc_model    = model_acc.idxmax()
    best_acc_val      = model_acc.max()

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card red"><div class="metric-label">Hallucination Rate</div><div class="metric-value">{h_rate:.1%}</div><div class="metric-sub">of {n_resp:,} responses · lowest: <b>{best_hallu_model}</b> ({best_hallu_rate:.1%})</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card amber"><div class="metric-label">Misinformation Rate</div><div class="metric-value">{m_rate:.1%}</div><div class="metric-sub">of {n_resp:,} responses · lowest: <b>{best_misinfo_model}</b> ({best_misinfo_rate:.1%})</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card green"><div class="metric-label">Mean Accuracy</div><div class="metric-value">{acc_mean:.2f}</div><div class="metric-sub">scale 0–2 · highest: <b>{best_acc_model}</b> ({best_acc_val:.2f})</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    co1, co2, co3 = st.columns(3)
    with co1:
        fig = go.Figure(go.Pie(
            labels=['0 — Inaccurate','1 — Partial','2 — Accurate'],
            values=[(df['accuracy_score']==0).mean(),(df['accuracy_score']==1).mean(),(df['accuracy_score']==2).mean()],
            hole=0.55, marker_colors=[COL_ACC0, COL_ACC1, COL_ACC2],
            textinfo='label+percent', textfont_size=12,
        ))
        apply_layout(fig, height=340, title='Accuracy Score Distribution', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with co2:
        model_h = df.groupby('model_label')['hallucination_label'].mean().reindex(
            [m for m in MODEL_ORDER if m in df['model_label'].unique()])
        fig = go.Figure()
        for m in model_h.index:
            fig.add_trace(go.Bar(x=[m], y=[model_h[m]], marker_color=MODEL_COLORS.get(m,'#888'),
                showlegend=False, text=[f"{model_h[m]:.1%}"], textposition='outside', textfont_size=12))
        apply_layout(fig, height=340, title='Hallucination Rate by Model',
                     yaxis_title='Hallucination Rate', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    with co3:
        model_m = df.groupby('model_label')['misinformation_label'].mean().reindex(
            [m for m in MODEL_ORDER if m in df['model_label'].unique()])
        fig = go.Figure()
        for m in model_m.index:
            fig.add_trace(go.Bar(x=[m], y=[model_m[m]], marker_color=MODEL_COLORS.get(m,'#888'),
                showlegend=False, text=[f"{model_m[m]:.1%}"], textposition='outside', textfont_size=12))
        apply_layout(fig, height=340, title='Misinformation Rate by Model',
                     yaxis_title='Misinformation Rate', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Hallucination Rate — Model × Condition Heatmap")
    pivot = df.groupby(['model_label','strategy_name'])['hallucination_label'].mean().unstack()
    pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index],
                          columns=[c for c in COND_ORDER if c in pivot.columns])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,'#EBF5FB'],[0.2,'#BADDF5'],[0.4,'#74AEDA'],[0.6,'#3D7AB0'],[0.8,'#285480'],[1,'#1B3A5C']], zmin=0, zmax=1,
        text=[[f"{v:.1%}" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont_size=13,
        colorbar=dict(title='Rate', tickformat='.0%'),
    ))
    apply_layout(fig, height=280, xaxis_title='Condition', yaxis_title='Model')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">🔵 <b>Lower = fewer hallucinations.</b> Light blue = safer, deep blue = more hallucinated.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 2×2 Factorial Design")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame({'':['Zero-Shot','Citation-Required'],
            'No RAG':['ZS_NoRAG (baseline)','CIT_NoRAG'],
            'RAG':['ZS_RAG','CIT_RAG (combined)']}).set_index(''), use_container_width=True)
    with col2:
        st.markdown('<div class="info-box"><b>4 models</b> × <b>4 conditions</b> × <b>1,000 questions</b> = <b>16,000 responses</b><br><br>Evaluated by <b>4 blind LLM judges</b> (64,000 evaluations). DeepSeek V4 Flash is the only fully independent judge.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Research Questions and Hypotheses")
    rq_data = {
        'ID': ['RQ1','RQ2','H2.1','H2.2','H2.3','H2.4'],
        'Full Statement': [
            'To what extent do large language models produce hallucinated, inaccurate, and medically misinformative responses when answering evidence-based medical questions?',
            'How do retrieval-augmented generation and citation prompting — separately and together — affect hallucination, accuracy, and misinformation in large language model responses to medical questions, and do these effects differ across models and question difficulty levels?',
            'Retrieval-augmented generation will reduce hallucination and misinformation rates and improve accuracy scores relative to the no-retrieval baseline.',
            'The effect of retrieval augmentation and citation prompting on hallucination and misinformation rates and accuracy scores will be stronger for smaller models than for larger models.',
            'Citation prompting will reduce hallucination and misinformation rates and improve accuracy scores relative to zero-shot prompting.',
            'The combination of retrieval-augmented generation and citation prompting will yield lower hallucination and misinformation rates and higher accuracy scores than retrieval augmentation or citation prompting applied independently.',
        ]
    }
    st.dataframe(pd.DataFrame(rq_data).set_index('ID'), use_container_width=True)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">RQ1 — Model-Level Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">To what extent do large language models produce hallucinated, inaccurate, and medically misinformative responses when answering evidence-based medical questions?</div>', unsafe_allow_html=True)

    model_stats = df.groupby('model_label').agg(
        hallucination_rate  = ('hallucination_label','mean'),
        misinformation_rate = ('misinformation_label','mean'),
        accuracy_mean       = ('accuracy_consensus','mean'),
        acc0_rate = ('accuracy_score', lambda x: (x==0).mean()),
        acc1_rate = ('accuracy_score', lambda x: (x==1).mean()),
        acc2_rate = ('accuracy_score', lambda x: (x==2).mean()),
        n = ('response_id','count'),
    ).reindex([m for m in MODEL_ORDER if m in df['model_label'].unique()])

    metric_choice = st.radio("Primary metric",
        ['Hallucination Rate','Misinformation Rate','Mean Accuracy'], horizontal=True)
    col_map = {
        'Hallucination Rate':  ('hallucination_rate', 'Hallucination Rate', '.0%', False),
        'Misinformation Rate': ('misinformation_rate','Misinformation Rate','.0%', False),
        'Mean Accuracy':       ('accuracy_mean','Mean Accuracy (0–2)','.2f', True),
    }
    col_key, col_title, fmt, higher_better = col_map[metric_choice]
    vals = model_stats[col_key]
    fig = go.Figure()
    for m, v in zip(vals.index, vals.values):
        fig.add_trace(go.Bar(x=[m], y=[v], marker_color=MODEL_COLORS.get(m,'#888'), showlegend=False,
            text=[f"{v:.2f}" if "%" not in fmt else f"{v:.1%}"], textposition='outside', textfont=dict(size=14, color='#1a1a2e')))
    apply_layout(fig, height=380, title=f'{col_title} by Model — all conditions combined',
                 yaxis_title=col_title, yaxis_tickformat=fmt if '%' in fmt else None,
                 yaxis_range=[0,2] if col_key=='accuracy_mean' else None)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="info-box">ℹ️ {"Higher = better" if higher_better else "Lower = better"} for <b>{col_title}</b>.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Accuracy Score Distribution by Model")
    models_in = [m for m in MODEL_ORDER if m in model_stats.index]
    fig = go.Figure()
    for label, ck, color in [('0 — Inaccurate','acc0_rate',COL_ACC0),('1 — Partial','acc1_rate',COL_ACC1),('2 — Accurate','acc2_rate',COL_ACC2)]:
        fig.add_trace(go.Bar(name=label, x=models_in, y=model_stats.loc[models_in, ck].values, marker_color=color))
    apply_layout(fig, height=360, barmode='stack', yaxis_title='Proportion', yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Statistical Comparison Across Models")
    groups = [df[df['model_label']==m]['accuracy_consensus'].dropna().values for m in models_in]
    if len(groups) >= 2:
        kw_stat, kw_p = kruskal(*groups)
        st.markdown(f'<table class="stat-table"><tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Sig.</th></tr><tr><td>Kruskal-Wallis (accuracy across models)</td><td>H={kw_stat:.3f}</td><td>{kw_p:.4f}</td><td>{pvalue_stars(kw_p)}</td></tr></table>', unsafe_allow_html=True)
        rows_mw = []
        for m1, m2 in combinations(models_in, 2):
            a = df[df['model_label']==m1]['accuracy_consensus'].dropna().values
            b = df[df['model_label']==m2]['accuracy_consensus'].dropna().values
            u, p = mannwhitneyu(a, b, alternative='two-sided')
            r = rank_biserial(u, len(a), len(b))
            rows_mw.append({'Comparison':f'{m1} vs {m2}','U':f'{u:.0f}','p-value':f'{p:.4f}',
                            'Effect size r':f'{r:.3f}','Interpretation':interpret_r(r),'Sig.':pvalue_stars(p)})
        st.dataframe(pd.DataFrame(rows_mw), use_container_width=True, hide_index=True)
    st.markdown('<div class="warning-box">⚠️ <b>Independence note:</b> Same 1,000 items appear across all conditions — observations clustered by item_id. Interpret p-values alongside effect sizes.</div>', unsafe_allow_html=True)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">RQ2 — Treatment Effects</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How do retrieval-augmented generation and citation prompting — separately and together — affect hallucination, accuracy, and misinformation in large language model responses to medical questions, and do these effects differ across models and question difficulty levels?</div>', unsafe_allow_html=True)

    no_rag   = df[~df['rag']]
    with_rag = df[df['rag']]
    rag_summary = pd.DataFrame({
        'Condition': ['No RAG','RAG'],
        'Hallucination Rate': [no_rag['hallucination_label'].mean(), with_rag['hallucination_label'].mean()],
        'Misinformation Rate': [no_rag['misinformation_label'].mean(), with_rag['misinformation_label'].mean()],
        'Mean Accuracy': [no_rag['accuracy_consensus'].mean(), with_rag['accuracy_consensus'].mean()],
    })

    st.markdown("#### RAG Effect — All Outcomes")
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        for metric, color in [('Hallucination Rate', COL_HALLU),('Misinformation Rate', COL_MISINFO)]:
            fig.add_trace(go.Bar(name=metric, x=rag_summary['Condition'], y=rag_summary[metric],
                marker_color=color, text=[f"{v:.1%}" for v in rag_summary[metric]], textposition='outside'))
        apply_layout(fig, height=360, barmode='group',
                     title='Hallucination & Misinformation — RAG vs No-RAG',
                     yaxis_title='Rate', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rag_summary['Condition'], y=rag_summary['Mean Accuracy'],
            marker_color=[COL_MISINFO, COL_ACC2],
            text=[f"{v:.3f}" for v in rag_summary['Mean Accuracy']],
            textposition='outside', showlegend=False))
        apply_layout(fig, height=360, title='Mean Accuracy — RAG vs No-RAG',
                     yaxis_title='Mean Accuracy (0–2)', yaxis_range=[0,2])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### RAG Delta Per Model")
    st.markdown('<div class="info-box">🔵 Negative hallucination delta = RAG <b>reduced</b> hallucination. Negative misinformation delta = RAG <b>reduced</b> misinformation. Positive accuracy delta = RAG <b>improved</b> accuracy.</div>', unsafe_allow_html=True)
    models_in = [m for m in MODEL_ORDER if m in df['model_label'].unique()]
    delta_rows = []
    for m in models_in:
        m_no  = df[(df['model_label']==m)&(~df['rag'])]
        m_yes = df[(df['model_label']==m)&(df['rag'])]
        delta_rows.append({'Model':m,
            'Hallucination Δ': m_yes['hallucination_label'].mean()-m_no['hallucination_label'].mean(),
            'Accuracy Δ':      m_yes['accuracy_consensus'].mean() -m_no['accuracy_consensus'].mean(),
            'Misinformation Δ':m_yes['misinformation_label'].mean()-m_no['misinformation_label'].mean()})
    delta_df = pd.DataFrame(delta_rows)
    fig = go.Figure()
    for cn, color in [('Hallucination Δ', COL_HALLU),('Accuracy Δ', COL_ACC),('Misinformation Δ', COL_MISINFO)]:
        fig.add_trace(go.Bar(name=cn, x=delta_df['Model'], y=delta_df[cn], marker_color=color))
    fig.add_hline(y=0, line_dash='dash', line_color='#57606a', line_width=1)
    apply_layout(fig, height=380, barmode='group',
                 title='RAG Effect Delta per Model — Hallucination, Misinformation &amp; Accuracy (RAG minus No-RAG)', yaxis_title='Delta')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### RAG × Citation Interaction Profile")
    conds_in    = [c for c in COND_ORDER if c in df['strategy_name'].unique()]
    metric_inter = st.radio("Interaction metric", ['Hallucination Rate','Misinformation Rate','Mean Accuracy'], horizontal=True, key='inter_metric')
    if metric_inter == 'Hallucination Rate':
        y_col, y_title = 'hallu', 'Hallucination Rate'
    elif metric_inter == 'Misinformation Rate':
        y_col, y_title = 'misinfo', 'Misinformation Rate'
    else:
        y_col, y_title = 'acc', 'Mean Accuracy (0–2)'
    interaction = df.groupby(['model_label','strategy_name']).agg(
        hallu=('hallucination_label','mean'),
        misinfo=('misinformation_label','mean'),
        acc=('accuracy_consensus','mean')).reset_index()
    fig = go.Figure()
    for m in models_in:
        sub = interaction[interaction['model_label']==m].set_index('strategy_name').reindex(conds_in)
        fig.add_trace(go.Scatter(x=conds_in, y=sub[y_col].values, name=m, mode='lines+markers',
            line=dict(color=MODEL_COLORS.get(m,'#888'), width=2), marker=dict(size=8)))
    apply_layout(fig, height=400, title=f'RAG × Citation Interaction — {y_title}',
                 xaxis_title='Condition', yaxis_title=y_title)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Difficulty Moderation")
    diff_stats = df.groupby(['difficulty','rag']).agg(acc=('accuracy_consensus','mean')).reset_index()
    diff_stats['rag_label'] = diff_stats['rag'].map({True:'RAG',False:'No RAG'})
    fig = px.bar(diff_stats, x='difficulty', y='acc', color='rag_label', barmode='group',
        color_discrete_map={'RAG': COL_ACC, 'No RAG': COL_MISINFO},
        labels={'acc':'Mean Accuracy (0–2)','difficulty':'Difficulty','rag_label':''},
        title='Mean Accuracy by Difficulty — RAG vs No-RAG', text_auto='.2f')
    fig.update_traces(textposition='outside')
    fig.update_layout(**BASE_LAYOUT, height=360)
    st.plotly_chart(fig, use_container_width=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Hypothesis Tests</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">H2.1 – H2.4 with effect sizes and verdicts</div>', unsafe_allow_html=True)

    no_rag   = df[~df['rag']]
    with_rag = df[df['rag']]
    no_cit   = df[~df['citation']]
    with_cit = df[df['citation']]

    with st.expander("H2.1 — Effect of Retrieval-Augmented Generation", expanded=True):
        st.markdown("*Retrieval-augmented generation will reduce hallucination and misinformation rates and improve accuracy scores relative to the no-retrieval baseline.*")
        u_acc, p_acc = mannwhitneyu(with_rag['accuracy_consensus'].dropna(), no_rag['accuracy_consensus'].dropna(), alternative='two-sided')
        r_acc = rank_biserial(u_acc, len(with_rag['accuracy_consensus'].dropna()), len(no_rag['accuracy_consensus'].dropna()))
        ct_h = df.groupby(['hallucination_label','rag']).size().unstack(fill_value=0)
        chi2_h, p_h, _, _ = chi2_contingency(ct_h)
        v_h = cramers_v(chi2_h, len(df), min(ct_h.shape))
        hallu_delta   = with_rag['hallucination_label'].mean()-no_rag['hallucination_label'].mean()
        misinfo_delta = with_rag['misinformation_label'].mean()-no_rag['misinformation_label'].mean()
        acc_delta     = with_rag['accuracy_consensus'].mean() -no_rag['accuracy_consensus'].mean()
        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f"""<table class="stat-table">
            <tr><th>Outcome</th><th>No RAG</th><th>RAG</th><th>Delta</th></tr>
            <tr><td>Hallucination</td><td>{no_rag['hallucination_label'].mean():.3f}</td><td>{with_rag['hallucination_label'].mean():.3f}</td>
                <td style="color:{COL_HALLU if hallu_delta>0 else COL_ACC2}">{hallu_delta:+.3f}</td></tr>
            <tr><td>Misinformation</td><td>{no_rag['misinformation_label'].mean():.3f}</td><td>{with_rag['misinformation_label'].mean():.3f}</td>
                <td style="color:{COL_HALLU if misinfo_delta>0 else COL_ACC2}">{misinfo_delta:+.3f}</td></tr>
            <tr><td>Accuracy</td><td>{no_rag['accuracy_consensus'].mean():.3f}</td><td>{with_rag['accuracy_consensus'].mean():.3f}</td>
                <td style="color:{COL_ACC2 if acc_delta>0 else COL_HALLU}">{acc_delta:+.3f}</td></tr>
            </table>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<table class="stat-table">
            <tr><th>Test</th><th>Stat</th><th>p</th><th>Effect</th></tr>
            <tr><td>Mann-Whitney (accuracy)</td><td>U={u_acc:.0f}</td><td>{p_acc:.4f} {pvalue_stars(p_acc)}</td><td>r={r_acc:.3f} ({interpret_r(r_acc)})</td></tr>
            <tr><td>χ² (hallucination)</td><td>χ²={chi2_h:.2f}</td><td>{p_h:.4f} {pvalue_stars(p_h)}</td><td>V={v_h:.3f} ({interpret_r(v_h)})</td></tr>
            </table>""", unsafe_allow_html=True)
        h21_hallu_ok   = hallu_delta < 0
        h21_misinfo_ok = misinfo_delta < 0
        h21_acc_ok     = acc_delta > 0 and p_acc < 0.05
        _h21_confirmed = [x for x,v in [('accuracy improvement',h21_acc_ok),('hallucination reduction',h21_hallu_ok),('misinformation reduction',h21_misinfo_ok)] if v]
        _h21_n = len(_h21_confirmed)
        if _h21_n == 3:
            h21_label, h21_badge = "Supported — all three outcomes confirmed", "supported"
        elif _h21_n > 0:
            h21_label, h21_badge = f"Partially Supported — confirmed: {', '.join(_h21_confirmed)}", "partial"
        else:
            h21_label, h21_badge = "Not Supported", "notsupported"
        st.markdown(f'**Verdict:** <span class="badge-{h21_badge}">{h21_label}</span> — Accuracy Δ={acc_delta:+.3f} (p={p_acc:.4f}, r={r_acc:.3f}, {interpret_r(r_acc)} effect). Hallucination Δ={hallu_delta:+.3f}. Misinformation Δ={misinfo_delta:+.3f}.', unsafe_allow_html=True)

    with st.expander("H2.2 — Model Size Moderation (Descriptive)"):
        st.markdown("*The effect of retrieval augmentation and citation prompting on hallucination and misinformation rates and accuracy scores will be stronger for smaller models than for larger models.*")
        st.markdown('<div class="warning-box">⚠️ Only one open-source small model (Llama 3.1 8B). Formal testing requires multiple models per size tier. The comparison below is exploratory only.</div>', unsafe_allow_html=True)
        models_in = [m for m in MODEL_ORDER if m in df['model_label'].unique()]
        d2 = []
        for m in models_in:
            mn=df[(df['model_label']==m)&(~df['rag'])]; my=df[(df['model_label']==m)&(df['rag'])]
            d2.append({'Model':m,'Type':'Open Source Small' if m=='Llama 3.1 8B' else 'Commercial Large',
                'RAG Δ Hallucination':  round(my['hallucination_label'].mean() -mn['hallucination_label'].mean(),  4),
                'RAG Δ Misinformation': round(my['misinformation_label'].mean()-mn['misinformation_label'].mean(), 4),
                'RAG Δ Accuracy':       round(my['accuracy_consensus'].mean()  -mn['accuracy_consensus'].mean(),   4)})
        st.dataframe(pd.DataFrame(d2), use_container_width=True, hide_index=True)
        st.markdown('<div class="info-box">ℹ️ Positive Δ hallucination = RAG increased hallucination rate. Negative Δ misinformation = RAG reduced misinformation. Positive Δ accuracy = RAG improved accuracy. Descriptive only — single open-source model.</div>', unsafe_allow_html=True)
        st.markdown('**Verdict:** <span class="badge-descriptive">Descriptive Only</span> — Single open-source model precludes formal statistical test. Replication with multiple models per size tier required.', unsafe_allow_html=True)

    with st.expander("H2.3 — Effect of Citation Prompting"):
        st.markdown("*Citation prompting will reduce hallucination and misinformation rates and improve accuracy scores relative to zero-shot prompting.*")
        u_c,p_c = mannwhitneyu(with_cit['accuracy_consensus'].dropna(), no_cit['accuracy_consensus'].dropna(), alternative='two-sided')
        r_c = rank_biserial(u_c, len(with_cit['accuracy_consensus'].dropna()), len(no_cit['accuracy_consensus'].dropna()))
        ct_hc = df.groupby(['hallucination_label','citation']).size().unstack(fill_value=0)
        chi2_hc,p_hc,_,_ = chi2_contingency(ct_hc)
        v_hc = cramers_v(chi2_hc, len(df), min(ct_hc.shape))
        acc_delta_c       = with_cit['accuracy_consensus'].mean()-no_cit['accuracy_consensus'].mean()
        cit_hallu_delta   = with_cit['hallucination_label'].mean()-no_cit['hallucination_label'].mean()
        cit_misinfo_delta = with_cit['misinformation_label'].mean()-no_cit['misinformation_label'].mean()
        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f"""<table class="stat-table">
            <tr><th>Outcome</th><th>Zero-Shot</th><th>Citation</th><th>Delta</th></tr>
            <tr><td>Hallucination</td><td>{no_cit['hallucination_label'].mean():.3f}</td><td>{with_cit['hallucination_label'].mean():.3f}</td>
                <td style="color:{COL_ACC2 if cit_hallu_delta<0 else COL_HALLU}">{cit_hallu_delta:+.3f}</td></tr>
            <tr><td>Misinformation</td><td>{no_cit['misinformation_label'].mean():.3f}</td><td>{with_cit['misinformation_label'].mean():.3f}</td>
                <td style="color:{COL_ACC2 if cit_misinfo_delta<0 else COL_HALLU}">{cit_misinfo_delta:+.3f}</td></tr>
            <tr><td>Accuracy</td><td>{no_cit['accuracy_consensus'].mean():.3f}</td><td>{with_cit['accuracy_consensus'].mean():.3f}</td>
                <td style="color:{COL_ACC2 if acc_delta_c>0 else COL_HALLU}">{acc_delta_c:+.3f}</td></tr>
            </table>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<table class="stat-table">
            <tr><th>Test</th><th>Stat</th><th>p</th><th>Effect</th></tr>
            <tr><td>Mann-Whitney (accuracy)</td><td>U={u_c:.0f}</td><td>{p_c:.4f} {pvalue_stars(p_c)}</td><td>r={r_c:.3f} ({interpret_r(r_c)})</td></tr>
            <tr><td>χ² (hallucination)</td><td>χ²={chi2_hc:.2f}</td><td>{p_hc:.4f} {pvalue_stars(p_hc)}</td><td>V={v_hc:.3f} ({interpret_r(v_hc)})</td></tr>
            </table>""", unsafe_allow_html=True)
        h23_hallu_ok   = cit_hallu_delta < 0
        h23_misinfo_ok = cit_misinfo_delta < 0
        h23_acc_ok     = p_c < 0.05 and acc_delta_c > 0
        _h23_confirmed = [x for x,v in [('accuracy improvement',h23_acc_ok),('hallucination reduction',h23_hallu_ok),('misinformation reduction',h23_misinfo_ok)] if v]
        _h23_n = len(_h23_confirmed)
        if _h23_n == 3:
            h23_label, h23_badge = "Supported — all three outcomes confirmed", "supported"
        elif _h23_n > 0:
            h23_label, h23_badge = f"Partially Supported — confirmed: {', '.join(_h23_confirmed)}", "partial"
        else:
            h23_label, h23_badge = "Not Supported", "notsupported"
        st.markdown(f'**Verdict:** <span class="badge-{h23_badge}">{h23_label}</span> — Accuracy Δ={acc_delta_c:+.3f} (p={p_c:.4f}, r={r_c:.3f}, {interpret_r(r_c)} effect). Hallucination Δ={cit_hallu_delta:+.3f}. Misinformation Δ={cit_misinfo_delta:+.3f}.', unsafe_allow_html=True)

    with st.expander("H2.4 — Combined RAG + Citation (CIT_RAG)"):
        st.markdown("*The combination of retrieval-augmented generation and citation prompting will yield lower hallucination and misinformation rates and higher accuracy scores than retrieval augmentation or citation prompting applied independently.*")
        cond_acc  = df.groupby('strategy_name')['accuracy_consensus'].mean()
        cond_hallu = df.groupby('strategy_name')['hallucination_label'].mean()
        best_cond_acc   = cond_acc.idxmax()
        best_cond_hallu = cond_hallu.idxmin()
        cs = df.groupby('strategy_name').agg(
            hallu=('hallucination_label','mean'), acc=('accuracy_consensus','mean'), misinfo=('misinformation_label','mean')
        ).reindex([c for c in COND_ORDER if c in df['strategy_name'].unique()])
        st.dataframe(cs.round(4).rename(columns={'hallu':'Hallucination Rate','acc':'Mean Accuracy','misinfo':'Misinformation Rate'}), use_container_width=True)
        fig = go.Figure()
        for cond in cs.index:
            fig.add_trace(go.Bar(name=cond, x=['Hallucination','Accuracy/2','Misinformation'],
                y=[cs.loc[cond,'hallu'],cs.loc[cond,'acc']/2,cs.loc[cond,'misinfo']],
                marker_color=COND_COLORS.get(cond,'#888')))
        apply_layout(fig, height=360, barmode='group', title='All Conditions — Normalised Outcomes', yaxis_title='Rate / Normalised Score')
        st.plotly_chart(fig, use_container_width=True)
        cond_misinfo   = df.groupby('strategy_name')['misinformation_label'].mean()
        best_cond_misinfo = cond_misinfo.idxmin()
        h24_hallu_ok   = best_cond_hallu  == 'CIT_RAG'
        h24_misinfo_ok = best_cond_misinfo == 'CIT_RAG'
        h24_acc_ok     = best_cond_acc     == 'CIT_RAG'
        _h24_confirmed = [x for x,v in [('accuracy',h24_acc_ok),('hallucination reduction',h24_hallu_ok),('misinformation reduction',h24_misinfo_ok)] if v]
        _h24_n = len(_h24_confirmed)
        if _h24_n == 3:
            h24_label, h24_badge = "Supported — CIT_RAG leads on all three outcomes", "supported"
        elif _h24_n > 0:
            h24_label, h24_badge = f"Partially Supported — CIT_RAG leads on: {', '.join(_h24_confirmed)}", "partial"
        else:
            h24_label, h24_badge = "Not Supported — CIT_RAG does not lead on any outcome", "notsupported"
        st.markdown(f'**Verdict:** <span class="badge-{h24_badge}">{h24_label}</span> — Best accuracy: <code>{best_cond_acc}</code>. Lowest hallucination: <code>{best_cond_hallu}</code>. Lowest misinformation: <code>{best_cond_misinfo}</code>.', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Hypothesis Verdict Summary")
    st.markdown(f"""
    <table class="stat-table">
    <tr><th>ID</th><th>Hypothesis</th><th>Verdict</th><th>Key Evidence</th></tr>
    <tr><td>H2.1</td>
        <td>Retrieval-augmented generation will reduce hallucination and misinformation rates and improve accuracy scores relative to the no-retrieval baseline.</td>
        <td><span class="badge-{h21_badge}">{h21_label}</span></td>
        <td>Accuracy Δ={acc_delta:+.3f}; Hallucination Δ={hallu_delta:+.3f}; Misinformation Δ={misinfo_delta:+.3f}</td></tr>
    <tr><td>H2.2</td>
        <td>The effect of retrieval augmentation and citation prompting will be stronger for smaller models than for larger models.</td>
        <td><span class="badge-descriptive">Descriptive Only</span></td>
        <td>Single open-source model precludes formal test</td></tr>
    <tr><td>H2.3</td>
        <td>Citation prompting will reduce hallucination and misinformation rates and improve accuracy scores relative to zero-shot prompting.</td>
        <td><span class="badge-{h23_badge}">{h23_label}</span></td>
        <td>Accuracy Δ={acc_delta_c:+.3f}; Hallucination Δ={cit_hallu_delta:+.3f}; Misinformation Δ={cit_misinfo_delta:+.3f}</td></tr>
    <tr><td>H2.4</td>
        <td>The combination of RAG and citation prompting will yield lower hallucination and higher accuracy than either intervention independently.</td>
        <td><span class="badge-{h24_badge}">{h24_label}</span></td>
        <td>Best accuracy={best_cond_acc}; Lowest hallucination={best_cond_hallu}; Lowest misinformation={best_cond_misinfo}</td></tr>
    </table>""", unsafe_allow_html=True)
    st.markdown('<div class="warning-box">⚠️ Observations clustered by item_id — interpret p-values alongside effect sizes r and V.</div>', unsafe_allow_html=True)

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse, filter, and inspect raw evaluation results</div>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1: exp_models = st.multiselect("Model", MODEL_ORDER, default=MODEL_ORDER[:2], key='exp_m')
    with col2: exp_conds  = st.multiselect("Condition", COND_ORDER, default=COND_ORDER[:2], key='exp_c')
    with col3: exp_diff   = st.multiselect("Difficulty", ['Easy','Medium','Hard'], default=['Easy','Medium','Hard'], key='exp_d')

    col4,col5,col6 = st.columns(3)
    with col4: exp_hallu  = st.selectbox("Hallucination", ['All','Yes (1)','No (0)'], key='exp_h')
    with col5: exp_misinfo= st.selectbox("Misinformation", ['All','Yes (1)','No (0)'], key='exp_mi')
    with col6: exp_acc    = st.selectbox("Accuracy Score", ['All','0','1','2'], key='exp_a')

    exp_df = df_full[
        df_full['model_label'].isin(exp_models) &
        df_full['strategy_name'].isin(exp_conds) &
        df_full['difficulty'].str.capitalize().isin([d.capitalize() for d in exp_diff])
    ].copy()
    if exp_hallu=='Yes (1)':   exp_df = exp_df[exp_df['hallucination_label']==1]
    elif exp_hallu=='No (0)':  exp_df = exp_df[exp_df['hallucination_label']==0]
    if exp_misinfo=='Yes (1)': exp_df = exp_df[exp_df['misinformation_label']==1]
    elif exp_misinfo=='No (0)':exp_df = exp_df[exp_df['misinformation_label']==0]
    if exp_acc!='All': exp_df = exp_df[exp_df['accuracy_score']==int(exp_acc)]

    disp_cols = ['model_label','strategy_name','difficulty','hallucination_category',
                 'hallucination_label','misinformation_label','accuracy_score','accuracy_consensus']
    av_cols = [c for c in disp_cols if c in exp_df.columns]
    st.markdown(f"**{len(exp_df):,} responses** match current filters")
    st.dataframe(exp_df[av_cols].rename(columns={
        'model_label':'Model','strategy_name':'Condition','difficulty':'Difficulty',
        'hallucination_category':'Category','hallucination_label':'Hallucinated',
        'misinformation_label':'Misinformation','accuracy_score':'Accuracy','accuracy_consensus':'Consensus'
    }).head(500), use_container_width=True, hide_index=True)
    if len(exp_df)>500: st.markdown(f"*Showing first 500 of {len(exp_df):,} rows.*")

    st.markdown("---")
    st.markdown("#### Hallucination by Category")
    cat_stats = exp_df.groupby('hallucination_category')['hallucination_label'].mean().sort_values(ascending=True)
    if len(cat_stats) > 0:
        fig = go.Figure(go.Bar(
            x=cat_stats.values, y=cat_stats.index, orientation='h',
            marker=dict(color=cat_stats.values, colorscale=[[0,'#EBF5FB'],[0.2,'#BADDF5'],[0.4,'#74AEDA'],[0.6,'#3D7AB0'],[0.8,'#285480'],[1,'#1B3A5C']], cmin=0, cmax=1,
                        showscale=True, colorbar=dict(title='Rate',tickformat='.0%')),
            text=[f"{v:.1%}" for v in cat_stats.values], textposition='outside',
        ))
        apply_layout(fig, height=300, title='Hallucination Rate by MedHallu Category',
                     xaxis_title='Hallucination Rate')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""<div style='text-align:center;color:#57606a;font-size:12px;padding:16px 0'>
  MedHAM Research Dashboard · Hussam Alqahtani ·
  <a href='https://huggingface.co/datasets/Hussam-q/MedHAM' style='color:#1B3A5C'>Dataset: Hussam-q/MedHAM</a>
</div>""", unsafe_allow_html=True)
