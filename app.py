import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from eda_utils import EDAAnalyzer
from ai_chatbot import SmartEDA

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="Smart EDA Chatbot", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header{font-size:2.5rem;font-weight:700;color:#fff!important;text-align:center;margin-bottom:.5rem}
    .sub-header{text-align:center;color:#94a3b8!important;margin-bottom:2rem;font-size:1rem}
    .user-message{background:#374151;border-radius:12px;padding:1rem;margin-bottom:.8rem;border-left:4px solid #3b82f6;color:#fff!important}
    .bot-message{background:#4b5563;border-radius:12px;padding:1rem;margin-bottom:.8rem;border-left:4px solid #f59e0b;white-space:pre-wrap;line-height:1.5;color:#fff!important}
    .metric-card{background:#374151;padding:1rem;border-radius:10px;border-left:4px solid #3b82f6;color:#fff!important}
    .upload-area{border:2px dashed #6b7280;border-radius:10px;padding:2rem;text-align:center;background:#374151;color:#fff!important}
    .stButton>button{background:#3b82f6;color:white;border:none;border-radius:8px;padding:.5rem 1rem;font-weight:600}
    .stButton>button:hover{background:#2563eb}
    h1,h2,h3,h4,h5,h6,p,span,div,label,.stMarkdown,.stMetric,.stMetric>div>div>div{color:#fff!important}
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────
for key, default in [('df', None), ('analyzer', None), ('engine', None),
                     ('chat_history', []), ('file_name', None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def load_data(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return pd.read_csv(uploaded_file)
        elif ext in ('xlsx', 'xls'):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None


def main():
    st.markdown('<h1 class="main-header">🧠 Smart EDA Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload any CSV → Auto-train ML models → Full EDA → Ask anything</p>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose CSV / Excel", type=['csv', 'xlsx', 'xls'])

        if uploaded_file and st.session_state.file_name != uploaded_file.name:
            with st.spinner("Loading data…"):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.analyzer = EDAAnalyzer(df)
                    engine = SmartEDA(df)
                    engine.run_full_eda()
                    # auto-detect target
                    target = engine.detect_target()
                    if target:
                        engine.set_target(target)
                    st.session_state.engine = engine
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✅ Loaded **{uploaded_file.name}** ({df.shape[0]}×{df.shape[1]})")

        if st.session_state.df is not None:
            st.markdown("---")
            info = st.session_state.analyzer.get_basic_info()
            c1, c2 = st.columns(2)
            c1.metric("Rows", info['shape'][0])
            c2.metric("Cols", info['shape'][1])
            st.metric("Memory", f"{info['memory_usage']:.2f} MB")

            total_miss = sum(info['missing_values'].values())
            if total_miss:
                st.warning(f"⚠️ {total_miss} missing values")
            else:
                st.success("✅ No missing values")

            # ── Train Models ─────────────────────────────────────
            st.markdown("---")
            st.header("🤖 Model Training")
            engine = st.session_state.engine
            target_options = st.session_state.df.columns.tolist()
            current_target = engine.target_col if engine.target_col else (target_options[0] if target_options else None)
            idx = target_options.index(current_target) if current_target in target_options else 0
            selected_target = st.selectbox("Target Column", target_options, index=idx)

            if st.button("🚀 Train Models", use_container_width=True):
                engine.set_target(selected_target)
                with st.spinner("Training 3 ML models…"):
                    report = engine.train_models()
                if 'error' in report:
                    st.error(report['error'])
                else:
                    st.success(f"✅ Best: **{report['best_model']}** ({report['best_score']:.4f})")
                    st.session_state.engine = engine

            if engine.training_report:
                r = engine.training_report
                st.caption(f"Task: {r['task_type']} | Best: {r['best_model']}")

    # ── Main Content ─────────────────────────────────────────────
    if st.session_state.df is None:
        _welcome_screen()
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📊 Full EDA", "📈 Visualizations", "🤖 Model Results"])

    # ── TAB 1: Chat ──────────────────────────────────────────────
    with tab1:
        st.subheader("Chat with your Data")
        st.caption("💡 Ask predictions, statistics, counts, correlations — anything!")
        for msg in st.session_state.chat_history:
            cls = "user-message" if msg['role'] == 'user' else "bot-message"
            label = "You" if msg['role'] == 'user' else "Bot"
            st.markdown(f'<div class="{cls}"><strong>{label}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

        user_input = st.chat_input("Ask anything about your data…")
        if user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            with st.spinner("Analyzing…"):
                response = st.session_state.engine.answer_question(user_input)
            st.session_state.chat_history.append({'role': 'bot', 'content': response})
            st.rerun()

        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ── TAB 2: Full EDA ──────────────────────────────────────────
    with tab2:
        _render_full_eda()

    # ── TAB 3: Visualizations ────────────────────────────────────
    with tab3:
        _render_visualizations()

    # ── TAB 4: Model Results ─────────────────────────────────────
    with tab4:
        _render_model_results()


# ═══════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════

def _welcome_screen():
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#667eea">👋 Welcome to Smart EDA Chatbot!</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8">Upload a CSV or Excel file to start.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    cols = st.columns(4)
    features = [("💬", "Chat Interface", "Ask any question in natural language"),
                ("🧠", "Self-Trained ML", "Trains models locally — no API needed"),
                ("📊", "Full Auto EDA", "Statistics, correlations, outliers, distributions"),
                ("🔮", "Predictions", "Predict target values with trained models")]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(f'<div style="text-align:center;padding:1rem"><div style="font-size:2rem">{icon}</div>'
                     f'<h4>{title}</h4><p style="color:#94a3b8;font-size:.9rem">{desc}</p></div>', unsafe_allow_html=True)


def _render_full_eda():
    st.subheader("📊 Full Exploratory Data Analysis")
    engine = st.session_state.engine
    eda = engine.full_eda_cache
    df = st.session_state.df

    # Overview
    o = eda['overview']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", o['rows'])
    c2.metric("Columns", o['columns'])
    c3.metric("Memory", f"{o['memory_mb']} MB")
    c4.metric("Duplicates", o['duplicates'])

    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Missing values
    st.subheader("❓ Missing Values")
    if eda['missing']:
        mv_df = pd.DataFrame([{'Column': c, 'Count': v['count'], 'Percent': v['pct']}
                               for c, v in eda['missing'].items()])
        st.dataframe(mv_df.sort_values('Count', ascending=False), use_container_width=True)
    else:
        st.success("No missing values!")

    # Numeric stats
    if eda['numeric_stats']:
        st.subheader("📈 Numeric Column Statistics")
        stats_rows = []
        for c, s in eda['numeric_stats'].items():
            stats_rows.append({'Column': c, 'Mean': s['mean'], 'Median': s['median'],
                               'Std': s['std'], 'Min': s['min'], 'Max': s['max'],
                               'Skew': s['skew'], 'Kurtosis': s['kurtosis']})
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    # Correlations
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        st.subheader("🔗 Correlation Heatmap")
        fig = px.imshow(df[num_cols].corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    # Outliers
    if eda['outliers']:
        st.subheader("⚡ Outlier Summary")
        out_df = pd.DataFrame([{'Column': c, 'Count': v['count'], '%': v['pct'],
                                'Lower': v['lower_bound'], 'Upper': v['upper_bound']}
                                for c, v in eda['outliers'].items()])
        st.dataframe(out_df.sort_values('Count', ascending=False), use_container_width=True)

    # Categorical
    if eda['categorical_stats']:
        st.subheader("🏷️ Categorical Columns")
        for c, info in eda['categorical_stats'].items():
            with st.expander(f"{c} ({info['unique']} unique)"):
                vc_df = pd.DataFrame(list(info['top_values'].items()), columns=['Value', 'Count'])
                st.dataframe(vc_df, use_container_width=True)


def _render_visualizations():
    st.subheader("📈 Interactive Visualizations")
    df = st.session_state.df
    analyzer = st.session_state.analyzer
    plot_type = st.selectbox("Plot type:", ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Correlation Heatmap"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if plot_type == "Histogram" and num_cols:
        col = st.selectbox("Column:", num_cols)
        bins = st.slider("Bins:", 10, 100, 30)
        fig = analyzer.create_histogram(col, bins)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Box Plot" and num_cols:
        col = st.selectbox("Column:", num_cols)
        fig = analyzer.create_box_plot(col)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Bar Chart" and cat_cols:
        col = st.selectbox("Column:", cat_cols)
        top_n = st.slider("Top N:", 5, 20, 10)
        fig = analyzer.create_bar_chart(col, top_n)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Scatter Plot" and len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", num_cols)
        y = c2.selectbox("Y:", num_cols, index=min(1, len(num_cols)-1))
        fig = analyzer.create_scatter_plot(x, y)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Correlation Heatmap":
        fig = analyzer.create_correlation_heatmap()
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def _render_model_results():
    """Dedicated tab for Model Training Results, Confusion Matrix, ROC Curve."""
    st.subheader("🤖 Model Training Results")
    engine = st.session_state.engine

    if not engine.training_report:
        st.info("⚠️ No models trained yet. Use the **Train Models** button in the sidebar.")
        return

    r = engine.training_report

    # ── Metrics table ──
    st.markdown(engine._format_training_report())

    # ── All models comparison chart ──
    if r['task_type'] == 'classification':
        model_data = []
        for name, metrics in r['results'].items():
            if 'error' not in metrics:
                model_data.append({'Model': name, 'Accuracy': metrics['accuracy'],
                                   'Precision': metrics['precision'], 'Recall': metrics['recall'],
                                   'F1': metrics['f1']})
        if model_data:
            st.subheader("📊 Model Comparison")
            comp_df = pd.DataFrame(model_data)
            fig = px.bar(comp_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                         x='Model', y='Score', color='Metric', barmode='group',
                         title='Model Performance Comparison')
            st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ──
    best = r.get('best_model')
    if r['task_type'] == 'classification' and best and best in r['results']:
        best_metrics = r['results'][best]
        cm = best_metrics.get('confusion_matrix')
        labels = r.get('class_labels')
        if cm is not None:
            st.subheader(f"📊 Confusion Matrix — {best}")
            cm_array = np.array(cm)
            labels_str = [str(l) for l in labels] if labels else [str(i) for i in range(len(cm_array))]
            fig_cm = px.imshow(cm_array, text_auto=True, aspect='auto',
                               x=labels_str, y=labels_str,
                               labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
                               color_continuous_scale='Blues',
                               title=f'Confusion Matrix — {best}')
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── ROC Curve ──
        roc_data = best_metrics.get('roc')
        if roc_data and 'fpr' in roc_data and 'tpr' in roc_data:
            st.subheader(f"📈 ROC Curve — {best} (AUC = {roc_data['auc']:.4f})")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                                         mode='lines', name=f"{best} (AUC={roc_data['auc']:.4f})",
                                         line=dict(color='#3b82f6', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                         name='Random', line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate',
                                  title=f'ROC Curve — {best}')
            st.plotly_chart(fig_roc, use_container_width=True)
        elif roc_data and 'auc' in roc_data:
            st.info(f"**ROC AUC (weighted, multi-class OVR):** {roc_data['auc']:.4f}")

    # ── Feature Importance ──
    if r.get('feature_importance'):
        st.subheader("🎯 Feature Importance")
        fi = r['feature_importance']
        fi_df = pd.DataFrame(list(fi.items())[:15], columns=['Feature', 'Importance'])
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()
