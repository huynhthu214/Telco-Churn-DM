"""
Customer Churn Analytics - dashboard demo (Streamlit)

3 tab ngang:
  1. Tổng quan            : biểu đồ churn, các yếu tố liên quan, so sánh 3 mô hình
  2. Dự đoán khách hàng   : nhập thông tin 1 khách hàng -> điểm rủi ro của 3 mô hình
  3. Quyết định can thiệp : so sánh 3 kịch bản (không can thiệp / đại trà / theo mô hình)

Chạy:  python -m streamlit run app.py
Cấu trúc thư mục cần có (cạnh app.py hoặc ở thư mục cha):
  data/WA_Fn-UseC_-Telco-Customer-Churn.csv
  models/knn.pkl, dt.pkl, nb.pkl, locked_thresholds.pkl, locked_net_benefit_thresholds.pkl
  .streamlit/config.toml   (tùy chọn, đặt màu chủ đạo của giao diện)
"""
import os

# Giống notebook 02: cố định số luồng để KNN tái lập kết quả ở các điểm đồng hạng.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

threadpool_limits(limits=1)

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Hằng số lấy từ notebook 02_Modeling_Models
# ----------------------------------------------------------------------------
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILES = {"KNN": "knn.pkl", "Decision Tree": "dt.pkl", "Naive Bayes": "nb.pkl"}
THRESHOLD_KEY = {"KNN": "KNN", "Decision Tree": "DT", "Naive Bayes": "NB"}

# Risk x Value (mục 11 notebook): cutoff khóa trên OOF / training
RISK_CUTOFF = 0.659505   # phân vị 70% điểm rủi ro OOF của Decision Tree
VALUE_CUTOFF = 70.50     # median MonthlyCharges của tập training

# Giả định Net Benefit gốc (mục 2 notebook) và kết quả test đã kiểm chứng
DEFAULT_COST, DEFAULT_VALUE, DEFAULT_SUCCESS = 10, 100, 0.30
NOTEBOOK_NET_BENEFIT = {"KNN": 2960, "Decision Tree": 3090, "Naive Bayes": 2620}

# Màu theo thực thể (không đổi giữa các biểu đồ)
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
C_NAVY = "#12355b"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
SERVICES = {
    "OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection", "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies",
}
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)"]

PRESETS = {
    "risky": dict(gender="Female", senior="No", partner="No", dependents="No",
                  phone="Yes", ml="No", internet="Fiber optic",
                  svc_OnlineSecurity="No", svc_OnlineBackup="No",
                  svc_DeviceProtection="No", svc_TechSupport="No",
                  svc_StreamingTV="Yes", svc_StreamingMovies="Yes",
                  contract="Month-to-month", paperless="Yes",
                  payment="Electronic check", tenure=2, monthly=89.9),
    "safe": dict(gender="Male", senior="No", partner="Yes", dependents="Yes",
                 phone="Yes", ml="Yes", internet="DSL",
                 svc_OnlineSecurity="Yes", svc_OnlineBackup="Yes",
                 svc_DeviceProtection="Yes", svc_TechSupport="Yes",
                 svc_StreamingTV="No", svc_StreamingMovies="No",
                 contract="Two year", paperless="No",
                 payment="Credit card (automatic)", tenure=60, monthly=64.5),
}

# ----------------------------------------------------------------------------
# Giao diện (CSS) và các thành phần HTML
# ----------------------------------------------------------------------------
CSS = """
<style>
:root{--navy:#12355b;--navy2:#1d4f87;--ink:#1a2433;--muted:#5b6b7f;--line:#e3e8ef;--soft:#f4f6f9;}
html, body, .stApp, [class*="css"]{font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;}
.stApp{background:#ffffff;color:var(--ink);}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]{display:none !important;}
.block-container{max-width:1240px;padding-top:1.4rem;padding-bottom:3rem;}

.banner{background:linear-gradient(120deg,#12355b 0%,#1d4f87 100%);color:#fff;border-radius:10px;
  padding:26px 32px;margin-bottom:6px;}
.banner-eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;opacity:.75;font-weight:600;}
.banner-title{font-size:1.75rem;font-weight:700;line-height:1.25;margin:6px 0 4px 0;}
.banner-sub{font-size:.95rem;opacity:.85;}

.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);margin-bottom:6px;}
.stTabs [data-baseweb="tab"]{height:48px;padding:0 20px;}
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p{
  font-size:1.05rem !important;font-weight:600 !important;color:var(--muted);}
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"] p{
  color:var(--navy) !important;}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--navy);height:3px;}

.section{margin:22px 0 10px 0;padding-left:12px;border-left:4px solid var(--navy);}
.section h3{margin:0;padding:0;font-size:1.2rem;font-weight:700;color:var(--ink);}
.section p{margin:2px 0 0 0;color:var(--muted);font-size:.9rem;}

.kpi{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 18px;height:100%;}
.kpi-label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600;}
.kpi-value{font-size:1.9rem;font-weight:700;color:var(--navy);line-height:1.2;margin-top:2px;}
.kpi-note{font-size:.8rem;color:var(--muted);margin-top:2px;}

.card-title{font-weight:700;color:var(--ink);font-size:.98rem;margin-bottom:2px;}
.card-sub{color:var(--muted);font-size:.82rem;margin-bottom:4px;}
.form-h{font-weight:700;color:var(--navy);font-size:.9rem;text-transform:uppercase;letter-spacing:.06em;
  border-bottom:1px solid var(--line);padding-bottom:4px;margin-bottom:8px;}

.callout{border-radius:8px;padding:12px 16px;margin:8px 0;font-size:.92rem;line-height:1.55;
  border:1px solid var(--line);border-left-width:4px;background:var(--soft);color:var(--ink);}
.callout.info{border-left-color:var(--navy2);}
.callout.ok{border-left-color:#1b8a5a;background:#f1faf5;}
.callout.warn{border-left-color:#c98500;background:#fff9ec;}
.callout.bad{border-left-color:#c0392b;background:#fdf3f2;}
.callout ul{margin:4px 0 0 0;padding-left:20px;}
.callout b{color:var(--ink);}

.res-card{border:1px solid var(--line);border-radius:10px;padding:16px 18px;background:#fff;}
.res-model{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700;}
.res-score{font-size:2.3rem;font-weight:700;color:var(--navy);line-height:1.15;}
.res-bar{height:8px;background:#e9edf3;border-radius:4px;margin:8px 0 12px 0;overflow:hidden;}
.res-fill{height:100%;border-radius:4px;}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:.82rem;font-weight:700;}
.badge.bad{background:#fde7e3;color:#a12d1f;}
.badge.ok{background:#e1f4ea;color:#17673f;}
.res-note{font-size:.78rem;color:var(--muted);margin-top:8px;}

.scn{border:1px solid var(--line);border-radius:10px;padding:16px 18px;background:#fff;height:100%;position:relative;}
.scn.best{border:2px solid var(--navy);background:#f6f9fd;}
.scn-tag{position:absolute;top:-11px;right:14px;background:var(--navy);color:#fff;font-size:.7rem;
  font-weight:700;letter-spacing:.06em;padding:3px 10px;border-radius:999px;text-transform:uppercase;}
.scn-name{font-weight:700;color:var(--ink);font-size:.98rem;}
.scn-value{font-size:2rem;font-weight:700;line-height:1.2;margin:4px 0;}
.scn-value.pos{color:#17673f;} .scn-value.neg{color:#a12d1f;} .scn-value.zero{color:var(--muted);}
.scn-lines{font-size:.82rem;color:var(--muted);line-height:1.6;}

button[kind="primary"], button[data-testid="stBaseButton-primary"]{background:var(--navy) !important;
  border-color:var(--navy) !important;color:#fff !important;font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def section(title, desc=""):
    st.markdown(f'<div class="section"><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True)


def kpi(label, value, note=""):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>')


def callout(html, kind="info"):
    st.markdown(f'<div class="callout {kind}">{html}</div>', unsafe_allow_html=True)


def card_title(title, sub=""):
    st.markdown(f'<div class="card-title">{title}</div>'
                + (f'<div class="card-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Nạp dữ liệu và mô hình
# ----------------------------------------------------------------------------
def find_root():
    here = Path(__file__).resolve().parent
    for base in [here, *here.parents]:
        if (base / "models" / "dt.pkl").is_file() and (base / "data" / CSV_NAME).is_file():
            return base
    return None


ROOT = find_root()
if ROOT is None:
    st.error(
        "Không tìm thấy thư mục dự án. Cần có `models/dt.pkl` và "
        f"`data/{CSV_NAME}` cạnh `app.py` hoặc ở thư mục cha."
    )
    st.stop()


@st.cache_data
def load_data():
    raw = pd.read_csv(ROOT / "data" / CSV_NAME)
    # Giống clean_raw_features ở notebook 01: ô trống = khách mới -> 0
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
    raw["Churn"] = raw["Churn"].map({"No": 0, "Yes": 1})
    return raw.drop(columns=["customerID"])


@st.cache_resource
def load_models():
    models = {n: joblib.load(ROOT / "models" / f) for n, f in MODEL_FILES.items()}
    thr_cost = joblib.load(ROOT / "models" / "locked_thresholds.pkl")
    thr_nb = joblib.load(ROOT / "models" / "locked_net_benefit_thresholds.pkl")
    return models, thr_cost, thr_nb


@st.cache_data
def test_scores():
    """Điểm rủi ro của 3 mô hình trên đúng tập test (split 80/20, seed 42)."""
    df = load_data()
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    models, _, _ = load_models()
    scores = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    return scores, y_test.to_numpy()


def confusion(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def net_benefit(tp, fp, cost, value, success):
    return tp * value * success - (tp + fp) * cost


def style_fig(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), bargap=0.45,
                      showlegend=False, font=dict(family=FONT, size=12, color="#1a2433"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def dt_rules(pipe, row_df):
    """Đường đi của 1 khách hàng trong Decision Tree, in thành luật đọc được."""
    pre, tree = pipe.named_steps["preprocessor"], pipe.named_steps["clf"]
    xt = pre.transform(row_df)
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    names = pre.get_feature_names_out()
    scaler = pre.named_transformers_["num"]
    num_cols = list(pre.transformers_[0][2])
    t = tree.tree_
    rules = []
    for node in tree.decision_path(xt).indices:
        if t.children_left[node] == t.children_right[node]:
            continue  # nút lá
        f = int(t.feature[node])
        go_left = xt[0, f] <= t.threshold[node]
        col = names[f].split("__", 1)[1]
        if names[f].startswith("num__"):
            i = num_cols.index(col)
            thr = t.threshold[node] * scaler.scale_[i] + scaler.mean_[i]
            rules.append(f"{col} {'≤' if go_left else '>'} {thr:.1f}")
        else:
            rules.append(f"{col.replace('_', ' = ', 1)}: {'Không' if go_left else 'Có'}")
    return rules


# ----------------------------------------------------------------------------
# Tab 1 - Tổng quan
# ----------------------------------------------------------------------------
def rate_bar(df, col, order=None):
    g = df.groupby(col, observed=True)["Churn"].agg(["mean", "count"]).reset_index()
    g["rate"] = g["mean"] * 100
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    overall = df["Churn"].mean() * 100
    fig = go.Figure(go.Bar(
        x=g[col].astype(str), y=g["rate"], marker_color=C_ORANGE,
        text=[f"{r:.1f}%" for r in g["rate"]], textposition="outside",
        customdata=g["count"],
        hovertemplate="%{x}<br>Churn: %{y:.1f}%<br>%{customdata:,} khách<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dot", line_color=C_GRAY,
                  annotation_text=f"Trung bình {overall:.1f}%",
                  annotation_position="top right")
    fig.update_yaxes(range=[0, max(60, g["rate"].max() * 1.25)], ticksuffix="%")
    return style_fig(fig, 290)


def page_overview():
    df = load_data()
    models, _, _ = load_models()
    n_yes = int(df["Churn"].sum())

    section("Tổng quan danh mục khách hàng")
    k = st.columns(4)
    k[0].markdown(kpi("Số khách hàng", f"{len(df):,}", "Toàn bộ bộ dữ liệu"), unsafe_allow_html=True)
    k[1].markdown(kpi("Tỉ lệ churn", f"{df['Churn'].mean() * 100:.1f}%",
                      f"{n_yes:,} khách đã rời bỏ"), unsafe_allow_html=True)
    k[2].markdown(kpi("Tenure trung bình", f"{df['tenure'].mean():.1f} tháng",
                      f"Trung vị {df['tenure'].median():.0f} tháng"), unsafe_allow_html=True)
    k[3].markdown(kpi("Cước tháng trung bình", f"${df['MonthlyCharges'].mean():.2f}",
                      f"Trung vị ${df['MonthlyCharges'].median():.2f}"), unsafe_allow_html=True)

    def rate(mask):
        return df.loc[mask, "Churn"].mean() * 100
    m2m, two = rate(df["Contract"] == "Month-to-month"), rate(df["Contract"] == "Two year")
    fib, dsl = rate(df["InternetService"] == "Fiber optic"), rate(df["InternetService"] == "DSL")
    new, old = rate(df["tenure"] <= 12), rate(df["tenure"] >= 49)
    st.write("")
    callout(
        "<b>Nhận định chính từ dữ liệu</b><ul>"
        f"<li>Khách dùng hợp đồng <b>Month-to-month</b> có tỉ lệ churn {m2m:.1f}%, "
        f"so với {two:.1f}% ở hợp đồng Two year.</li>"
        f"<li>Khách dùng <b>Fiber optic</b> churn {fib:.1f}%, cao hơn DSL ({dsl:.1f}%).</li>"
        f"<li>Khách mới (tenure ≤ 12 tháng) churn {new:.1f}%, trong khi nhóm gắn bó từ 49 tháng "
        f"chỉ còn {old:.1f}%.</li></ul>", "info")

    left, right = st.columns(2)
    with left, st.container(border=True):
        card_title("Phân bố churn", "Số khách hàng theo trạng thái")
        n_no = len(df) - n_yes
        fig = go.Figure(go.Bar(
            x=["Không churn", "Churn"], y=[n_no, n_yes],
            marker_color=[C_BLUE, C_ORANGE],
            text=[f"{n_no:,} ({n_no / len(df) * 100:.1f}%)",
                  f"{n_yes:,} ({n_yes / len(df) * 100:.1f}%)"],
            textposition="outside", hovertemplate="%{x}: %{y:,} khách<extra></extra>"))
        fig.update_yaxes(range=[0, n_no * 1.15])
        st.plotly_chart(style_fig(fig, 300))
    with right, st.container(border=True):
        card_title("Top 10 đặc trưng quan trọng nhất", "Theo Decision Tree (mức giảm impurity)")
        pipe = models["Decision Tree"]
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        imp = (pd.Series(pipe.named_steps["clf"].feature_importances_,
                         index=[n.split("__", 1)[1] for n in names])
               .sort_values(ascending=False).head(10).iloc[::-1])
        fig = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation="h", marker_color=C_BLUE,
            text=[f"{v:.3f}" for v in imp.values], textposition="outside",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        style_fig(fig, 300)
        fig.update_xaxes(range=[0, imp.max() * 1.2], showgrid=True,
                         gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

    section("Tỉ lệ churn theo các yếu tố chính",
            "Đường chấm là tỉ lệ churn trung bình toàn bộ dữ liệu.")
    d = df.copy()
    d["Tenure_Group"] = pd.cut(d["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-12 tháng", "13-24 tháng", "25-48 tháng", "49-72 tháng"])
    d["Monthly_Quartile"] = pd.qcut(d["MonthlyCharges"], q=4,
                                    labels=["Q1 (thấp nhất)", "Q2", "Q3", "Q4 (cao nhất)"])
    charts = [
        ("Loại hợp đồng", "Contract", CONTRACTS),
        ("Dịch vụ Internet", "InternetService", ["Fiber optic", "DSL", "No"]),
        ("Thời gian gắn bó (tenure)", "Tenure_Group", None),
        ("Cước hàng tháng (tứ phân vị)", "Monthly_Quartile", None),
    ]
    for pair in (charts[:2], charts[2:]):
        for col, (title, name, order) in zip(st.columns(2), pair):
            with col, st.container(border=True):
                card_title(title)
                st.plotly_chart(rate_bar(d, name, order))

    section("Hiệu năng 3 mô hình trên tập test",)
    scores, y = test_scores()
    rows = []
    for name, p in scores.items():
        pred = (p >= 0.5).astype(int)
        rows.append({"Mô hình": name, "Accuracy": accuracy_score(y, pred),
                     "Precision": precision_score(y, pred), "Recall": recall_score(y, pred),
                     "F1": f1_score(y, pred), "ROC-AUC": roc_auc_score(y, p)})
    comp = pd.DataFrame(rows)
    a, b = st.columns(2)
    with a, st.container(border=True):
        card_title("Bảng chỉ số")
        st.dataframe(comp.round(4), hide_index=True)
        callout("Decision Tree là mô hình khuyến nghị chính theo F1. KNN cao nhất về "
                "Accuracy, Precision và AUC. Naive Bayes cao nhất về Recall nhưng đánh đổi "
                "nhiều cảnh báo sai.", "info")
    with b, st.container(border=True):
        card_title("Precision, Recall và F1")
        fig = go.Figure()
        for metric, color in [("Precision", C_BLUE), ("Recall", C_ORANGE), ("F1", C_AQUA)]:
            fig.add_bar(name=metric, x=comp["Mô hình"], y=comp[metric], marker_color=color,
                        text=[f"{v:.2f}" for v in comp[metric]], textposition="outside")
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1.05])
        style_fig(fig, 290)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14))
        st.plotly_chart(fig)


# ----------------------------------------------------------------------------
# Tab 2 - Dự đoán khách hàng
# ----------------------------------------------------------------------------
def apply_preset(name):
    st.session_state.update(PRESETS[name])


def page_predict():
    models, _, thr_nb = load_models()
    section("Dự đoán rủi ro churn theo từng khách hàng")

    for key, v in PRESETS["risky"].items():
        st.session_state.setdefault(key, v)

    yn = ["No", "Yes"]
    with st.container(border=True):
        pb1, pb2, _ = st.columns([1.2, 1.2, 3])
        pb1.button("Điền mẫu: khách rủi ro cao", on_click=apply_preset, args=("risky",))
        pb2.button("Điền mẫu: khách an toàn", on_click=apply_preset, args=("safe",))
        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="form-h">Thông tin cá nhân</div>', unsafe_allow_html=True)
            st.selectbox("Giới tính", ["Female", "Male"], key="gender")
            st.selectbox("Người cao tuổi (Senior)", yn, key="senior")
            st.selectbox("Có bạn đời (Partner)", yn, key="partner")
            st.selectbox("Có người phụ thuộc (Dependents)", yn, key="dependents")
        with c2:
            st.markdown('<div class="form-h">Dịch vụ sử dụng</div>', unsafe_allow_html=True)
            st.selectbox("Dịch vụ điện thoại (Phone)", yn, key="phone")
            if st.session_state["phone"] == "No":
                st.selectbox("Nhiều đường dây (Multiple Lines)", ["No phone service"],
                             disabled=True, key="off_ml")
                multiple_lines = "No phone service"
            else:
                multiple_lines = st.selectbox("Nhiều đường dây (Multiple Lines)", yn, key="ml")
            st.selectbox("Dịch vụ Internet", ["DSL", "Fiber optic", "No"], key="internet")
            no_internet = st.session_state["internet"] == "No"
            service_values = {}
            for col, label in SERVICES.items():
                if no_internet:
                    st.selectbox(label, ["No internet service"], disabled=True, key=f"off_{col}")
                    service_values[col] = "No internet service"
                else:
                    service_values[col] = st.selectbox(label, yn, key=f"svc_{col}")
        with c3:
            st.markdown('<div class="form-h">Hợp đồng và thanh toán</div>', unsafe_allow_html=True)
            st.selectbox("Loại hợp đồng", CONTRACTS, key="contract")
            st.selectbox("Hóa đơn điện tử (Paperless)", yn, key="paperless")
            st.selectbox("Phương thức thanh toán", PAYMENTS, key="payment")
            st.slider("Thời gian gắn bó (tenure, tháng)", 0, 72, key="tenure")
            st.number_input("Cước hàng tháng (MonthlyCharges, $)", min_value=18.0,
                            max_value=120.0, step=0.05, key="monthly")
            total = round(st.session_state["tenure"] * st.session_state["monthly"], 2)
            st.caption(f"TotalCharges = tenure × cước tháng = ${total:,.2f}")

        record = {
            "gender": st.session_state["gender"],
            "SeniorCitizen": 1 if st.session_state["senior"] == "Yes" else 0,
            "Partner": st.session_state["partner"],
            "Dependents": st.session_state["dependents"],
            "tenure": st.session_state["tenure"],
            "PhoneService": st.session_state["phone"],
            "MultipleLines": multiple_lines,
            "InternetService": st.session_state["internet"],
            **service_values,
            "Contract": st.session_state["contract"],
            "PaperlessBilling": st.session_state["paperless"],
            "PaymentMethod": st.session_state["payment"],
            "MonthlyCharges": float(st.session_state["monthly"]),
            "TotalCharges": float(total),
        }
        row = pd.DataFrame([record])[FEATURE_ORDER]
        st.write("")
        if st.button("Dự đoán", type="primary"):
            st.session_state["snap"] = {"row": row, "record": record}

    snap = st.session_state.get("snap")
    if snap is None:
        callout("Chọn thông tin khách hàng rồi bấm <b>Dự đoán</b> để xem kết quả.", "info")
        return
    if not snap["row"].equals(row):
        callout("Thông tin trên form đã thay đổi so với kết quả bên dưới. "
                "Bấm <b>Dự đoán</b> để cập nhật.", "warn")
    row, record = snap["row"], snap["record"]

    scores = {n: float(m.predict_proba(row)[0, 1]) for n, m in models.items()}
    flags = {n: scores[n] >= thr_nb[THRESHOLD_KEY[n]] for n in scores}

    section("Kết quả dự đoán", "Khuyến nghị dựa trên ngưỡng Net Benefit đã khóa của từng mô hình.")
    for col, name in zip(st.columns(3), scores):
        s, thr = scores[name], thr_nb[THRESHOLD_KEY[name]]
        fill = C_ORANGE if flags[name] else C_BLUE
        badge = ('<span class="badge bad">Nên can thiệp giữ chân</span>' if flags[name]
                 else '<span class="badge ok">Chưa cần can thiệp</span>')
        col.markdown(
            f'<div class="res-card"><div class="res-model">{name}</div>'
            f'<div class="res-score">{s:.2f}</div>'
            f'<div class="res-bar"><div class="res-fill" style="width:{min(max(s, 0), 1) * 100:.0f}%;'
            f'background:{fill}"></div></div>{badge}'
            f'<div class="res-note">Ngưỡng Net Benefit đã khóa: {thr:.2f}</div></div>',
            unsafe_allow_html=True)

    n_flag = sum(flags.values())
    st.write("")
    if n_flag == 3:
        callout("<b>Cả 3 mô hình đều cảnh báo rủi ro churn cao.</b>", "bad")
    elif n_flag == 0:
        callout("<b>Cả 3 mô hình đều đánh giá rủi ro thấp.</b>", "ok")
    else:
        callout(f"<b>{n_flag}/3 mô hình cảnh báo rủi ro cao.</b> Kết quả chưa đồng thuận, "
                "nên xem xét thêm theo nghiệp vụ.", "warn")

    if record["Contract"] != "Month-to-month" and not flags["Decision Tree"]:
        callout("<b>Lưu ý:</b> khách có hợp đồng 1-2 năm. Trên tập test, Decision Tree đã bỏ sót "
                "45/45 khách churn thuộc nhóm hợp đồng dài hạn, nên điểm thấp ở đây chưa chắc "
                "đáng tin, cần rà soát thủ công.", "warn")

    risk_high = scores["Decision Tree"] >= RISK_CUTOFF
    value_high = record["MonthlyCharges"] >= VALUE_CUTOFF
    seg_name, seg_note = {
        (True, True): ("High Risk - High Value", "Ưu tiên cho chương trình giữ chân có mục tiêu."),
        (True, False): ("High Risk - Low Value", "Cân nhắc chi phí can thiệp so với giá trị khách hàng."),
        (False, True): ("Low Risk - High Value", "Duy trì trải nghiệm và loyalty, chưa cần can thiệp khẩn cấp."),
        (False, False): ("Low Risk - Low Value", "Chăm sóc chi phí thấp hoặc tự động hóa."),
    }[(risk_high, value_high)]
    callout(f"<b>Phân khúc Risk × Value: {seg_name}.</b> {seg_note}<br>"
            f"<span style='color:#5b6b7f;font-size:.82rem'>Risk: điểm Decision Tree ≥ {RISK_CUTOFF:.2f}. "
            f"Value: cước tháng ≥ ${VALUE_CUTOFF:.2f} (proxy, không phải CLV).</span>", "info")

# ----------------------------------------------------------------------------
# Tab 3 - Quyết định can thiệp
# ----------------------------------------------------------------------------
def page_decision():
    models, _, thr_nb = load_models()
    scores, y = test_scores()

    section("So sánh 3 kịch bản hỗ trợ khách hàng", "Giả định chi phí và giá trị giữ chân khách hàng để tính Net Benefit.")
    with st.container(border=True):
        card_title("Giả định kịch bản")
        a, b, c = st.columns(3)
        cost = a.slider("Chi phí liên hệ mỗi khách", 1, 50, DEFAULT_COST)
        value = b.slider("Giá trị giữ được mỗi khách", 10, 500, DEFAULT_VALUE, step=5)
        success = c.slider("Tỉ lệ giữ chân thành công (%)", 5, 80,
                           int(DEFAULT_SUCCESS * 100)) / 100
        model_name = st.radio("Mô hình dùng cho kịch bản 3", list(MODEL_FILES), index=1,
                              horizontal=True)

    n, pos = len(y), int(y.sum())
    thr = thr_nb[THRESHOLD_KEY[model_name]]
    tp, fp, fn, tn = confusion(y, scores[model_name], thr)
    scen = {
        "Không can thiệp": dict(contacted=0, tp=0, fp=0, fn=pos, color=C_GRAY),
        "Can thiệp đại trà": dict(contacted=n, tp=pos, fp=n - pos, fn=0, color=C_ORANGE),
        f"Theo mô hình ({model_name})": dict(contacted=tp + fp, tp=tp, fp=fp, fn=fn, color=C_BLUE),
    }
    for s in scen.values():
        s["cost"] = s["contacted"] * cost
        s["saved"] = s["tp"] * value * success
        s["net"] = s["saved"] - s["cost"]
    names = list(scen)
    nets = [scen[k]["net"] for k in names]
    best = names[int(np.argmax(nets))] if max(nets) > 0 else names[0]

    st.write("")
    for col, k in zip(st.columns(3), names):
        s = scen[k]
        cls = "pos" if s["net"] > 0 else ("neg" if s["net"] < 0 else "zero")
        tag = '<div class="scn-tag">Khuyến nghị</div>' if k == best else ""
        col.markdown(
            f'<div class="scn {"best" if k == best else ""}">{tag}'
            f'<div class="scn-name">{k}</div>'
            f'<div class="scn-value {cls}">{s["net"]:,.0f}</div>'
            f'<div class="scn-lines">Net Benefit (đơn vị tiền giả định)<br>'
            f'Liên hệ {s["contacted"]:,} khách · TP {s["tp"]:,} · FP {s["fp"]:,}<br>'
            f'Chi phí {s["cost"]:,.0f} · Giá trị giữ được {s["saved"]:,.0f}</div></div>',
            unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        card_title("Lợi ích ròng theo kịch bản")
        fig = go.Figure(go.Bar(
            x=names, y=nets, marker_color=[scen[k]["color"] for k in names],
            text=[f"{v:,.0f}" for v in nets], textposition="outside",
            hovertemplate="%{x}<br>Net Benefit: %{y:,.0f}<extra></extra>"))
        fig.add_hline(y=0, line_color=C_GRAY)
        span = max(abs(v) for v in nets) or 1
        fig.update_yaxes(range=[min(0, min(nets)) - span * 0.2, max(0, max(nets)) + span * 0.2])
        st.plotly_chart(style_fig(fig, 320))

    if max(nets) <= 0:
        callout("<b>Kết luận:</b> với giả định hiện tại, không kịch bản can thiệp nào có lãi. "
                "Nên <b>không can thiệp</b>.", "warn")
    elif best.startswith("Theo mô hình"):
        callout(f"<b>Kết luận: chọn {best}.</b> Net Benefit cao nhất, chỉ liên hệ {tp + fp:,} khách "
                f"thay vì {n:,} khách như can thiệp đại trà.", "ok")
    else:
        callout(f"<b>Kết luận: chọn {best}</b> với giả định hiện tại.", "ok")

    churn_rate = pos / n
    precision = tp / (tp + fp) if tp + fp else 0.0
    be_success = cost / (value * churn_rate)
    be_precision = cost / (value * success)
    callout("<b>Điểm hòa vốn</b><ul>"
            f"<li>Can thiệp đại trà chỉ có lãi khi tỉ lệ giữ chân thành công lớn hơn "
            f"<b>{be_success * 100:.1f}%</b> (hiện tại {success * 100:.0f}%).</li>"
            f"<li>Can thiệp theo mô hình có lãi khi Precision lớn hơn "
            f"<b>{be_precision * 100:.1f}%</b> ({model_name} đạt {precision * 100:.1f}%).</li></ul>",
            "info")

    section("Chi tiết các kịch bản")
    table = pd.DataFrame([{
        "Kịch bản": k, "Khách được liên hệ": s["contacted"], "TP (churn được tiếp cận)": s["tp"],
        "FP (liên hệ thừa)": s["fp"], "FN (bỏ sót)": s["fn"],
        "Chi phí": s["cost"], "Giá trị giữ được": round(s["saved"], 1),
        "Net Benefit": round(s["net"], 1)} for k, s in scen.items()])
    st.dataframe(table, hide_index=True)

    with st.expander("Kiểm tra tái lập với notebook (giả định gốc 10 / 100 / 30%)"):
        rows = []
        for name in MODEL_FILES:
            t_tp, t_fp, _, _ = confusion(y, scores[name], thr_nb[THRESHOLD_KEY[name]])
            got = net_benefit(t_tp, t_fp, DEFAULT_COST, DEFAULT_VALUE, DEFAULT_SUCCESS)
            exp = NOTEBOOK_NET_BENEFIT[name]
            rows.append({"Mô hình": name, "Ngưỡng": round(thr_nb[THRESHOLD_KEY[name]], 2),
                         "Net Benefit app": round(got), "Net Benefit notebook": exp,
                         "Khớp": "✓" if round(got) == exp else "✗ (kiểm tra phiên bản thư viện)"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ----------------------------------------------------------------------------
# Bố cục chính
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="banner"><div class="banner-eyebrow">Báo cáo phân tích khách hàng · Data Mining</div>'
    '<div class="banner-title">Phân tích và dự đoán khách hàng rời bỏ dịch vụ</div>'
    '<div class="banner-sub">Telco Customer Churn · KNN · Decision Tree · Naive Bayes</div></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Tổng quan", "Dự đoán khách hàng", "Quyết định can thiệp"])
with tab1:
    page_overview()
with tab2:
    page_predict()
with tab3:
    page_decision()