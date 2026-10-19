"""
Kiểm thử tự động cho dashboard Customer Churn.

Chạy (trong thư mục gốc dự án, đã cài requirements.txt):
    python -m pytest test_app.py -v

Các test cần data/ và models/ thật sẽ tự bỏ qua (skip) nếu thiếu file.
"""
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
CSV = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS = ROOT / "models"
TIMEOUT = 180

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
APP_SOURCE = APP.read_text(encoding="utf-8")

has_artifacts = CSV.is_file() and all(
    (MODELS / f).is_file() for f in
    ["knn.pkl", "dt.pkl", "nb.pkl", "locked_thresholds.pkl",
     "locked_net_benefit_thresholds.pkl"])
needs_artifacts = pytest.mark.skipif(
    not has_artifacts, reason="Thiếu data/ hoặc models/ (chạy notebook 01 và 02 trước)")

PRESET_RISKY = "Điền mẫu: khách rủi ro cao"
PRESET_SAFE = "Điền mẫu: khách an toàn"
PREDICT = "Dự đoán"


# ----------------------------------------------------------------------------
# Hàm hỗ trợ
# ----------------------------------------------------------------------------
def new_app():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def html_blocks(at):
    return [m.value for m in at.markdown]


def count_html(at, marker):
    return sum(marker in v for v in html_blocks(at))


# ----------------------------------------------------------------------------
# 1. config.json
# ----------------------------------------------------------------------------
def test_config_has_required_keys():
    for model in ("KNN", "DT", "NB"):
        assert 0 < CONFIG["cost_sensitive_thresholds"][model] < 1
        assert 0 < CONFIG["net_benefit_thresholds"][model] < 1
    assump = CONFIG["business_assumptions"]
    assert assump["C"] > 0 and assump["h"] > 0 and 0 < assump["s"] <= 1
    seg = CONFIG["risk_value_segmentation"]
    assert 0 < seg["risk_threshold"] < 1 and seg["value_threshold"] > 0
    assert CONFIG["split"] == {"test_size": 0.2, "random_state": 42, "stratify": "Churn"}


def test_config_matches_app_constants():
    """Hằng số trong app.py phải khớp config.json."""
    risk = float(re.search(r"RISK_CUTOFF\s*=\s*([\d.]+)", APP_SOURCE).group(1))
    value = float(re.search(r"VALUE_CUTOFF\s*=\s*([\d.]+)", APP_SOURCE).group(1))
    cost, h, s = map(float, re.search(
        r"DEFAULT_COST, DEFAULT_VALUE, DEFAULT_SUCCESS\s*=\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)",
        APP_SOURCE).groups())
    seg, assump = CONFIG["risk_value_segmentation"], CONFIG["business_assumptions"]
    assert risk == pytest.approx(seg["risk_threshold"])
    assert value == pytest.approx(seg["value_threshold"])
    assert (cost, h, s) == pytest.approx((assump["C"], assump["h"], assump["s"]))

    block = re.search(r"NOTEBOOK_NET_BENEFIT\s*=\s*\{([^}]*)\}", APP_SOURCE).group(1)
    in_app = {k: int(v) for k, v in re.findall(r'"([^"]+)":\s*(\d+)', block)}
    expected = CONFIG["notebook_test_net_benefit"]
    assert in_app == {"KNN": expected["KNN"], "Decision Tree": expected["DT"],
                      "Naive Bayes": expected["NB"]}


@needs_artifacts
def test_config_matches_locked_pickles():
    """Ngưỡng trong config.json phải khớp các file .pkl đã khóa ở notebook."""
    cost = joblib.load(MODELS / "locked_thresholds.pkl")
    nb = joblib.load(MODELS / "locked_net_benefit_thresholds.pkl")
    for model in ("KNN", "DT", "NB"):
        assert cost[model] == pytest.approx(
            CONFIG["cost_sensitive_thresholds"][model], abs=1e-6)
        assert nb[model] == pytest.approx(
            CONFIG["net_benefit_thresholds"][model], abs=1e-6)


# ----------------------------------------------------------------------------
# 2. Tái lập kết quả notebook
# ----------------------------------------------------------------------------
@needs_artifacts
def test_net_benefit_reproduces_notebook():
    """Net Benefit trên tập test phải đúng 2960 / 3090 / 2620 như notebook 02."""
    from threadpoolctl import threadpool_limits
    df = pd.read_csv(CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    X, y = df.drop(columns=["customerID", "Churn"]), df["Churn"]
    split = CONFIG["split"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=split["test_size"], stratify=y, random_state=split["random_state"])

    a = CONFIG["business_assumptions"]
    files = {"KNN": "knn.pkl", "DT": "dt.pkl", "NB": "nb.pkl"}
    with threadpool_limits(limits=1):
        for model, filename in files.items():
            pipe = joblib.load(MODELS / filename)
            pred = pipe.predict_proba(X_test)[:, 1] >= CONFIG["net_benefit_thresholds"][model]
            tp = int(((pred == 1) & (y_test.to_numpy() == 1)).sum())
            fp = int(((pred == 1) & (y_test.to_numpy() == 0)).sum())
            net = tp * a["h"] * a["s"] - (tp + fp) * a["C"]
            assert round(net) == CONFIG["notebook_test_net_benefit"][model], model


# ----------------------------------------------------------------------------
# 3. Giao diện Streamlit
# ----------------------------------------------------------------------------
@needs_artifacts
def test_app_loads_with_three_tabs():
    at = new_app()
    assert [t.label for t in at.tabs] == [
        "Tổng quan", "Dự đoán khách hàng", "Quyết định can thiệp"]


@needs_artifacts
@pytest.mark.parametrize("preset", [PRESET_RISKY, PRESET_SAFE])
def test_predict_shows_three_model_cards(preset):
    at = new_app()
    click(at, preset)
    click(at, PREDICT)
    assert count_html(at, 'class="res-card"') == 3
    assert any("Phân khúc Risk" in v for v in html_blocks(at))


@needs_artifacts
def test_predict_edge_cases_without_internet_and_phone():
    """Khách không dùng Internet và điện thoại: form phải tự đặt giá trị hợp lệ."""
    at = new_app()
    at.selectbox(key="internet").set_value("No").run()
    at.selectbox(key="phone").set_value("No").run()
    assert not at.exception, [e.value for e in at.exception]
    click(at, PREDICT)
    assert count_html(at, 'class="res-card"') == 3


@needs_artifacts
def test_predict_new_customer_with_zero_tenure():
    """Khách mới (tenure = 0, TotalCharges = 0) vẫn dự đoán được."""
    at = new_app()
    at.slider(key="tenure").set_value(0).run()
    click(at, PREDICT)
    assert count_html(at, 'class="res-card"') == 3


@needs_artifacts
def test_decision_tab_has_three_scenarios_for_each_model():
    at = new_app()
    for model in ("KNN", "Decision Tree", "Naive Bayes"):
        at.radio[0].set_value(model).run()
        assert not at.exception, [e.value for e in at.exception]
        assert count_html(at, 'class="scn') == 3
        assert any(f"Theo mô hình ({model})" in v for v in html_blocks(at))