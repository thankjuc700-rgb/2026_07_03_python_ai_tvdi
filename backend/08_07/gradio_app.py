# -*- coding: utf-8 -*-
"""
薪資預測系統 - Gradio 前端
==========================
美觀響應式介面，支援手機瀏覽。

執行方式:
    uv run gradio_app.py

功能:
    - 薪資預測: 輸入工作經驗 / 學歷 / 城市，即時預測月薪與年薪
    - 模型訓練: 支援 LinearRegression / Lasso / Ridge 線上重新訓練
"""
import os
import joblib
import pandas as pd
import gradio as gr

from train_save import train_and_save_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "salary_model.joblib")
MODEL_STATE: dict = {}

EDUCATION_LEVELS = ["高中以下", "大學", "碩士以上"]
CITIES = ["城市A", "城市B", "城市C"]
MODEL_TYPES = ["LinearRegression", "Lasso", "Ridge"]

CSS = """
:root {
    --brand: #4f46e5;
    --brand-light: #6366f1;
    --accent: #06b6d4;
    --text-main: #0f172a;
    --muted: #64748b;
}

body {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

.hero-card {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 45%, #06b6d4 100%);
    border-radius: 20px;
    padding: 28px 26px;
    color: #fff;
    margin-bottom: 18px;
    box-shadow: 0 12px 30px -8px rgba(79, 70, 229, 0.45);
    position: relative;
    overflow: hidden;
}
.hero-card::after {
    content: "";
    position: absolute;
    right: -60px;
    top: -60px;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.12);
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 12px;
    letter-spacing: 1px;
    margin-bottom: 10px;
}
.hero-card h1 {
    font-size: 26px;
    margin: 0 0 6px 0;
    font-weight: 800;
}
.hero-card p {
    margin: 0;
    font-size: 14px;
    opacity: 0.92;
}

.model-chip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.model-chip span {
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 12px;
}

.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    width: 100%;
}
@media (max-width: 640px) {
    .result-grid { grid-template-columns: 1fr 1fr; }
}
.result-box {
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    color: #fff;
    box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.18);
}
.result-box .r-label {
    font-size: 12px;
    opacity: 0.9;
    margin-bottom: 6px;
}
.result-box .r-value {
    font-size: 24px;
    font-weight: 800;
}
.result-box .r-sub {
    font-size: 12px;
    opacity: 0.85;
    margin-top: 4px;
}
.box-moon { background: linear-gradient(135deg, #6366f1, #4f46e5); }
.box-sun  { background: linear-gradient(135deg, #06b6d4, #0ea5e9); }

.result-note {
    margin-top: 10px;
    font-size: 13px;
    color: var(--muted);
}

.train-card {
    border-left: 4px solid var(--brand);
    background: #eef2ff;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    color: var(--text-main);
}

.footer-note {
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    margin-top: 20px;
}
"""


def load_model_state() -> None:
    """載入最新模型狀態至全域變數；若模型檔不存在則自動訓練。"""
    global MODEL_STATE
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    MODEL_STATE = joblib.load(MODEL_PATH)


def predict_salary(years_experience: float, education_level: str, city: str) -> tuple:
    """依輸入預測月薪與年薪 (年薪 = 月薪 x 14)。"""
    try:
        state = MODEL_STATE
        oe = state["oe"]
        ohe = state["ohe"]
        scaler = state["scaler"]
        model = state["model"]

        edu_encoded = int(oe.transform(
            pd.DataFrame([[education_level]], columns=["EducationLevel"]))[0][0])
        city_vector = ohe.transform(pd.DataFrame([[city]], columns=["City"]))
        city_cols = ohe.get_feature_names_out(["City"])
        feature_row = [years_experience, edu_encoded] + list(city_vector[0])
        features = pd.DataFrame(
            [feature_row],
            columns=["YearsExperience", "EducationLevel"] + list(city_cols),
        )
        X_scaled = scaler.transform(features)
        monthly = float(model.predict(X_scaled)[0])
        annual = monthly * 14

        html = f"""
        <div class="result-grid">
          <div class="result-box box-moon">
            <div class="r-label">預測月薪</div>
            <div class="r-value">{monthly:,.1f} <span style="font-size:13px">萬元</span></div>
            <div class="r-sub">NT$ {monthly * 10000:,.0f}</div>
          </div>
          <div class="result-box box-sun">
            <div class="r-label">預估年薪 (14 個月)</div>
            <div class="r-value">{annual:,.1f} <span style="font-size:13px">萬元</span></div>
            <div class="r-sub">NT$ {annual * 10000:,.0f}</div>
          </div>
        </div>
        <div class="result-note">
          工作經驗 {years_experience} 年・{education_level}・{city}｜目前模型: {state.get('model_type', '-')}・R² = {state.get('r2', 0):.4f}
        </div>
        """
        return html
    except Exception as e:
        raise gr.Error(f"預測失敗: {e}")


def train_model(
    test_size: float,
    random_state: int,
    model_type: str,
    alpha: float,
    use_seed: bool,
) -> tuple:
    """線上重新訓練模型並更新模型狀態。"""
    try:
        res = train_and_save_model(
            test_size=test_size,
            random_state=random_state if use_seed else None,
            model_type=model_type,
            alpha=alpha,
        )
        load_model_state()
        coef_df = pd.DataFrame(
            [{"特徵": k, "權重": round(v, 4)} for k, v in res["feature_coefs"].items()]
        )
        note = (
            f"✅ <b>{res['message']}</b><br>"
            f"📊 決定係數 R² = <b>{res['r2']:.4f}</b>　|　⏱️ 耗時 {res['train_time']:.2f} 秒"
        )
        return res["r2"], res["train_time"], note, coef_df
    except Exception as e:
        raise gr.Error(f"訓練失敗: {e}")


def model_info_html() -> str:
    """產生目前模型狀態卡片。"""
    state = MODEL_STATE
    coefs = state.get("feature_coefs", {})
    chips = "".join(
        f"<span>{k}: {v:+.2f}</span>" for k, v in coefs.items()
    )
    return f"""
    <div class="train-card">
      <b>目前模型狀態</b><br>
      演算法: {state.get('model_type', '-')}・
      決定係數 R² = {state.get('r2', 0):.4f}・
      特徵數: {len(coefs)}
      <div class="model-chip">{chips}</div>
    </div>
    """


load_model_state()

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
)

hero = f"""
<div class="hero-card">
  <span class="hero-badge">AI・MACHINE LEARNING</span>
  <h1>💼 薪資預測系統</h1>
  <p>輸入工作經驗、學歷與所在城市，運用線性迴歸模型即時估算您的薪資水準</p>
</div>
"""

with gr.Blocks(title="薪資預測系統") as demo:
    gr.HTML(hero)

    with gr.Tabs():
        with gr.Tab("🔮 薪資預測"):
            gr.HTML(model_info_html())
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    years_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=3.0,
                        step=0.5,
                        label="工作經驗 (年)",
                        info="0 ~ 50 年，以 0.5 年為單位",
                    )
                    edu_dropdown = gr.Dropdown(
                        choices=EDUCATION_LEVELS,
                        value="大學",
                        label="學歷",
                    )
                    city_dropdown = gr.Dropdown(
                        choices=CITIES,
                        value="城市A",
                        label="所在城市",
                    )
                    with gr.Row():
                        predict_btn = gr.Button(
                            "🚀 立即預測", variant="primary", scale=2
                        )
                        clear_btn = gr.Button("↺ 重置", variant="secondary", scale=1)
                with gr.Column(scale=1):
                    result_html = gr.HTML(
                        """<div class="result-note">👈 請先設定條件，再點擊「立即預測」</div>"""
                    )

            predict_btn.click(
                predict_salary,
                inputs=[years_slider, edu_dropdown, city_dropdown],
                outputs=[result_html],
            )
            clear_btn.click(
                lambda: (
                    3.0,
                    "大學",
                    "城市A",
                    '<div class="result-note">👈 請先設定條件，再點擊「立即預測」</div>',
                ),
                outputs=[years_slider, edu_dropdown, city_dropdown, result_html],
            )

        with gr.Tab("🧠 模型訓練"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    test_size_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="測試集分割比例",
                    )
                    model_type_dropdown = gr.Dropdown(
                        choices=MODEL_TYPES,
                        value="LinearRegression",
                        label="模型演算法",
                        info="LinearRegression / Lasso / Ridge",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.001,
                        maximum=10.0,
                        value=1.0,
                        step=0.001,
                        label="正則化強度 α (Lasso / Ridge)",
                    )
                    with gr.Row():
                        seed_checkbox = gr.Checkbox(
                            value=True, label="固定隨機種子", scale=1
                        )
                        seed_number = gr.Number(
                            value=76,
                            precision=0,
                            minimum=0,
                            label="隨機種子",
                            scale=1,
                        )
                    train_btn = gr.Button(
                        "⚡ 開始訓練", variant="primary"
                    )
                with gr.Column(scale=1):
                    r2_label = gr.Label(label="決定係數 R²")
                    time_number = gr.Number(
                        label="訓練耗時 (秒)", precision=2
                    )
                    train_note = gr.HTML()
                    coef_table = gr.Dataframe(
                        headers=["特徵", "權重"],
                        label="特徵權重",
                        interactive=False,
                        wrap=True,
                    )

            train_btn.click(
                train_model,
                inputs=[
                    test_size_slider,
                    seed_number,
                    model_type_dropdown,
                    alpha_slider,
                    seed_checkbox,
                ],
                outputs=[r2_label, time_number, train_note, coef_table],
            )

    gr.HTML('<div class="footer-note">Powered by scikit-learn × Gradio ｜ 手機友善響應式介面</div>')

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
