import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import os  
import xgboost as xgb

# Load the model
model = joblib.load('xgb_model.pkl')

# Define feature names used for the model
feature_names = [
    "Hb", "PLT", "ALT", "BUN", "UA", "HDL"
]

# Streamlit user interface
st.title("社区老年二型糖尿病患者糖尿病肾病风险预测")

# Hb: numerical input
Hb = st.number_input("血红蛋白 <g/L>:", min_value=50, max_value=200, value=120)

# PLT: numerical input
PLT = st.number_input("血小板 <10^9/L>:", min_value=10, max_value=500, value=280)

# ALT: numerical input
ALT = st.number_input("血清谷丙转氨酶 <U/L>:",  value=25)

# BUN: numerical input
BUN = st.number_input("血尿素氮 <mmol/L>:", min_value=0, max_value=50, value=5)

# UA: numerical input
UA = st.number_input("尿酸)<μmol/L>:", min_value=100, max_value=800, value=350)

# HDL: numerical input
HDL = st.number_input("高密度脂蛋白胆固醇 <mmol/L>:", value=2)


# Process inputs and make predictions
feature_values = [Hb, PLT , ALT, BUN, UA, HDL]
features = np.array([feature_values], dtype=float)


if st.button("预测"):
    try:
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**预测类别:** {'高风险' if predicted_class == 1 else '低风险'} (DKD=1表示高风险)")
        st.write(f"**预测概率:** 无事件={predicted_proba[0]:.2%}, 事件={predicted_proba[1]:.2%}")

        # Generate advice with your requested wording
        if predicted_class == 1:
            probability_DKD = predicted_proba[1] * 100  # 正确定义变量
            advice = (
                "根据模型预测，您可能存在较高的糖尿病肾病发病风险。\n"
                f"模型预测的发病概率为 {probability_DKD:.1f}%。\n"
                "强烈建议您与主治医生详细讨论此结果，并采取积极的干预措施，如调整药物治疗、改变生活方式等。"
            )
        else:
            probability_no_DKD = predicted_proba[0] * 100  # 正确定义变量
            advice = (
                "根据模型预测，您的糖尿病肾病风险较低。\n"
                f"模型预测的无事件概率为 {probability_no_DKD:.1f}%。\n"
                "建议您继续保持健康的生活方式，并遵医嘱定期复查。"
            )
        
        st.info(advice)  # 使用info框使建议更醒目

        # Calculate SHAP values and display force plot
        explainer = shap.TreeExplainer(model)
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # For binary classification, shap_values is a list with two arrays
        # We want the SHAP values for the positive class (class 1)
        shap_values_for_positive_class = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # Create force plot in memory
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            shap_values_for_positive_class[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False,
            text_rotation=45
        )
        
        # Save to buffer instead of file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        
        # Display the plot
        st.image(buf, caption='特征贡献度解释 (红色增加风险，蓝色降低风险)', use_column_width=True)
        
        # Add clinical interpretation note
        st.caption("""
        **结果解读**：上图展示了各因素对预测结果的贡献。箭头向右（红色）表示增加糖尿病肾病风险，箭头向左（蓝色）表示降低风险。
        基线值（Base value）是所有患者的平均预测概率，最终预测值（f(x)）是考虑所有特征后的结果。
        """)

    except Exception as e:
        st.error(f"预测过程中出错: {str(e)}")

        st.exception(e)  # 显示完整错误堆栈




