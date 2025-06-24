# app.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import plotly.graph_objects as go

# ============================
# 1. Load dan Preprocessing Data
# ============================

# Pastikan file ada di folder `data/`
df = pd.read_csv('D:\\dashboard_beban_listrik\\beban_listrik_harian.csv', parse_dates=['Tanggal'], index_col='Tanggal')
df = df.asfreq('D')
df = df[-60:]  # Ambil 60 hari terakhir

y = df['Beban_MW']

def create_lag_features(series, lags=7):
    df_lag = pd.DataFrame()
    for i in range(1, lags+1):
        df_lag[f'lag_{i}'] = series.shift(i)
    df_lag['target'] = series
    return df_lag.dropna()

initial_data = y[:-30]
forecast_horizon = 30

# ============================
# 2. Rolling Forecast - XGBoost
# ============================
rolling_forecast = []
current_series = initial_data.copy()

for _ in range(forecast_horizon):
    lagged = create_lag_features(current_series)
    X_train = lagged.drop(columns='target')
    y_train = lagged['target']

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    last_values = current_series[-7:]
    input_features = np.array(last_values).reshape(1, -1)
    pred = model.predict(input_features)[0]
    rolling_forecast.append(pred)

    next_day = current_series.index[-1] + pd.Timedelta(days=1)
    current_series.loc[next_day] = pred

forecast_index = current_series.index[-forecast_horizon:]
forecast_series = pd.Series(rolling_forecast, index=forecast_index)

# ============================
# 3. Forecast SARIMA
# ============================
sarima_model = SARIMAX(initial_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.get_forecast(steps=forecast_horizon)
sarima_mean = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

# ============================
# 4. Streamlit Dashboard
# ============================
st.set_page_config(page_title="Dashboard Beban Listrik", layout="wide")
st.title("⚡ Forecast Beban Listrik Harian - XGBoost vs SARIMA")

st.markdown("## Data Aktual")
st.line_chart(y.rename("Beban Aktual"))

st.markdown("## Perbandingan Forecast 30 Hari")

fig = go.Figure()
fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Aktual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines+markers', name='XGBoost', line=dict(color='green', dash='dash')))

fig.add_trace(go.Scatter(x=sarima_mean.index, y=sarima_mean, mode='lines+markers', name='SARIMA', line=dict(color='red', dash='dot')))

fig.add_trace(go.Scatter(x=sarima_ci.index, y=sarima_ci.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=sarima_ci.index, y=sarima_ci.iloc[:, 1], mode='lines', fill='tonexty', name='SARIMA CI', line=dict(width=0), fillcolor='rgba(255,0,0,0.2)'))

fig.update_layout(title="Forecast 30 Hari ke Depan", xaxis_title="Tanggal", yaxis_title="Beban (MW)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)


# CHUNK 5 - Evaluasi (Jika Ada Data Aktual)
# ============================
if y.index[-1] >= forecast_index[-1]:
    y_true = y[forecast_index[0]:forecast_index[-1]]
    y_pred_xgb = forecast_series
    y_pred_sarima = sarima_mean

    print("\nEvaluasi Prediksi XGBoost:")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred_xgb)))
    print("MAE:", mean_absolute_error(y_true, y_pred_xgb))
    print("R^2:", r2_score(y_true, y_pred_xgb))

    print("\nEvaluasi Prediksi SARIMA:")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred_sarima)))
    print("MAE:", mean_absolute_error(y_true, y_pred_sarima))
    print("R^2:", r2_score(y_true, y_pred_sarima))