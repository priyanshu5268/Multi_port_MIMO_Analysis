import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fpdf import FPDF
import os

# === Load Excel File ===
file_path = "C:\\Users\\HP\\OneDrive\\Desktop\\multiport_mimo_data.xlsx"  # ✅ Update your path here
data = pd.read_excel(file_path)

# === Extract Frequency and Ports ===
frequency_col = data.columns[0]
port_columns = data.columns[1:]
data["Frequency_GHz"] = data[frequency_col] * 1000
freq_GHz = data["Frequency_GHz"].values.reshape(-1, 1)

# === PDF Setup ===
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()
pdf.set_font("Times", "B", 42)
pdf.cell(200, 20, "Multi-Port MIMO Antenna Analysis Report", ln=True, align='C')
pdf.ln(20)

# === Loop Over Each Port ===
for port in port_columns:
    S11_dB = data[port].values
    S11_linear = 10 ** (S11_dB / 20)
    VSWR = (1 + np.abs(S11_linear)) / (1 - np.abs(S11_linear))
    radiation_efficiency = (1 - np.abs(S11_linear) ** 2) * 100
    group_delay = -np.gradient(S11_dB, data["Frequency_GHz"])
    avg_group_delay = np.mean(group_delay) * 1e9
    Z0 = 50
    Zin = Z0 * (1 + S11_linear) / (1 - S11_linear)
    min_s11_threshold = -10
    valid_range = data[data[port] <= min_s11_threshold]
    bandwidth = valid_range["Frequency_GHz"].iloc[-1] - valid_range["Frequency_GHz"].iloc[0] if not valid_range.empty else 0
    polarization_angle = np.degrees(np.angle(S11_linear))

    # === ML Models ===
    classifiers = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVM Regressor": SVR(kernel="rbf"),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    }

    # === Parameters to Analyze ===
    parameters = {
        "Return Loss (S11)": S11_dB,
        "VSWR": VSWR,
        "Radiation Efficiency": radiation_efficiency,
        "Group Delay (ns)": group_delay * 1e9,
        "Impedance Matching (Real)": Zin.real,
        "Impedance Matching (Imag)": Zin.imag,
        "Bandwidth (GHz)": [bandwidth] * len(freq_GHz),
        "Group Delay Average (ns)": [avg_group_delay] * len(freq_GHz),
        "Polarization Angle (deg)": polarization_angle
    }

    results = {}
    best_classifiers = {}

    for param, values in parameters.items():
        y = np.array(values).reshape(-1, 1)
        results[param] = {}

        for name, model in classifiers.items():
            model.fit(freq_GHz, y.ravel())
            predicted = model.predict(freq_GHz)
            mse = mean_squared_error(y, predicted)
            r2 = r2_score(y, predicted)
            mae = mean_absolute_error(y, predicted)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y.ravel() - predicted) / np.maximum(np.abs(y.ravel()), 1e-8))) * 100
            var_score = model.score(freq_GHz, y)

            results[param][name] = {
                "MSE": mse, "R2 Score": r2, "MAE": mae, "RMSE": rmse,
                "MAPE": mape, "Variance Score": var_score, "Predicted": predicted
            }

        best_model = min(results[param], key=lambda x: results[param][x]["MSE"])
        best_classifiers[param] = best_model

        # === Plotting ===
        plt.figure(figsize=(14, 8))
        plt.plot(freq_GHz, values, label="Actual", linewidth=3)
        for name, res in results[param].items():
            plt.plot(freq_GHz, res["Predicted"], linestyle="--", label=f"{name} (MSE: {res['MSE']:.4f})")
        plt.xlabel("Frequency (GHz)", fontsize=20, fontweight='bold')
        plt.ylabel(param, fontsize=20, fontweight='bold')
        plt.title(f"{param} - {port}", fontsize=22, fontweight='bold')
        plt.grid(True)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        filename = f"{port}_{param.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        # === Add Plot to PDF ===
        pdf.add_page()
        pdf.set_font("Times", "B", 42)
        pdf.multi_cell(0, 20, f"{param} - {port}", align='L')
        pdf.image(filename, x=10, y=40, w=190)

    # === Summary Page ===
    pdf.add_page()
    pdf.set_font("Times", "B", 42)
    pdf.cell(0, 20, f"Summary for Port: {port}", ln=True)
    pdf.set_font("Times", size=18)

    for param in parameters:
        best = best_classifiers[param]
        metrics = results[param][best]
        summary = (f"{param} | Best: {best} | MSE: {metrics['MSE']:.4f} | R²: {metrics['R2 Score']:.4f} | "
                   f"Var: {metrics['Variance Score']:.4f} | MAE: {metrics['MAE']:.4f} | "
                   f"RMSE: {metrics['RMSE']:.4f} | MAPE: {metrics['MAPE']:.2f}%")
        pdf.multi_cell(0, 10, summary)

    min_idx = np.argmin(S11_dB)
    pdf.multi_cell(0, 10, f"Bandwidth: {bandwidth:.2f} GHz")
    pdf.multi_cell(0, 10, f"Avg. Group Delay: {avg_group_delay:.2f} ns")
    pdf.multi_cell(0, 10, f"Min Return Loss: {S11_dB[min_idx]:.2f} dB at {data['Frequency_GHz'].iloc[min_idx]:.2f} GHz")
    pdf.multi_cell(0, 10, f"Resonant Frequency: {data['Frequency_GHz'].iloc[min_idx]:.2f} GHz")

# === Save Final PDF Report ===
pdf_file = "Multiport_MIMO_Analysis_Report.pdf"
pdf.output(pdf_file, "F")
print(f"✅ High-resolution report generated: {pdf_file}")
