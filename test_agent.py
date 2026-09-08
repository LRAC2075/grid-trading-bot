import torch
import numpy as np
import pickle
from stable_baselines3 import PPO

# 1. Exportar el modelo a ONNX
model = PPO.load("grid_rl_agent_best.zip", device="cpu")

# El tamaño de tu espacio de observación es 1170
dummy_input = torch.FloatTensor(np.random.randn(1, 1170))

torch.onnx.export(
    model.policy,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)
print("✅ Modelo exportado exitosamente a model.onnx")

# 2. Extraer parámetros de normalización (VecNormalize)
# Carga el archivo generado durante el entrenamiento
with open("vec_normalize_stats.pkl", "rb") as f:
    vec_normalize = pickle.load(f)

# Extraer media y varianza
obs_mean = vec_normalize.obs_rms.mean
obs_var = vec_normalize.obs_rms.var

# Guardarlos como arrays puros de NumPy (ligeros para el bot de 1GB RAM)
np.save("obs_mean.npy", obs_mean)
np.save("obs_var.npy", obs_var)
print("✅ Estadísticas de normalización guardadas (obs_mean.npy, obs_var.npy)")