import torch
import torch.nn as nn
import numpy as np
import pickle
from stable_baselines3 import PPO

# 1. Envoltorio para aislar la red determinista del Actor
class OnnxablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # Extraer características
        features = self.policy.extract_features(observation)
        # Pasar por las capas ocultas del Actor (evadiendo al Crítico)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        # Devolver la acción determinista (media) sin crear la distribución Normal
        return self.policy.action_net(latent_pi)

# 2. Cargar modelo original
model = PPO.load("grid_rl_agent_best.zip", device="cpu")

# 3. Envolver y preparar para evaluación
onnxable_model = OnnxablePolicy(model.policy)
onnxable_model.eval()

# Tensor de prueba con tus 1170 features
dummy_input = torch.FloatTensor(np.random.randn(1, 1170))

# 4. Exportar el modelo envuelto
torch.onnx.export(
    onnxable_model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=18,
    input_names=["input"],
    output_names=["output"]
)
print("✅ Modelo determinista exportado exitosamente a model.onnx")

# 5. Extraer y guardar estadísticas de normalización
with open("vec_normalize_stats.pkl", "rb") as f:
    vec_normalize = pickle.load(f)

np.save("obs_mean.npy", vec_normalize.obs_rms.mean)
np.save("obs_var.npy", vec_normalize.obs_rms.var)
print("✅ Estadísticas de normalización guardadas")