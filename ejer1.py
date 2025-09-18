# ejercicio_arviz_bayes.py
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm     

# ---------------------------------------------------
# 1) Simulamos datos
# ---------------------------------------------------
np.random.seed(123)

n_a = 60
n_b = 55

# Grupo A: media 50, sigma 8
# Grupo B: media 53 (efecto real pequeño-moderado), sigma 8
mu_a = 50
mu_b = 53
sigma_true = 8

grupo_a = np.random.normal(loc=mu_a, scale=sigma_true, size=n_a)
grupo_b = np.random.normal(loc=mu_b, scale=sigma_true, size=n_b)

df = pd.DataFrame({
    "valor": np.concatenate([grupo_a, grupo_b]),
    "grupo": ["A"]*n_a + ["B"]*n_b
})

print("Primeras filas del dataset:\n", df.head())
print("\nResumen por grupo:\n", df.groupby("grupo").describe().T)

# Visualización rápida 
sns.boxplot(x="grupo", y="valor", data=df)
plt.title("Boxplot: Grupo A vs Grupo B")
plt.show()


# ---------------------------------------------------
# 2) Modelo bayesiano: Modelo 1 (efecto de grupo)
#    Normal con media por grupo y sigma compartida
# ---------------------------------------------------
with pm.Model() as model_group:
    # Priors: no informativos razonables
    mu_A = pm.Normal("mu_A", mu=0, sigma=100)
    mu_B = pm.Normal("mu_B", mu=0, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=50)

    # Likelihood
    obs_A = pm.Normal("obs_A", mu=mu_A, sigma=sigma, observed=grupo_a)
    obs_B = pm.Normal("obs_B", mu=mu_B, sigma=sigma, observed=grupo_b)

    # Sample
    idata_group = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.9)

print("\nMuestreo terminado para model_group.")
print(az.summary(idata_group, var_names=["mu_A", "mu_B", "sigma"], round_to=2))


# ---------------------------------------------------
# 3) Modelo nulo: misma media para ambos grupos
# ---------------------------------------------------
with pm.Model() as model_null:
    mu = pm.Normal("mu", mu=0, sigma=100)
    sigma0 = pm.HalfNormal("sigma0", sigma=50)

    obs = pm.Normal("obs", mu=mu, sigma=sigma0, observed=df["valor"].values)

    idata_null = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.9)

print("\nMuestreo terminado para model_null.")
print(az.summary(idata_null, var_names=["mu", "sigma0"], round_to=2))


# ---------------------------------------------------
# 4) Análisis con ArviZ - diagnóstico y visualización
# ---------------------------------------------------

# 4.1 Trace plots (convergencia)
az.plot_trace(idata_group, var_names=["mu_A", "mu_B", "sigma"])
plt.suptitle("Traceplot - Modelo con efecto de grupo", y=1.02)
plt.show()

# 4.2 Posterior de la diferencia (mu_B - mu_A)
# Convertimos a array y calculamos la diferencia
mu_A_samples = idata_group.posterior["mu_A"].values.flatten()
mu_B_samples = idata_group.posterior["mu_B"].values.flatten()
diff = mu_B_samples - mu_A_samples

# Hagamos un InferenceData ad-hoc para la diferencia
idata_diff = az.from_dict(posterior={"diff_mu": diff})

az.plot_posterior(idata_diff, var_names=["diff_mu"], credible_interval=0.95, point_estimate="mean")
plt.title("Posterior: mu_B - mu_A")
plt.show()

# Probabilidad posterior de que la diferencia sea > 0
prob_pos = (diff > 0).mean()
print(f"Probabilidad posterior P(mu_B - mu_A > 0) = {prob_pos:.3f}")

# 4.3 Intervalos, resumen
print("\nResumen de la diferencia (mu_B - mu_A):")
print(az.summary(idata_diff, var_names=["diff_mu"], hdi_prob=0.95))

# 4.4 Posterior predictive check (usamos model_group)
with model_group:
    ppc = pm.sample_posterior_predictive(idata_group, var_names=["obs_A", "obs_B"], random_seed=123)
# Convert ppc to InferenceData and plot
idata_ppc = az.from_pymc(posterior_predictive=ppc, prior=None)
az.plot_ppc(idata_ppc)
plt.show()


# 4.5 Comparación de modelos con LOO
comp = az.compare({"model_group": idata_group, "model_null": idata_null}, ic="loo")
print("\nComparación de modelos (LOO):\n", comp)

# 4.6 Forest plot (intervalos) para mu_A y mu_B
az.plot_forest(idata_group, var_names=["mu_A", "mu_B"], combined=True, hdi_prob=0.95)
plt.title("Intervalos de credibilidad para mu_A y mu_B")
plt.show()
