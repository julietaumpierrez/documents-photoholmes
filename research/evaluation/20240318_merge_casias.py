# %%
import numpy as np

# %%
sp = [0.178, 0.073, 0.045, 0.104, 0.735, 0.735, 0.093, 0.185, 0.478, 0.734, 0.588]
cm = [0.129, 0.036, 0.001, 0.068, 0.577, 0.577, 0.052, 0.102, 0.428, 0.520, 0.347]

num_sp = 461
num_cm = 459
tot = num_cm + num_sp

# %%
sp = np.array(sp)
cm = np.array(cm)
total = sp * num_sp / tot + cm * num_cm / tot
print(total)
# %%
sp_osn = [0.171, 0.074, 0.000, 0.099, 0.589, 0.589, 0.087, 0.158, 0.359, 0.685, 0.447]
cm_osn = [0.129, 0.036, 0.000, 0.069, 0.563, 0.563, 0.053, 0.103, 0.422, 0.516, 0.338]

sp = np.array(sp_osn)
cm = np.array(cm_osn)
total = sp * num_sp / tot + cm * num_cm / tot
print(total)
# %%
sp_det = [0.376, 0.121, 0.539, 0.516, 0.761]
cm_det = [0.418, 0.013, 0.537, 0.551, 0.649]

sp = np.array(sp_det)
cm = np.array(cm_det)
total = sp * num_sp / tot + cm * num_cm / tot
print(total)

# %%
