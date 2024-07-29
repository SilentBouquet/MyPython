import numpy as np
import kaiwu as kw
import matplotlib.pyplot as plt

kw.license.init(user_id="51648452576182274", sdk_code="AvuHse0ji1LEsAMWjI1Y3u7igbvxxF")

matrix = -np.array([
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]])

matrix_n = kw.cim.normalizer(matrix, normalization=0.5)
output = kw.cim.simulator_core(
    matrix_n,
    c=0,
    pump=0.7,
    noise=0.01,
    laps=1000,
    dt=0.1)

h = kw.sampler.hamiltonian(matrix, output)

plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.plot(output, linewidth=1)
plt.title("Pulse Phase")
plt.ylabel("Phase")
plt.xlabel("T")
plt.subplot(212)
plt.plot(h, linewidth=1)
plt.title("Hamiltonian")
plt.ylabel("H")
plt.xlabel("T")
plt.show()

c_set = kw.sampler.binarizer(output)
opt = kw.sampler.optimal_sampler(matrix, c_set, 0)
best = opt[0][0]
print(best)