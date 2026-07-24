# Heart Disease Prediction — VQC matching the conference paper

This folder is the corrected version of `quantum_model.py` — the part of the
original project that was supposed to implement the paper's Variational
Quantum Classifier (VQC), but instead only ran an untrained, single-qubit
toy circuit.

## What changed vs. the original `quantum_model.py`

| Paper (Section III-D / IV-B) | Original project code | This folder (`vqc_model.py`) |
|---|---|---|
| 4-qubit VQC | 1 qubit | 4 qubits ✅ |
| PCA reduces 13 → 4 features | No PCA | PCA(n_components=4) ✅ |
| RY feature map on each qubit | RY on 1 qubit using raw feature 0 only | RY(x_i) on all 4 qubits ✅ |
| Full CNOT entanglement chain | None | CNOT(0,1), CNOT(1,2), CNOT(2,3) ✅ |
| Trainable RX ansatz | None (no trainable parameters at all) | RX(θ_i) on each qubit, trained ✅ |
| COBYLA optimizer minimizing cross-entropy | No training loop / no optimizer | `scipy.optimize.minimize(method="COBYLA")` on binary cross-entropy ✅ |
| Classical baseline = Logistic Regression | Logistic Regression ✅ (this part already matched) | Logistic Regression ✅ |
| StandardScaler | StandardScaler ✅ | StandardScaler ✅ |

The original code's docstring even said outright: *"a simple demonstration
… not a production quantum ML model, so lower accuracy vs. the classical
model is expected."* That's an honest comment, but it means the code never
implemented QSVM, QANN, QBE, or VQC — none of the four quantum algorithms
the paper claims to benchmark.

## Honest note on results

Running this real VQC (`python vqc_model.py`) on `heart.csv` gives roughly:

- Classical Logistic Regression: ~81% accuracy
- 4-qubit VQC (RY feature map + RX ansatz only, no data re-uploading, COBYLA/200): ~55% accuracy

This is expected: a single layer of RX rotations after a fixed RY encoding is
a very low-capacity model (only 4 free parameters), and COBYLA on a
non-convex loss with a handful of parameters converges to a mediocre optimum quickly (it stopped after ~39 evaluations here). The paper's claim of quantum
models *beating* classical ones by +0.6% is not something this faithful
implementation reproduces — that's a common gap between what small NISQ-style
circuits can practically deliver and what the paper's tables state. If you
need the reported numbers to hold up, the ansatz would need to be deeper
(multiple RX/RY/CNOT layers), use more optimizer iterations/restarts, and
ideally use data re-uploading — happy to extend this if you want that next.

## Files

- `vqc_model.py` — the VQC implementation described above, plus the
  classical Logistic Regression baseline and full comparison pipeline.
- `model.py` — unchanged from your original project (Random Forest, used by
  `/predict` for the main disease-prediction form; this part was never part
  of the paper's comparison and didn't need fixing).
- `app.py` — same Flask endpoints as before; `/quantum-compare` now calls
  the real VQC instead of the toy circuit.
- `heart.csv` — your original dataset (13 attributes, matches the paper's
  Cleveland-based VQC dataset description).

## Run it

```bash
pip install -r requirements.txt
python vqc_model.py        # prints accuracy/timing comparison as JSON
# or
python app.py              # Flask app (needs your templates/index.html)
```
