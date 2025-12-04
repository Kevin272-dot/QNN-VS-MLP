# Quantum Neural Network (QNN) vs Classical MLP 🚀

A hybrid quantum-classical machine learning project comparing Quantum Neural Networks with classical Multi-Layer Perceptrons on binary classification tasks.

## 🎯 What This Project Does

- **Quantum Neural Network (QNN)**: Uses PennyLane to create a variational quantum circuit with:
  - Angle encoding for classical data
  - StronglyEntanglingLayers as trainable ansatz
  - Parameter-shift gradient computation
  - PyTorch integration for hybrid training

- **Classical Baseline**: Simple 2-layer MLP with similar parameter count

- **Dynamic Visualizations**: Real-time plots showing:
  - Training loss progression
  - Validation accuracy curves
  - ROC AUC scores
  - Live metric updates during training
  - Final head-to-head comparison

## 📋 Requirements

Install dependencies using:

```powershell
pip install -r requirements.txt
```

Or manually:

```powershell
pip install pennylane torch scikit-learn matplotlib numpy
```

## 🚀 Quick Start

### Run the complete experiment:

```powershell
python qnn_pennylane_pytorch.py
```

This will:
1. Train the Quantum Neural Network (with live plots)
2. Train the Classical MLP (with live plots)
3. Generate a final comparison visualization
4. Save results as `qnn_vs_mlp_comparison.png`

## ⚙️ Configuration

Edit these parameters in `qnn_pennylane_pytorch.py`:

```python
n_qubits = 2           # Number of qubits (2 for 2D features)
n_layers = 2           # Variational circuit depth
n_epochs = 40          # Training epochs
lr = 0.01              # Learning rate
shots = None           # None for analytic, or int (e.g., 1024) for shot simulation
```

## 📊 Expected Runtime

- **QNN Training**: ~1.5-3 minutes (40 epochs)
- **MLP Training**: ~5-15 seconds
- **Total**: ~2-5 minutes on modern CPU

### To speed up:
- Reduce `n_epochs` (e.g., from 40 to 20)
- Reduce `n_layers` (e.g., from 2 to 1)

## 🎨 Dynamic Features

### Real-Time Plots
- Updates every 2 epochs during training
- Shows live metrics in a dedicated panel
- Color-coded for easy comparison

### Final Comparison
- Side-by-side loss/accuracy/AUC curves
- Bar charts comparing final metrics
- Winner declaration based on best performance
- Automatically saved as PNG

## 📈 Output Files

- `qnn_vs_mlp_comparison.png` - Final comparison visualization

## 🔬 Dataset

Uses `make_moons` from scikit-learn:
- 300 samples total
- 75/25 train/test split
- 0.2 noise level
- StandardScaler preprocessing

## 🧪 Experiment with Different Settings

### Use finite measurement shots (realistic quantum hardware):
```python
shots = 1024  # Adds measurement noise
```

### Try different encodings:
Replace `AngleEmbedding` with `AmplitudeEmbedding` or custom encoding.

### Increase circuit complexity:
```python
n_layers = 4  # Deeper circuit (slower but more expressive)
```

### Use different ansatz:
Replace `StronglyEntanglingLayers` with `BasicEntanglerLayers` or custom gates.

## 📊 Typical Results

**Quantum Neural Network:**
- Accuracy: ~85-92%
- ROC AUC: ~0.90-0.95

**Classical MLP:**
- Accuracy: ~88-95%
- ROC AUC: ~0.92-0.98

*Note: QNNs may match or slightly underperform classical models on this toy task. The goal is to understand quantum ML mechanics, not achieve quantum advantage on small problems.*

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pennylane'`
**Solution:** Install PennyLane: `pip install pennylane`

### Issue: Plots don't show
**Solution:** 
- Check if matplotlib backend supports interactive display
- Plots are automatically saved as PNG anyway

### Issue: Training is too slow
**Solution:**
- Reduce `n_epochs` or `n_layers`
- Use analytic mode (`shots=None`)
- The loop-based qnode calls are intentionally simple for clarity

### Issue: Different results each run
**Solution:** Seeds are set for reproducibility. If using `shots` (finite measurements), results will vary slightly due to quantum measurement noise.

## 🔮 Next Steps

1. **Try different datasets**: Iris (binary subset), circles, or custom data
2. **Experiment with noise models**: Use PennyLane's noise channels
3. **Quantum kernels**: Try quantum kernel SVM instead of variational circuits
4. **Hardware execution**: Connect to IBM Quantum or other providers
5. **Hyperparameter tuning**: Grid search over learning rates, layers, etc.

## 📚 Key Concepts

- **Angle Encoding**: Maps classical features to qubit rotation angles
- **Variational Circuit**: Parameterized quantum gates trained like neural network weights
- **Parameter-Shift Rule**: Method for computing gradients on quantum hardware
- **Hybrid Training**: Classical optimizer updates quantum circuit parameters

## 🏆 Success Criteria

✅ Script runs without errors
✅ Dynamic plots display during training
✅ Both models train to >80% accuracy
✅ Final comparison plot is generated
✅ Results are reproducible

## 📖 References

- [PennyLane Documentation](https://pennylane.ai/)
- [Quantum Machine Learning Course](https://pennylane.ai/qml/)
- [Variational Quantum Circuits](https://arxiv.org/abs/1802.06002)

---

**Author**: Quantum Neural Network Explorer  
**Framework**: PennyLane + PyTorch  
**License**: MIT (or your choice)
