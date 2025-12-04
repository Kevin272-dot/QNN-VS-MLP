# Requirements: see requirements.txt (pennylane, torch, scikit-learn, matplotlib, numpy)
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os

# Enable interactive plotting
plt.ion()

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------
# Config / hyperparams
# -----------------------
N_SAMPLES = 300
TEST_SIZE = 0.25
NOISE = 0.2
n_qubits = 2           # for make_moons (2 features)
n_layers = 2           # variational layers
batch_size = None      # None -> full-batch (small dataset)
lr = 0.01
n_epochs = 40
use_vectorized_qnode = False  # newer PennyLane versions handle batching differently
shots = None  # analytic gradients if None; set to int for shot simulation

device_name = "default.qubit"
dev = qml.device(device_name, wires=n_qubits, shots=shots)

# -----------------------
# Dataset
# -----------------------
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

# Torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -----------------------
# QNodes and circuit
# -----------------------
wires = list(range(n_qubits))

# We'll use templates: AngleEmbedding for encoding and StronglyEntanglingLayers for ansatz
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def qnode(batch_x, params):
    # batch_x: single sample shape (n_qubits,)
    # params: shape (n_layers, n_qubits, 3)
    qml.templates.AngleEmbedding(batch_x, wires=wires, rotation="Y")
    qml.templates.StronglyEntanglingLayers(params, wires=wires)
    # return expectation on first wire
    return qml.expval(qml.PauliZ(wires[0]))


# -----------------------
# Hybrid PyTorch Module
# -----------------------
class QNNClassifier(nn.Module):
    def __init__(self, n_layers, n_qubits):
        super().__init__()
        # params shaped for StronglyEntanglingLayers: (n_layers, n_qubits, 3)
        init_params = 0.01 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        self.q_params = nn.Parameter(init_params)
        # small classical layer to map expectation [-1,1] -> logit
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        # x shape: (batch, features)
        # Process each sample individually since vectorized=False
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            ev = qnode(x[i], self.q_params)  # returns scalar
            results.append(ev)
        ev_tensor = torch.stack(results).view(-1, 1).to(torch.get_default_dtype())
        logits = self.fc(ev_tensor)  # raw logits
        return logits  # return logits (BCEWithLogitsLoss expects logits)

# -----------------------
# Classical baseline
# -----------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)  # logits

# -----------------------
# Training utilities with dynamic plotting
# -----------------------
def train_model_with_plots(model, X_tr, y_tr, X_val, y_val, n_epochs=100, lr=0.01, model_name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    val_acc = []
    val_auc = []
    
    # Setup dynamic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Training Progress (Live Update)', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).detach().numpy().flatten()
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            preds = (val_probs > 0.5).astype(int)
            acc = accuracy_score(y_val.numpy().flatten(), preds)
            try:
                auc = roc_auc_score(y_val.numpy().flatten(), val_probs)
            except Exception:
                auc = float("nan")
            val_acc.append(acc)
            val_auc.append(auc)

        # Update plots every epoch
        if epoch % 2 == 0 or epoch == n_epochs - 1:
            for ax in axes.flat:
                ax.clear()
            
            # Plot 1: Training Loss
            axes[0, 0].plot(train_losses, color='#e74c3c', linewidth=2, label='Training Loss')
            axes[0, 0].set_xlabel('Epoch', fontweight='bold')
            axes[0, 0].set_ylabel('Loss', fontweight='bold')
            axes[0, 0].set_title('Training Loss Over Time', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot 2: Validation Accuracy
            axes[0, 1].plot(val_acc, color='#2ecc71', linewidth=2, label='Val Accuracy')
            axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
            axes[0, 1].set_xlabel('Epoch', fontweight='bold')
            axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
            axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
            axes[0, 1].set_ylim([0, 1.05])
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Plot 3: ROC AUC
            axes[1, 0].plot(val_auc, color='#3498db', linewidth=2, label='ROC AUC')
            axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
            axes[1, 0].set_xlabel('Epoch', fontweight='bold')
            axes[1, 0].set_ylabel('AUC Score', fontweight='bold')
            axes[1, 0].set_title('ROC AUC Score', fontweight='bold')
            axes[1, 0].set_ylim([0, 1.05])
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Plot 4: Current Metrics Summary
            axes[1, 1].axis('off')
            current_metrics = f"""
            CURRENT METRICS (Epoch {epoch+1}/{n_epochs})
            {'='*40}
            
            Training Loss:     {loss.item():.4f}
            Validation Acc:    {acc:.4f} ({acc*100:.1f}%)
            ROC AUC:          {auc:.4f}
            
            Best Val Acc:     {max(val_acc):.4f}
            Best AUC:         {max([x for x in val_auc if not np.isnan(x)]):.4f}
            
            Progress:         {(epoch+1)/n_epochs*100:.1f}%
            """
            axes[1, 1].text(0.1, 0.5, current_metrics, fontsize=12, family='monospace',
                           verticalalignment='center', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.3))
            
            plt.draw()
            plt.pause(0.01)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f} | AUC: {auc:.4f}")
    
    plt.ioff()
    return {"loss": train_losses, "val_acc": val_acc, "val_auc": val_auc, "figure": fig}

# -----------------------
# Run QNN experiment
# -----------------------
def run_qnn_experiment():
    print("QNN experiment config:", {"n_qubits": n_qubits, "n_layers": n_layers, "shots": shots})
    qnn = QNNClassifier(n_layers=n_layers, n_qubits=n_qubits)
    start = time.time()
    history = train_model_with_plots(qnn, X_train_t, y_train_t, X_test_t, y_test_t, 
                                     n_epochs=n_epochs, lr=lr, model_name="Quantum Neural Network")
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    # final evaluation
    qnn.eval()
    with torch.no_grad():
        logits = qnn(X_test_t).numpy().flatten()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
    print("QNN Test acc:", acc, "ROC AUC:", auc)
    print("Confusion matrix:\n", cm)
    return qnn, history

# -----------------------
# Run classical baseline
# -----------------------
def run_mlp_baseline():
    print("\n" + "="*50)
    print("Training Classical MLP Baseline...")
    print("="*50)
    mlp = SimpleMLP(input_dim=X_train_t.shape[1], hidden=8)
    
    # Use the same dynamic plotting function
    start = time.time()
    history = train_model_with_plots(mlp, X_train_t, y_train_t, X_test_t, y_test_t,
                                     n_epochs=n_epochs, lr=lr, model_name="Classical MLP")
    elapsed = time.time() - start
    print(f"MLP training finished in {elapsed:.1f}s")
    
    mlp.eval()
    with torch.no_grad():
        logits = mlp(X_test_t).numpy().flatten()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
    print("MLP Test acc:", acc, "ROC AUC:", auc)
    print("Confusion matrix:\n", cm)
    return mlp, history

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("\n" + "🚀"*25)
    print(" QUANTUM vs CLASSICAL NEURAL NETWORK SHOWDOWN")
    print("🚀"*25 + "\n")
    
    # Run experiments
    qnn_model, qnn_history = run_qnn_experiment()
    mlp_model, mlp_history = run_mlp_baseline()

    # Final Comparison Plot
    print("\n" + "="*50)
    print("Generating final comparison plots...")
    print("="*50)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Quantum Neural Network vs Classical MLP - Complete Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Loss Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(qnn_history["loss"], color='#9b59b6', linewidth=2.5, label='QNN', alpha=0.8)
    ax1.plot(mlp_history["loss"], color='#3498db', linewidth=2.5, label='Classical MLP', alpha=0.8)
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax1.set_title('Training Loss Comparison', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Accuracy Comparison
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(qnn_history["val_acc"], color='#9b59b6', linewidth=2.5, label='QNN', alpha=0.8)
    ax2.plot(mlp_history["val_acc"], color='#3498db', linewidth=2.5, label='Classical MLP', alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Baseline')
    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax2.set_title('Validation Accuracy Comparison', fontweight='bold', fontsize=13)
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: AUC Comparison
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(qnn_history["val_auc"], color='#9b59b6', linewidth=2.5, label='QNN', alpha=0.8)
    ax3.plot(mlp_history["val_auc"], color='#3498db', linewidth=2.5, label='Classical MLP', alpha=0.8)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Baseline')
    ax3.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax3.set_ylabel('ROC AUC', fontweight='bold', fontsize=11)
    ax3.set_title('ROC AUC Score Comparison', fontweight='bold', fontsize=13)
    ax3.set_ylim([0, 1.05])
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Final Metrics Bar Chart
    ax4 = fig.add_subplot(gs[0, 2])
    final_qnn_acc = qnn_history["val_acc"][-1]
    final_mlp_acc = mlp_history["val_acc"][-1]
    models = ['QNN', 'MLP']
    accuracies = [final_qnn_acc, final_mlp_acc]
    colors_bar = ['#9b59b6', '#3498db']
    bars = ax4.bar(models, accuracies, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Final Accuracy', fontweight='bold', fontsize=10)
    ax4.set_title('Final Accuracy', fontweight='bold', fontsize=12)
    ax4.set_ylim([0, 1.0])
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 5: AUC Bar Chart
    ax5 = fig.add_subplot(gs[1, 2])
    final_qnn_auc = qnn_history["val_auc"][-1]
    final_mlp_auc = mlp_history["val_auc"][-1]
    aucs = [final_qnn_auc, final_mlp_auc]
    bars2 = ax5.bar(models, aucs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Final AUC', fontweight='bold', fontsize=10)
    ax5.set_title('Final ROC AUC', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1.0])
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, auc in zip(bars2, aucs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 6: Summary Stats
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    best_qnn_acc = max(qnn_history["val_acc"])
    best_mlp_acc = max(mlp_history["val_acc"])
    best_qnn_auc = max([x for x in qnn_history["val_auc"] if not np.isnan(x)])
    best_mlp_auc = max([x for x in mlp_history["val_auc"] if not np.isnan(x)])
    
    winner_acc = "QNN" if best_qnn_acc > best_mlp_acc else "MLP"
    winner_auc = "QNN" if best_qnn_auc > best_mlp_auc else "MLP"
    
    summary_text = f"""
    FINAL RESULTS SUMMARY
    {'='*30}
    
    Quantum Neural Network:
      Best Accuracy: {best_qnn_acc:.4f}
      Best AUC:      {best_qnn_auc:.4f}
      Final Loss:    {qnn_history['loss'][-1]:.4f}
    
    Classical MLP:
      Best Accuracy: {best_mlp_acc:.4f}
      Best AUC:      {best_mlp_auc:.4f}
      Final Loss:    {mlp_history['loss'][-1]:.4f}
    
    🏆 Winner (Accuracy): {winner_acc}
    🏆 Winner (AUC):      {winner_auc}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=1))
    
    plt.savefig('qnn_vs_mlp_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Comparison plot saved as 'qnn_vs_mlp_comparison.png'")
    plt.show()
    
    print("\n" + "✨"*25)
    print(" EXPERIMENT COMPLETE!")
    print("✨"*25 + "\n")