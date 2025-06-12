# Trabalho-AI-Bruno---2025-1
Disponilização recursos experimentais do Trabalho da disciplina Inteligencia Artificial (FACOM/UFMS) - 1 semestre 2025

Olá! Sou o Rodrigo Kbessa no meu primeiro projeto aqui! Espero que se divirtam!

Tutorial: CVNN vs MLP no Plano Complexo

Vou tentar mostrar como instalar o Ubuntu, configurar o ambiente Python/Jupyter, e executar redes neurais reais (MLP) e complexas (CVNN) para classificar pontos no plano complexo.

Sumário

1. Instalação do Ubuntu
2. Instalação do Jupyter Notebook e dependências
3. Execução dos códigos
   - Dataset sintético e visualização
   - MLP real rasa
   - CVNN rasa
   - Fronteira de decisão CVNN rasa
   - MLP real rasa (benchmark)
   - Fronteira de decisão MLP real rasa
   - CVNN profunda
   - MLP profunda
   - Comparação final em tabela (opcional)

1. Instalação do Ubuntu

- Baixe o Ubuntu em ubuntu.com/download
- Crie um pendrive bootável (Rufus no Windows, Startup Disk Creator no Linux)
- Inicie o computador pelo pendrive e siga as instruções

2. Instalação do Jupyter Notebook e dependências

Abra o terminal e digite

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 -m venv venv
source venv/bin/activate
pip install notebook numpy matplotlib torch
pip install pandas    # opcional, para visualizar tabelas
```

Para abrir o Jupyter Notebook

```bash
jupyter notebook
```

Se não abrir o navegador automaticamente, copie o link do terminal e cole no navegador.

3. Execução dos códigos

Copie cada sessão abaixo para uma célula no Jupyter Notebook, na ordem.

Geração e visualização do dataset sintético

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 1000
x = np.random.uniform(-1.5, 1.5, N)
y = np.random.uniform(-1.5, 1.5, N)
z = x + 1j * y
labels = (np.abs(z) >= 1).astype(int)

idx = np.random.permutation(N)
train_idx, test_idx = idx[:800], idx[800:]
z_train, z_test = z[train_idx], z[test_idx]
labels_train, labels_test = labels[train_idx], labels[test_idx]

plt.figure(figsize=(6,6))
plt.scatter(z[labels==0].real, z[labels==0].imag, c='blue', label='Classe 0 (|z|<1)', alpha=0.6)
plt.scatter(z[labels==1].real, z[labels==1].imag, c='red', label='Classe 1 (|z|≥1)', alpha=0.6)
circle = plt.Circle((0,0), 1, color='k', linestyle='--', fill=False, label='Fronteira ideal |z|=1')
plt.gca().add_artist(circle)
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.title('Distribuição dos pontos no plano complexo')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
```

MLP real rasa

```python
import torch
import torch.nn as nn
import torch.optim as optim

def to_tensor(z):
    return torch.tensor(np.stack([z.real, z.imag], axis=1), dtype=torch.float32)

X_train = to_tensor(z_train)
y_train = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)
X_test = to_tensor(z_test)
y_test = torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1)

class MLPReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

mlp = MLPReal()
optimizer = optim.Adam(mlp.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

n_epochs = 50
losses = []
accs = []

for epoch in range(n_epochs):
    mlp.train()
    optimizer.zero_grad()
    out = mlp(X_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    mlp.eval()
    with torch.no_grad():
        pred = torch.sigmoid(mlp(X_test)) > 0.5
        acc = (pred.float() == y_test).float().mean().item()
        accs.append(acc)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de perda (MLP real rasa)")
plt.subplot(1,2,2)
plt.plot(accs)
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Curva de acurácia (MLP real rasa)")
plt.show()
```

CVNN rasa

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def to_complex_tensor(z):
    return torch.tensor(np.stack([z.real, z.imag], axis=1), dtype=torch.float32)

X_train_c = to_complex_tensor(z_train)
X_test_c = to_complex_tensor(z_test)
y_train_c = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)
y_test_c = torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_real = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.W_imag = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.b_real = nn.Parameter(torch.zeros(out_features))
        self.b_imag = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)
        Xr = input[..., 0]
        Xi = input[..., 1]
        real = Xr @ self.W_real - Xi @ self.W_imag
        imag = Xr @ self.W_imag + Xi @ self.W_real
        real = real + self.b_real
        imag = imag + self.b_imag
        return torch.stack([real, imag], dim=2)

class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features))
    def forward(self, input):
        real = input[..., 0]
        imag = input[..., 1]
        z = torch.complex(real, imag)
        mag = torch.abs(z)
        phase = torch.angle(z)
        activated = torch.relu(mag + self.bias)
        out_real = activated * torch.cos(phase)
        out_imag = activated * torch.sin(phase)
        return torch.stack([out_real, out_imag], dim=2)

class CVNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = ComplexLinear(1, 16)
        self.act1 = ModReLU(16)
        self.fc2 = ComplexLinear(16, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        out = x[:, 0, 0]
        return out.unsqueeze(1)

cvnn = CVNN()
optimizer = optim.Adam(cvnn.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

n_epochs = 50
losses_c = []
accs_c = []

for epoch in range(n_epochs):
    cvnn.train()
    optimizer.zero_grad()
    out = cvnn(X_train_c)
    loss = criterion(out, y_train_c)
    loss.backward()
    optimizer.step()
    losses_c.append(loss.item())
    cvnn.eval()
    with torch.no_grad():
        logits = cvnn(X_test_c)
        pred = torch.sigmoid(logits) > 0.5
        acc = (pred.float() == y_test_c).float().mean().item()
        accs_c.append(acc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses_c)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de perda (CVNN rasa)")
plt.subplot(1,2,2)
plt.plot(accs_c)
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Curva de acurácia (CVNN rasa)")
plt.show()
```

Fronteira de decisão CVNN rasa

```python
xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-1.6, 1.6, 300))
zz = xx + 1j * yy
mesh_points = np.stack([zz.real.ravel(), zz.imag.ravel()], axis=1)
mesh_tensor = torch.tensor(mesh_points, dtype=torch.float32)

cvnn.eval()
with torch.no_grad():
    out_mesh = cvnn(mesh_tensor)
    preds_mesh = (torch.sigmoid(out_mesh) > 0.5).numpy().reshape(xx.shape)

plt.figure(figsize=(8,8))
plt.contourf(xx, yy, preds_mesh, alpha=0.4, cmap="coolwarm")
plt.scatter(X_test_c[:,0], X_test_c[:,1], c=labels_test, cmap="coolwarm", edgecolor="k")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.title("Fronteira de decisão da CVNN rasa (dados de teste sobrepostos)")
plt.show()
```

MLP real rasa (benchmark para CVNN)

```python
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mlp = MLP()
optimizer_mlp = optim.Adam(mlp.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

losses_mlp = []
accs_mlp = []

for epoch in range(50):
    mlp.train()
    optimizer_mlp.zero_grad()
    out = mlp(X_train_c)
    loss = criterion(out, y_train_c)
    loss.backward()
    optimizer_mlp.step()
    losses_mlp.append(loss.item())
    mlp.eval()
    with torch.no_grad():
        logits = mlp(X_test_c)
        pred = torch.sigmoid(logits) > 0.5
        acc = (pred.float() == y_test_c).float().mean().item()
        accs_mlp.append(acc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses_mlp)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de perda (MLP real rasa)")
plt.subplot(1,2,2)
plt.plot(accs_mlp)
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Curva de acurácia (MLP real rasa)")
plt.show()
```

Fronteira de decisão da MLP real rasa

```python
mlp.eval()
with torch.no_grad():
    out_mesh_mlp = mlp(mesh_tensor)
    preds_mesh_mlp = (torch.sigmoid(out_mesh_mlp) > 0.5).numpy().reshape(xx.shape)

plt.figure(figsize=(8,8))
plt.contourf(xx, yy, preds_mesh_mlp, alpha=0.4, cmap="coolwarm")
plt.scatter(X_test_c[:,0], X_test_c[:,1], c=labels_test, cmap="coolwarm", edgecolor="k")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.title("Fronteira de decisão da MLP real rasa (dados de teste sobrepostos)")
plt.show()
```

CVNN profunda

```python
class DeepCVNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = ComplexLinear(1, 64)
        self.act1 = ModReLU(64)
        self.fc2 = ComplexLinear(64, 64)
        self.act2 = ModReLU(64)
        self.fc3 = ComplexLinear(64, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        out = x[:, 0, 0]
        return out.unsqueeze(1)

deep_cvnn = DeepCVNN()
optimizer_deep_cvnn = optim.Adam(deep_cvnn.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

losses_deep_cvnn = []
accs_deep_cvnn = []

for epoch in range(50):
    deep_cvnn.train()
    optimizer_deep_cvnn.zero_grad()
    out = deep_cvnn(X_train_c)
    loss = criterion(out, y_train_c)
    loss.backward()
    optimizer_deep_cvnn.step()
    losses_deep_cvnn.append(loss.item())
    deep_cvnn.eval()
    with torch.no_grad():
        logits = deep_cvnn(X_test_c)
        pred = torch.sigmoid(logits) > 0.5
        acc = (pred.float() == y_test_c).float().mean().item()
        accs_deep_cvnn.append(acc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses_deep_cvnn)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de perda (Deep CVNN)")
plt.subplot(1,2,2)
plt.plot(accs_deep_cvnn)
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Curva de acurácia (Deep CVNN)")
plt.show()

deep_cvnn.eval()
with torch.no_grad():
    out_mesh = deep_cvnn(mesh_tensor)
    preds_mesh = (torch.sigmoid(out_mesh) > 0.5).numpy().reshape(xx.shape)

plt.figure(figsize=(8,8))
plt.contourf(xx, yy, preds_mesh, alpha=0.4, cmap="coolwarm")
plt.scatter(X_test_c[:,0], X_test_c[:,1], c=labels_test, cmap="coolwarm", edgecolor="k")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.title("Fronteira de decisão (Deep CVNN)")
plt.show()
```

MLP profunda

```python
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

deep_mlp = DeepMLP()
optimizer_deep_mlp = optim.Adam(deep_mlp.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

losses_deep_mlp = []
accs_deep_mlp = []

for epoch in range(50):
    deep_mlp.train()
    optimizer_deep_mlp.zero_grad()
    out = deep_mlp(X_train_c)
    loss = criterion(out, y_train_c)
    loss.backward()
    optimizer_deep_mlp.step()
    losses_deep_mlp.append(loss.item())
    deep_mlp.eval()
    with torch.no_grad():
        logits = deep_mlp(X_test_c)
        pred = torch.sigmoid(logits) > 0.5
        acc = (pred.float() == y_test_c).float().mean().item()
        accs_deep_mlp.append(acc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses_deep_mlp)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de perda (Deep MLP)")
plt.subplot(1,2,2)
plt.plot(accs_deep_mlp)
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Curva de acurácia (Deep MLP)")
plt.show()

deep_mlp.eval()
with torch.no_grad():
    out_mesh = deep_mlp(mesh_tensor)
    preds_mesh = (torch.sigmoid(out_mesh) > 0.5).numpy().reshape(xx.shape)

plt.figure(figsize=(8,8))
plt.contourf(xx, yy, preds_mesh, alpha=0.4, cmap="coolwarm")
plt.scatter(X_test_c[:,0], X_test_c[:,1], c=labels_test, cmap="coolwarm", edgecolor="k")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.title("Fronteira de decisão (Deep MLP)")
plt.show()
```

Comparação final em tabela (opcional, requer pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

resultados = {
    "Modelo": ["CVNN rasa", "MLP rasa", "CVNN profunda", "MLP profunda"],
    "Acurácia final (%)": [
        round(100 * accs_c[-1], 1),
        round(100 * accs_mlp[-1], 1),
        round(100 * accs_deep_cvnn[-1], 1),
        round(100 * accs_deep_mlp[-1], 1)
    ]
}

df = pd.DataFrame(resultados)
fig, ax = plt.subplots(figsize=(6,2))
ax.axis('off')
tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
plt.title("Acurácias finais dos modelos", pad=20)
plt.savefig("tabela_acuracias_finais.png", dpi=300, bbox_inches='tight')
plt.show()
```

Pronto para usar! Modifique, adapte e use como quiser. Duvidas? Call me (por aqui mesmo) ou rodrigo.campos@embrapa.br
