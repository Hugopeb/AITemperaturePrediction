import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import os
from datetime import datetime

print('Librerías importadas')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

hidden1 = 128
hidden2 = 64
hidden3 = 32
layers = [hidden1, hidden2, hidden3]

learning_rate = 0.001

dropout = 0.2
batch_size = 256


run_suffix = f"{hidden1}_{hidden2}_{hidden3}_{dropout}_{batch_size}_{learning_rate}"

ds = pd.read_csv('WRFTA_v1B.csv')
ds['fecha'] = pd.to_datetime(ds['fecha'])

f = ds.drop(columns = ['fecha', 'estacion', 'QSNOW_0','QSNOW_7','QSNOW_12','hora','dia'])
df = ds.dropna()
print(df.columns)

X = df.drop(columns =['TA','peso'])
Y = df['TA'] 
weights = df['peso']

# We create a directory to save the model and artifacts
run_name = f"WRFTA_{run_suffix}"
run_dir = f"./Modelos/{run_name}/{timestamp}"
os.makedirs(run_dir, exist_ok=True) 

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
# Escalado de salida (Y)S
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten()
# Es necesario guardar los reescalados para después cargarlos en el código modelo_entrenado.py y poder reescalar así los valores finales.
joblib.dump(scaler_X, os.path.join(run_dir, 'scaler_XWRF.save'))
joblib.dump(scaler_Y, os.path.join(run_dir, 'scaler_YWRF.save'))

# Separamos los datos en conjuntos de entrenamiento y de testeo diferentes para ver si realmente está aprendiendo la red.
X_train, X_val, Y_train, Y_val = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

np.save(os.path.join(run_dir,'featuresWRF.npy'), X_scaled)
np.save(os.path.join(run_dir,'targetsWRF.npy'), Y_scaled)

np.save(os.path.join(run_dir,'features_trainWRF.npy'), X_train)
np.save(os.path.join(run_dir,'targets_trainWRF.npy'), Y_train)

np.save(os.path.join(run_dir,'features_valWRF.npy'), X_val)
np.save(os.path.join(run_dir,'targets_valWRF.npy'), Y_val)

import torch

class DATA(Dataset):
    def __init__(self, path_x, path_y):
        self.features = np.load(path_x) 
        self.targets = np.load(path_y) 

    def __len__(self): 
        return len(self.features) 

    def __getitem__(self,idx):
        x = torch.from_numpy(self.features[idx].copy()).float()
        y = torch.tensor(self.targets[idx], dtype=torch.float32) 
        return x, y
    
train_dataset = DATA(
    os.path.join(run_dir, 'features_trainWRF.npy'), 
    os.path.join(run_dir, 'targets_trainWRF.npy'),
    )
    
val_dataset = DATA(
    os.path.join(run_dir, 'features_valWRF.npy'), 
    os.path.join(run_dir, 'targets_valWRF.npy'),
    )

dataset = DATA(
    os.path.join(run_dir, 'featuresWRF.npy'), 
    os.path.join(run_dir, 'targetsWRF.npy'),
    )

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = False) 

total_batches = len(train_loader) 
print(f'Número de batches: {total_batches}')

import torch.nn as nn

class RedNeuronal(nn.Module):
    def __init__(self, input_size):
        super(RedNeuronal, self).__init__()
        self.red = nn.Sequential( 
            nn.Linear(input_size,hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2,hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden3,1),  # hidden 3
        )

    def forward(self,x):
        return self.red(x)
    
input_size = X_scaled.shape[1] 
print(f'input_size: {input_size}')

model = RedNeuronal(input_size)
criterion = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience            
        self.min_delta = min_delta          
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=10, min_delta=0.001)

def train_one_epoch(model, train_loader, criterion, optimizer):
    epoch_loss = 0
    for batch_idx, (x_batch, y_batch, weights_batch) in enumerate(train_loader):
        pred = model(x_batch).squeeze()
        losses = criterion(pred, y_batch)
        loss = losses.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def validate(model, val_loader, criterion, scaler_Y):
    model.eval()
    val_loss = 0
    preds_celsius = []
    targets_celsius = []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            pred_val = model(x_val).squeeze()
            loss = criterion(pred_val, y_val).mean()
            val_loss += loss.item()

            pred_kelvin = scaler_Y.inverse_transform(pred_val.cpu().numpy().reshape(-1, 1))
            target_kelvin = scaler_Y.inverse_transform(y_val.cpu().numpy().reshape(-1, 1))
            preds_celsius.extend((pred_kelvin - 273.15).flatten())
            targets_celsius.extend((target_kelvin - 273.15).flatten())

    avg_val_loss = val_loss / len(val_loader)
    MAE_celsius = np.mean(np.abs(np.array(preds_celsius) - np.array(targets_celsius)))

    return avg_val_loss, MAE_celsius


num_epochs = 60
best_val_loss = float('inf')

for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    avg_val_loss, MAE_celsius = validate(model, val_loader, criterion, scaler_Y)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = f"./Modelos/{run_name}/{timestamp}/WRFTA_{run_suffix}.pt"
        torch.save(model.state_dict(), best_model_path)

    # Scheduler checks the average loss
    scheduler.step(avg_val_loss) 
    # Early stopper checks the average loss
    early_stopper(avg_val_loss) 

    model.train()
    # Printing the metrics for each epoch
    print(f"Epoch {epoch+1}, Loss entrenamiento: {avg_loss:.4f}, Loss validación (escalado): {avg_val_loss:.4f}, MAE validación (°C): {MAE_celsius:.4f}")
    
    # Early stopping check
    if early_stopper.early_stop:
        print(f"Parada temprana en la época {epoch+1}")
        break

# Loading the best model
print('Cargando el mejor modelo guardado...')
model.load_state_dict(torch.load(best_model_path))
print('Comenzando la predicción...')
model.eval()
all_preds_scaled = []
all_targets_scaled = []

# We use the best model to make predictions on the entire dataset
with torch.no_grad():
    for x_batch, y_batch in loader:
        y_pred = model(x_batch).squeeze()
        all_preds_scaled.append(y_pred.numpy())
        all_targets_scaled.append(y_batch.numpy())
        