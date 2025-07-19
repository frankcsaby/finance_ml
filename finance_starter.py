import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
torch.manual_seed(42)
np.random.seed(42)

class DataLoader:
    def __init__(self, symbol='SPY', period='2y'):
        self.symbol = symbol
        self.period = period
        self.data = None
    
    def fetch_data(self):
        print(f"Data downloading: {self.symbol}")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        return self.data
    
    def calculate_returns(self):
        if self.data is None:
            raise ValueError("First please fill the datas!")
        
        #Daily returns, Volatility and Realized Volatility
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        self.data['Future_Vol'] = self.data['Volatility'].shift(-5)
        
        #Technical indicators
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        
        return self.data
    
    def calculate_rsi(self, prices, window=14):
        #RSI calc
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100/ (1 + rs))
        return rsi
    
    def create_features(self):
        #Feature engineering
        
        #Lag features
        for i in range(1, 6):
            self.data[f'Vol_lag{i}'] = self.data['Volatility'].shift(i)
            self.data[f'Return_lag_{i}'] = self.data['Returns'].shift(i)
        
        #Moving averages
        self.data['Vol_MA_5'] = self.data['Volatility'].rolling(window=5).mean()
        self.data['Vol_MA_10'] = self.data['Volatility'].rolling(window=10).mean()
        
        #High-Low spread
        self.data['HL_Spread'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        
        #Volume-based features
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        #Drop rows with NaN values
        self.data = self.data.dropna()
        
        return self.data

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(VolatilityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #LSTM layers
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        #Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        #Exit layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        #LSTM forward pass
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]
        
        #Applying dropout
        output = self.dropout(output)
        
        #Final prediction
        prediction = self.fc(output)
        return prediction

class ModelTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, data, features, target, sequence_length=20):
        #Feature selection
        X = data[features].values
        y = data[target].values
        
        #Normalizing
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        #Creating sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i+sequence_length])
            y_seq.append(y_scaled[i+sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001):
    #Data transform to tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    #Creating DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #Optimizer and loss function
    optimizer = optim.Adam(self.model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    #Training loop
    for epoch in range(epochs):
        #Training
        self.model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        #Validation
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
                
        #Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        self.train_losses.append(avg_train_loss)
        self.val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

def evaulate(self, X_test, y_test):
    self.model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        predictions = self.model(X_test_tensor).cpu().numpy()
        
        #Transforming back
        predictions_orig = self.scaler_y.inverse_transform(predictions)
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        #Metrics
        mse = mean_squared_error(y_test_orig, predictions_orig)
        mae = mean_absolute_error(y_test_orig, predictions_orig)
        rmse = np.sqrt(mse)
        
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        
        return predictions_orig, y_test_orig, {'mse' : mse, 'mae' : mae, 'rmse' : rmse}

def plot_training_history(self):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(self.train_losses, label='Training Loss')
    plt.plot(self.val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(self.train_losses[-50:], label='Training Loss (last 50)')
    plt.plot(self.val_losses[-50:], label='Validation Loss (last 50)')
    plt.title('Training History (Last 50 epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def main():
    print("=== Financial Neural Network Project ===")
    print("S&P 500 Volatility Prediction\n")
    
    #Download data
    print("Downloading data...")
    data_loader = DataLoader(symbol='SPY', period='2y')
    data = data_loader.fetch_data()
    print(f"Downloaded datas: {len(data)} day")
    
    #Feature engineering
    print("\nFeature engineering...")
    data = data_loader.calculate_returns()
    data = data_loader.create_features()
    print(f"Processed data: {len(data)} day")
    print(f"Features number: {len(data.columns)}")
    
    #Visualizing datas
    print("\nVisualizing data...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    #Price and volatility
    axes[0, 0].plot(data.index, data['Close'])
    axes[0, 0].set_title('S&P 500 Price')
    axes[0, 0].set_ylabel('Price ($)')
    
    axes[0, 1].plot(data.index, data['Volatility'])
    axes[0, 1].set_title('Volatility')
    axes[0, 1].set_ylabel('Volatility')
    
    #Returns and RSI
    axes[1, 0].plot(data.index, data['Returns'])
    axes[1, 0].set_title('Daily Returns')
    axes[1, 0].set_ylabel('Returns')
    
    axes[1, 1].plot(data.index, data['RSI'])
    axes[1, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('RSI')
    axes[1, 1].set_ylabel('RSI')
    
    plt.tight_layout()
    plt.show()
    
    #Making the model
    print("\nCreating model...")
    
    #Feature selections
    features = [
        'Vol_lag1_1, Vol_lag_2, Vol_lag_3, Vol_lag_4, Vol_lag_5, Return_lag_1, Return_lag_2, Return_lag_3, Return_lag_4, Return_lag_5, Vol_MA_5, Vol_MA_10, HL_Spread, Volume_Ratio, RSI'
    ]
    
    target = 'Future_Vol'
    
    #Model and train initialization
    model = VolatilityLSTM(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.3)
    trainer = ModelTrainer(model)
    
    #Data preparing
    X, y = trainer.prepare_data(data, features, target, sequence_length=20)
    
    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    #Training model
    print("\nTraining model...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001)
    
    #Results
    print("\nResults...")
    trainer.plot_training_history()
    
    #Review test set
    predictions, actual, metrics = trainer.evaulate(X_test, y_test)
    
    #Predictions plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Volatility', alpha=0.7)
    plt.plot(predictions, label='Predicted Volatility', alpha=0.7)
    plt.title('Volatility Prediction vs Reality')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()
    
    #Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions, alpha=0.6)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.xlabel('Actual Volatility')
    plt.ylabel('Predicted Volatility')
    plt.title('Predicted vs Actual Volatility')
    plt.show
    
    if  __name__ == "__main__":
        main()