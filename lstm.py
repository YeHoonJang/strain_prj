import os
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    # 반드시 init, len, getitem 구현
    def __init__(self, df, n_past, n_future):
        # 빈 리스트 생성 <- 데이터 저장
        self.X = []     # n_past 만큼의 feature 데이터
        self.y = []     # n_future 만큼의 label 데이터
        x_col = (df.shape[1]) - 1   # df 에서 -1번째 columns 까지 x
        for i in range(n_past, len(df) - n_future + 1):
            self.X.append(df[i - n_past:i, 0:x_col])
            self.y.append(df[i + n_future - 1: i + n_future, x_col])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# TODO: argparse 바꾸기

# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

# Data Setting
file_path = os.path.join(os.getcwd(), "data/total_data.csv")
total_data = pd.read_csv(file_path)

minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
minmax_scaler = minmax_scaler.fit(total_data)
total_data_scaled = minmax_scaler.transform(total_data)

minmax_scaler2 = minmax_scaler.fit(np.array(total_data.str3[:]).reshape(-1, 1))
total_data_scaled2 = minmax_scaler2.transform(np.array(total_data.str3[:]).reshape(-1, 1))


train_valid_split = int(len(total_data_scaled) * 0.3)   #argparse
df_train = total_data_scaled[:-train_valid_split]
df_valid = total_data_scaled[-train_valid_split:]
# print(df_train.shape, df_valid.shape)

train_data = CustomDataset(df_train, 30, 1)     #argparse
valid_data = CustomDataset(df_valid, 30, 1)

batch_size = 200     #argparse
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=8)    # drop_last = drop the last incomplete batch
valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=8)


# LSTM
class LSTM(nn.Module):
    def __init__(self, n_feature, n_past, n_future, n_layers, dim_model, dim_embed, dropout):
        super(LSTM, self).__init__()
        self.n_feature = n_feature
        self.n_past = n_past
        self.n_future = n_future
        self.n_layers = n_layers
        self.dim_model = dim_model
        self.dim_embed = dim_embed
        self.dropout = dropout

        self.embedding = nn.Linear(n_feature, dim_embed)    # n_feature -> dim_model 사이즈로 embedding
        self.fc = nn.Linear(dim_model, n_future)    # dim_model -> n_future 사이즈
        self.lstm = nn.LSTM(input_size=dim_embed, hidden_size=dim_model, dropout=dropout, batch_first=False)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state/cell state 초기화
        h = torch.zeros(1, self.n_past, self.dim_model).to(device)
        c = torch.zeros(1, self.n_past, self.dim_model).to(device)

        # out, (h, c) = self.lstm(self.embedding(x),)
        out, (h, c) = self.lstm(self.embedding(x), (h, c))
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(n_feature=3, n_past=30, n_future=1, n_layers=8,dim_model=512, dim_embed=256, dropout=0.1)
model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    criterion.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            output = model(X)   # forward
            loss = criterion(output, y)
            loss_value = loss.item()
            train_loss += loss_value

            optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total

@torch.no_grad()    #no autograd (backpropagation X)
def evaluate(model, data_loader, criterion, device):
    y_list = []
    output_list = []

    model.eval()
    criterion.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            output = model(X)
            loss = criterion(output, y)
            loss_value = loss.item()
            valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()
            pbar.update(1)

    return valid_loss/total, y_list, output_list


# Train
start_epoch = 0
epochs = 30  #argparse
print("Start Training..")
for epoch in range(start_epoch, epochs):
    print(f"Epoch: {epoch}")
    epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Training Loss: {epoch_loss:.3f}")

    valid_loss, y_list, output_list = evaluate(model, valid_loader, criterion, device)
    rmse = np.sqrt(valid_loss)
    print(f"Validation Loss: {valid_loss:.3f}")
    print(f'RMSE is {rmse:.3f}')

    y_list = minmax_scaler2.inverse_transform(np.array(y_list).reshape(-1, 1)).reshape(-1)
    output_list = minmax_scaler2.inverse_transform(np.array(output_list).reshape(-1, 1)).reshape(-1)

    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.plot(y_list)
    plt.plot(output_list)
    data_path = os.path.join(os.getcwd(), "data", "figure")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    plt.savefig(f"{data_path}/figure_{epoch}.png")







