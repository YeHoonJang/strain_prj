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
        for i in range(n_past, len(df) - n_future + 1): # -1 말고 변수로
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
file_path = os.path.join(os.getcwd(), "data/01000002.txt")
total_data = pd.read_csv(file_path, sep="\t", index_col="Timestamp")
# total_data = total_data.loc[:, ["AccZ", "Str1", "Str2", "Str3"]]
total_data = total_data.loc[:, ["Str1", "Str2", "Str3"]]

minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
minmax_scaler = minmax_scaler.fit(total_data)
total_data_scaled = minmax_scaler.transform(total_data)

minmax_scaler2 = minmax_scaler.fit(np.array(total_data.Str3[:]).reshape(-1, 1))
total_data_scaled2 = minmax_scaler2.transform(np.array(total_data.Str3[:]).reshape(-1, 1))


train_valid_split = int(len(total_data_scaled) * 0.3)   #argparse
df_train = total_data_scaled[:-train_valid_split]
df_valid = total_data_scaled[-train_valid_split:]
# print(df_train.shape, df_valid.shape)

train_data = CustomDataset(df_train, 30, 1)     #argparse
valid_data = CustomDataset(df_valid, 30, 1)

batch_size = 100     #argparse
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=8)    # drop_last = drop the last incomplete batch
valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=8)


# Transformer
class Transformer(nn.Module):
    def __init__(self, n_feature, n_past, n_future, n_layers, n_head, dim_model, dim_embed, dropout):
        super(Transformer, self).__init__()
        self.n_feature = n_feature
        self.n_past = n_past
        self.n_future = n_future
        self.n_layers = n_layers
        self.n_head = n_head
        self.dim_model = dim_model
        self.dim_embed = dim_embed
        self.dropout = dropout

        self.embedding_1 = nn.Linear(n_feature, dim_model)    # n_feature -> dim_model 사이즈로 embedding
        self.embedding_2 = nn.Linear(n_future, dim_model)
        # self.embedding_2 = nn.Embedding(batch_size, n_future, dim_embed)
        self.fc = nn.Linear(dim_model, n_future)    # dim_model -> n_future 사이즈
        self.transformer = nn.Transformer(batch_first=True)
        # self.relu = nn.ReLU()

    def forward(self, x, y):
        # print(y)
        # print(x.size())
        # y = torch.zeros(batch_size, self.n_future, self.dim_model).to(device)
        # print("self.embedding(x), y:", self.embedding_1(x).size(), self.embedding_2(y).size())
        # y = y.view(batch_size, self.n_future, 1)
        out = self.transformer(self.embedding_1(x), self.embedding_2(y))
        # print("output:", out.size(), "out[:, -1, :]", out[:, :, -1].size())
        # print(out)
        # print("self.fc(out):", self.fc(out).size(), "self.fc(out[:, -1, :])", self.fc(out[:, -1, :]).size())
        out = self.fc(out)
        return out[:, -1, :]


model = Transformer(n_feature=2, n_past=30, n_future=1, n_layers=8, n_head=8, dim_model=512, dim_embed=256, dropout=0.1)
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
            # y = y.unsqueeze(-1)

            output = model(X, y.unsqueeze(-1))   # forward
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
            # y = y.unsqueeze(-1)

            output = model(X, y.unsqueeze(-1))
            loss = criterion(output, y)
            loss_value = loss.item()
            valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()
            # print("y:", y_list[:100], "\nout:", output_list[:100])
            pbar.update(1)

    return valid_loss/total, y_list, output_list


# Train
start_epoch = 0
epochs = 100  #argparse
print("Start Training..")
for epoch in range(start_epoch, epochs):
    print(f"Epoch: {epoch}")
    epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Training Loss: {epoch_loss:.5f}")

    valid_loss, y_list, output_list = evaluate(model, valid_loader, criterion, device)
    # print("y:", y_list[:100], "\nout:", output_list[:100])
    rmse = np.sqrt(valid_loss)
    print(f"Validation Loss: {valid_loss:.5f}")
    print(f'RMSE is {rmse:.5f}')

    y_list = minmax_scaler2.inverse_transform(np.array(y_list).reshape(-1, 1)).reshape(-1)
    output_list = minmax_scaler2.inverse_transform(np.array(output_list).reshape(-1, 1)).reshape(-1)

    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.plot(y_list)
    plt.plot(output_list)
    data_path = os.path.join(os.getcwd(), "data", "figure")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    plt.savefig(f"{data_path}/figure_{int(epoch)+1}.png")
    plt.close()






