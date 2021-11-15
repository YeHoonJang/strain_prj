import os
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.nn.functional as F
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

train_data = CustomDataset(df_train, 30, 1)     #argparse
valid_data = CustomDataset(df_valid, 30, 1)

batch_size = 100     #argparse
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=8)    # drop_last = drop the last incomplete batch
valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=8)


class VAELSTM(nn.Module):
    def __init__(self, n_features, n_past, n_future, n_layers, hidden_size, embed_size, dropout):
        super(VAELSTM, self).__init__()

        self.n_features = n_features
        self.n_past = n_past
        self.n_future = n_future
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout = dropout

        # encoder
        self.embedding_embed = nn.Linear(n_features, embed_size)
        self.embedding_hidden = nn.Linear(embed_size, hidden_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, dropout=dropout)

        #decoder
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=embed_size, dropout=dropout)
        self.fc = nn.Linear(embed_size, n_future)


        self.hidden2mean = nn.Linear(hidden_size, embed_size)   #embedsize == latent_size
        self.hidden2logv = nn.Linear(hidden_size, embed_size)   #embedsize == latent_size
        self.latent2hidden = nn.Linear(embed_size, embed_size)
        self.hidden2embed = nn.Linear(self.hidden_size, self.embed_size)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        # encoding
        _, (hidden, cell) = self.encoder(self.embedding_embed(x))

        # get mu, log_var
        mu = self.hidden2mean(hidden).to(device)
        logv = self.hidden2logv(hidden).to(device)
        z = self.reparameterize(mu, logv).to(device)

        # decoding
        hidden = self.latent2hidden(z).to(device)
        cell = self.hidden2embed(cell)
        outputs, _ = self.decoder(self.embedding_hidden(self.embedding_embed(x)), (hidden, cell)) # [batch_size, n_past, hidden]
        out = self.fc(outputs[:, -1, :]) # [batch_size, n_future]

        return out, mu, logv, z


def kl_anneal_function(epoch, k, x0):
    # logistic
    return float(1/(1+np.exp(-k*(epoch-x0))))



def loss_fn(out, target, mu, logv, epoch, k, xo):
    MSE = nn.MSELoss()
    MSE_loss = MSE(out, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1+logv-mu.pow(2)-logv.exp())
    KL_weight = kl_anneal_function(epoch, k, xo)

    return MSE_loss, KL_loss, KL_weight

model = VAELSTM(n_features=2, n_past=30, n_future=1, n_layers=8, hidden_size=512, embed_size=256, dropout=0.1)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            output, mu, logv, z = model(X)   # forward

            MSE_loss, KL_loss, KL_weight = loss_fn(output, y, mu, logv, epoch, 0.0025, 2500)   # k=0.0025, x0=2500
            loss = (MSE_loss + KL_loss*KL_weight)
            loss_value = loss.item()
            train_loss += loss_value

            optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total

@torch.no_grad()    #no autograd (backpropagation X)
def evaluate(model, data_loader, device):
    y_list = []
    output_list = []

    model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            output, mu, logv, z = model(X)
            MSE_loss, KL_loss, KL_weight = loss_fn(output, y, mu, logv, epoch, 0.0025, 2500)  # k=0.0025, x0=2500
            loss = (MSE_loss + KL_loss * KL_weight)
            loss_value = loss.item()
            valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()
            pbar.update(1)

    return valid_loss/total, y_list, output_list


# Train
start_epoch = 0
epochs = 100  #argparse
print("Start Training..")
for epoch in range(start_epoch, epochs+1):
    print(f"Epoch: {epoch}")
    epoch_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Training Loss: {epoch_loss:.5f}")

    valid_loss, y_list, output_list = evaluate(model, valid_loader, device)
    # rmse = np.sqrt(valid_loss)
    print(f"Validation Loss: {valid_loss:.5f}")
    # print(f'RMSE is {rmse:.5f}')

    y_list = minmax_scaler2.inverse_transform(np.array(y_list).reshape(-1, 1)).reshape(-1)
    output_list = minmax_scaler2.inverse_transform(np.array(output_list).reshape(-1, 1)).reshape(-1)

    if epoch%5 == 0:
        plt.clf()
        plt.figure(figsize=(10, 8))
        plt.plot(y_list)
        plt.plot(output_list)
        data_path = os.path.join(os.getcwd(), "data", "figure")
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        plt.savefig(f"{data_path}/figure_{int(epoch)+1}.png")
        plt.close()






