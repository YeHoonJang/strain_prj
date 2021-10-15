import tqdm
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast


class CustomDataset(Dataset):
    def __init__(self, df, n_past, n_future):
        df = np.nan_to_num(df)
        x_col_num = (df.shape[1]) - 1
        self.X_list, self.y_list = list(), list()
        for i in range(n_past, len(df) - n_future + 1):
            self.X_list.append(df[i - n_past:i, :x_col_num])
            self.y_list.append(df[i + n_future - 1: i + n_future, x_col_num])

    def __getitem__(self, index):
        return self.X_list[index], self.y_list[index]

    def __len__(self):
        return len(self.X_list)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')
# TODO: argparse로 변경

# 1) Data load
# total_dat = pd.read_csv(os.path.join(args.data_path, args.data_name), sep=',')
file_path = os.path.join(os.getcwd(), "data/total_data.csv")
#file_path = './data/total_data.csv'
# total_data = pd.read_csv("./data/01000002.txt", sep='\t')
total_data = pd.read_csv(file_path)
# change columns order
# cols = total_data.columns.tolist()
# cols = cols[1:] + cols[:1]
# total_data = total_data[cols]

# 2) Data scaling
# minmax_scaler = MinMaxScaler(feature_range=(-args.minmax_scaler, args.minmax_scaler))
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
minmax_scaler = minmax_scaler.fit(total_data)
total_data_scaled = minmax_scaler.transform(total_data)

minmax_scaler2 = MinMaxScaler(feature_range=(-1, 1))
minmax_scaler2 = minmax_scaler2.fit(np.array(total_data.str3[:]).reshape(-1, 1))
total_data_scaled2 = minmax_scaler2.transform(np.array(total_data.str3[:]).reshape(-1, 1))

# 3) Data split
# train_valid_split = int(len(total_dat_scaled) * args.train_valid_ratio)
train_valid_split = int(len(total_data_scaled) * 0.3)
df_train = total_data_scaled[:-train_valid_split]
df_valid = total_data_scaled[-train_valid_split:]


dataset_dict = {
    'train': CustomDataset(df_train, n_past=30, n_future=1),
    'valid': CustomDataset(df_valid, n_past=30, n_future=1)
}

dataloader_dict = {
    'train': DataLoader(dataset_dict['train'], batch_size=16, drop_last=True,
                        num_workers=4),
    'valid': DataLoader(dataset_dict['valid'], batch_size=16, drop_last=True,
                        num_workers=4)
}

class LSTM(nn.Module):
    # TODO: argparse로 변경
    def __init__(self, n_feature: int = 3, n_past: int = 30, n_future: int = 1,
                 d_model: int = 512, d_embedding: int = 256,
                 dropout: float = 0.1, n_layers: int = 8):

        super(LSTM, self).__init__()

        # Hyper-parameter & Dropout setting
        self.n_past = n_past
        self.n_future = n_future
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.d_model = d_model

        self.src_embedding = nn.Linear(n_feature, d_model)

        # LSTM
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                            batch_first=False, bidirectional=False)

        # Target linear
        self.trg_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.trg_norm = nn.LayerNorm(d_embedding)
        self.trg_linear2 = nn.Linear(d_embedding, 1, bias=True)

        # TODO: autocast?
#     @autocast()
    def forward(self, src: Tensor, h0: Tensor = None, c0: Tensor = None) -> Tensor:

        # Source linear
        rnn_out = self.src_embedding(src).transpose(0, 1)
        h0 = torch.zeros(self.n_layers, 16, self.d_model, requires_grad=True).to(device, non_blocking=True) # 16 = batch_size -> argparse로
        c0 = torch.zeros(self.n_layers, 16, self.d_model, requires_grad=True).to(device, non_blocking=True) # 16 = batch_size -> argparse로
#         h0 = h0.transpose(0, 1).contiguous()
#         c0 = c0.transpose(0, 1).contiguous()

        # LSTM model
        rnn_out, _ = self.lstm(rnn_out, (h0, c0))

        # Target linear
        rnn_out = self.trg_norm(self.dropout(F.relu(self.trg_linear(rnn_out))))
        rnn_out = self.trg_linear2(rnn_out).transpose(0, 1).contiguous()
        rnn_out = rnn_out[:,-self.n_future:,:]
        return rnn_out



model = LSTM(n_feature = 3, n_past = 30, n_future = 1, d_model = 512,
             d_embedding = 256, dropout = 0.1, n_layers = 8)
model.to(device)
model.train()

# 확인용 - remove ok
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of params: {n_parameters}")

# params = [
#     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
#     {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
#      "lr": 1e-5} #args.lr_backbone
# ]

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)



def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:

        for i, (src, trg) in enumerate(data_loader):
            src = src.float().to(device, non_blocking=True)
            trg = trg.float().to(device, non_blocking=True)


            outputs = model(src)
            loss = criterion(outputs, trg)
            loss_value = loss.item()
            epoch_loss += loss_value

            optimizer.zero_grad()
            loss.backward()

            pbar.update(1)

    return epoch_loss / total


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    y_list = []
    output_list = []

    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for i, (src, trg) in enumerate(data_loader):
            src = src.float().to(device, non_blocking=True)
            trg = trg.float().to(device, non_blocking=True)

            outputs = model(src)
            loss = criterion(outputs, trg)

            validation_loss += loss.item()

            y_list += trg.detach().reshape(-1).tolist()
            output_list += outputs.detach().reshape(-1).tolist()

            pbar.update(1)

    return validation_loss / total, y_list, output_list


# Train
start_epoch = 0
print("Start Training..")
for epoch in range(start_epoch, 30):  # args.epochs
    print(f"Epoch: {epoch}")
    epoch_loss = train_one_epoch(
        model, criterion, dataloader_dict['train'], optimizer, device, epoch, 0.1)  # 0.1 = args.clip_max_norm
    lr_scheduler.step()
    # print(f"Training Loss: {epoch_loss**0.5}")

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
    }, "./checkpoint.pth")  # args.checkpoint

    validation_loss, y_list, output_list = evaluate(model, criterion, dataloader_dict['valid'], device)

    y_list = minmax_scaler2.inverse_transform(np.array(y_list).reshape(-1, 1)).reshape(-1)
    output_list = minmax_scaler2.inverse_transform(np.array(output_list).reshape(-1, 1)).reshape(-1)

    print(f"Training Loss: {epoch_loss}")
    print(f"Validation Loss: {validation_loss}")

    MSE = mean_squared_error(y_list, output_list)
    RMSE = np.sqrt(MSE)
    print(f'MSE is {MSE:.3f}')
    print(f'RMSE is {RMSE:.3f}')

    plt.clf()

    # y_list = minmax_scaler2.inverse_transform(np.array(y_list).reshape(-1, 1)).reshape(-1)
    # output_list = minmax_scaler2.inverse_transform(np.array(output_list).reshape(-1, 1)).reshape(-1)

    plt.figure(figsize=(20, 8))
    plt.plot(y_list)
    plt.plot(output_list)
    #plt.show()
    data_path = os.path.join(os.getcwd(), "data", "figure")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    plt.savefig(f"{data_path}/figure_{epoch}.png")
