from transformers import BertConfig, BertModel
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
import random
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from transformers import AdamW
# import wandb
from tqdm.notebook import tqdm

from dataset.vocab import AttrDict

device = torch.device("cuda")
scaler = torch.cuda.amp.GradScaler()

SEED = 0
N_FOLDS = 5
MAX_LEN = 320

LR = 2e-5
WEIGHT_DECAY = 1e-6
N_EPOCHS = 8
WARM_UP_RATIO = 0.1

BS = 32
ACCUMULATE = 1
MIXED_PRECISION = False

EXP_NAME = 'baseline'

def create_folds(data):
    data["kfold"] = -1
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 不明な処理
    # num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # data.loc[:, "bins"] = pd.cut(
    #     data["Is Fraud?"], bins=num_bins, labels=False
    # )
    # kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
    #     data.loc[v_, 'kfold'] = f
    # data = data.drop("bins", axis=1)
    
    return data

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CommonDataset(Dataset):
    
    def __init__(self, df):
        self.texts = df["User"].tolist()
        self.labels = df["Is Fraud?"].tolist()

        for index, value in enumerate(self.labels):
            if value == "No":
                self.labels[index] = 0
            else:
                self.labels[index] = 1

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[BOS]", "[EOS]"]
        special_field_tag = "SPECIAL"
        special_tokens_map = {}

        for key, token in zip(keys, special_tokens):
            token = "%s_%s" % (special_field_tag, token)
            special_tokens_map[key] = token
        
        tok = BertTokenizer(
            vocab_file="./output_card/vocab.nb", 
            do_lower_case=False,
            **AttrDict(special_tokens_map))
        
        print(tok)
        
        d = {
            # "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
            # "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
            # "token_type_ids": torch.tensor(tok["token_type_ids"], dtype=torch.long),
            # "label": torch.tensor(label, dtype=torch.double),
            "label": torch.tensor(label, dtype=torch.int),
        }
        
        return d

class CommonModel(nn.Module):
    
    def __init__(self):
        super(CommonModel, self).__init__()
        self.config = BertConfig.from_pretrained("./output_card/checkpoint-35000/config.json")
        self.model = BertModel.from_pretrained("./output_card/checkpoint-35000/pytorch_model.bin", config=self.config)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, batch_first=True)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        out, _ = self.lstm(outputs['last_hidden_state'], None)
        sequence_output = out[:, -1, :]
        logits = self.regressor(sequence_output)

        return logits

    def loss_fn(self, logits, label):
        loss = torch.sqrt(nn.MSELoss(reduction='mean')(logits[:, 0], label))
        return loss

    def validation_loop(valid_loader, model):
        model.eval()
        preds = []
        for d in valid_loader:
            with torch.no_grad():
                logits = model(
                        # d["input_ids"].to(device),
                        # d["attention_mask"].to(device),
                        # d["token_type_ids"].to(device)
                        input_ids=None,
                        attention_mask=None,
                        token_type_ids=None,
                )
            preds.append(logits[:, 0])
        y_pred = torch.hstack(preds).cpu().numpy()
        y_true = valid_loader.dataset.labels
        mse_loss = mean_squared_error(y_true, y_pred, squared=False)
        return mse_loss

# def main(): 一旦コメントアウト
train_df = pd.read_csv("./data/credit_card/card_transaction.v2.csv")
train_df = create_folds(train_df)

train_index = train_df.query('kfold!=0').index.tolist()
valid_index = train_df.query('kfold==0').index.tolist()

# set dataset
train_dataset = CommonDataset(train_df.loc[train_index])
valid_dataset = CommonDataset(train_df.loc[valid_index])

train_loader = DataLoader(train_dataset, batch_size=BS,
                        pin_memory=True, shuffle=True, drop_last=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BS,
                        pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

# set models
model = CommonModel()
model.to(device)

# set optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
max_train_steps = N_EPOCHS * len(train_loader)
warmup_steps = int(max_train_steps * WARM_UP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_train_steps
)

# wandb.init(project='CommonLit', entity='trtd56', name=EXP_NAME)
# wandb.watch(model)

bar = tqdm(total=max_train_steps)

set_seed()
optimizer.zero_grad()
train_iter_loss, valid_best_loss, all_step = 0, 999, 0
for epoch in range(N_EPOCHS):
    for d in train_loader:
        all_step += 1
        model.train()
        
        if MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                logits = model(
                    # d["input_ids"].to(device),
                    # d["attention_mask"].to(device),
                    # d["token_type_ids"].to(device)
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                )
                loss = model.loss_fn(logits, d["label"].float().to(device))
                loss = loss / ACCUMULATE
        else:
            logits = model(
                # d["input_ids"].to(device),
                # d["attention_mask"].to(device),
                # d["token_type_ids"].to(device)
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
            )
            loss = model.loss_fn(logits, d["label"].float().to(device))
            loss = loss / ACCUMULATE

        train_iter_loss += loss.item()

        if MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if all_step % ACCUMULATE == 0:
            if MIXED_PRECISION:
                scaler.step(optimizer) 
                scaler.update() 
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            valid_loss = validation_loop(valid_loader, model)
            if valid_best_loss > valid_loss:  
                valid_best_loss = valid_loss

            wandb.log({
                "train_loss": train_iter_loss,
                "valid_loss": valid_loss,
                "valid_best_loss": valid_best_loss,
            })
            train_iter_loss = 0
        bar.update(1)
# wandb.finish()
   
# if __name__ == "main":
#     main()