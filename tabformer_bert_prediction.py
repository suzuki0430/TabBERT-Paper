import pandas as pd
import torch
from models.common import CommonModel
from dataset.vocab import Vocabulary
import pickle

device = torch.device("cuda")

adap_thres=10 ** 8

column_names = ['User',
                'Card',
                'Timestamp',
                'Amount',
                'Use Chip',
                'Merchant Name',
                'Merchant City',
                'Merchant State',
                'Zip',
                'MCC',
                'Errors?',
                'Is Fraud?']

# prepare data
data = pd.read_csv('./data/credit_card/preprocessed/card_transaction.v3.encoded.csv', dtype='Int64')
# data = torch.tensor(data.iloc[1, :].drop("Is Fraud?").tolist())
data = data.iloc[1, :].drop("Is Fraud?").tolist()

# load token2id
with open('vocab_token2id.bin', 'rb') as p:
    vocab_dic = pickle.load(p)

# transfer data to input_ids
vocab = Vocabulary(adap_thres)
sep_id = vocab.get_id(vocab.sep_token, special_token=True)

vocab_ids = []
for jdx, field in enumerate(data):
    vocab_id, _ = vocab_dic[column_names[jdx]][field]
    vocab_ids.append(vocab_id)

vocab_ids.append(sep_id)
input_ids = torch.tensor([vocab_ids], dtype=torch.long)

# load model
model = CommonModel()
model.to(device)
model.load_state_dict(torch.load("./output_fine_tuning/fine_tuning_model.pth"))
model.eval()
with torch.no_grad():
  output = model(input_ids.to(device))
  print("output", output)
  ans = torch.argmax(output, 1)
  print("ans", ans)

