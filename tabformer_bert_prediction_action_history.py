import pandas as pd
import torch
from models.common import CommonModel
from dataset.vocab import Vocabulary
import pickle

device = torch.device("cuda")

adap_thres=10 ** 8

column_names = ['year',
                'month',
                'day',
                'hour',
                'company_id',
                'device',
                'MA/CRM',
                'SFA',
                'URL',
                'stay_seconds',
                'day_of_week',
                #  'revisit',
                'reaction']

# prepare data
data = pd.read_csv('./data/action_history/preprocessed/call_chat_summary.20220901-20220902.encoded.csv', dtype='Int64')
single_data = data.iloc[1, :].drop("reaction").tolist()
print("single_data", single_data)

# load token2id
with open('./output_pretraining/action_history/vocab_token2id.bin', 'rb') as p:
    vocab_dic = pickle.load(p)

# transfer data to input_ids
vocab = Vocabulary(adap_thres, target_column_name="reaction")
sep_id = vocab.get_id(vocab.sep_token, special_token=True)

vocab_ids = []
for jdx, field in enumerate(single_data):
    vocab_id, _ = vocab_dic[column_names[jdx]][field]
    vocab_ids.append(vocab_id)

vocab_ids.append(sep_id)

input_ids = torch.tensor([vocab_ids], dtype=torch.long)

# load model
model = CommonModel()
model.to(device)
model.load_state_dict(torch.load("./output_fine_tuning/action_history/fine_tuning_model.pth"))
model.eval()
with torch.no_grad():
  output = model(input_ids.to(device))
  print("output", output)
  pred = torch.argmax(output, 1)
  print("pred", pred)


