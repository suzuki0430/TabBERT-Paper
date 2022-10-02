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
input_data = data.drop("reaction", axis=1).values.tolist()
labels = data['reaction'].values.tolist()

# load token2id
with open('./output_pretraining/action_history/vocab_token2id.bin', 'rb') as p:
    vocab_dic = pickle.load(p)

# transfer data to input_ids
vocab = Vocabulary(adap_thres, target_column_name="reaction")
sep_id = vocab.get_id(vocab.sep_token, special_token=True)

user_vocab_ids = []

for data in input_data:
    vocab_ids = []
    for jdx, field in enumerate(data):
        vocab_id, _ = vocab_dic[column_names[jdx]][field]
        vocab_ids.append(vocab_id)
    vocab_ids.append(sep_id)
    user_vocab_ids.append(vocab_ids)

# load model
model = CommonModel()
model.to(device)
model.load_state_dict(torch.load("./output_fine_tuning/action_history/fine_tuning_model.pth"))
model.eval()

# calculate F1
tp = 0
tn = 0
fp = 0
fn = 0

for user_vocab_id, label in zip(user_vocab_ids, labels):
    input_ids = torch.tensor([user_vocab_id], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids.to(device))
        pred = torch.argmax(output, 1).item()
      
    if pred == label:
        if pred == 1:
          tp += 1
        else:
          tn += 1
    elif pred == 0:
        fn += 1
    elif pred == 1:
        fp +=1
    
    print("tp", tp)
    print("tn", tn)
    print("fn", fn)
    print("fp", fp)

recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * precision * recall / (precision + recall)

print("recall", recall)
print("precision", precision)
print("f1_score", f1_score)