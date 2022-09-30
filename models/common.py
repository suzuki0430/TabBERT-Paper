from transformers import BertConfig, BertModel
import torch.nn as nn
import torch

class CommonModel(nn.Module):
    def __init__(self):
        super(CommonModel, self).__init__()
        # self.config = BertConfig.from_pretrained("./output_card/checkpoint-35000/config.json")
        # self.model = BertModel.from_pretrained("./output_card/checkpoint-35000/pytorch_model.bin", config=self.config)
        self.config = BertConfig.from_pretrained("./output_pretraining/action_history/checkpoint-500/config.json")
        self.model = BertModel.from_pretrained("./output_pretraining/action_history/checkpoint-500/pytorch_model.bin", config=self.config)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, batch_first=True)
        self.regressor = nn.Linear(self.config.hidden_size, 2) # 2クラスで

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        out, _ = self.lstm(outputs[0], None)
        # out, _ = self.lstm(outputs['last_hidden_state'], None)
        sequence_output = out[:, -1, :]
        logits = self.regressor(sequence_output)

        return logits

    def loss_fn(self, logits, label):
        # loss = torch.sqrt(nn.MSELoss(reduction='mean')(logits[:, 0], label))
        loss = torch.sqrt(nn.MSELoss(reduction='mean')(logits, label))
        return loss