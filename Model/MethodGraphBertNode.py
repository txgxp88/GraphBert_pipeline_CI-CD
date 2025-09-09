import torch
from Model.MethodGBert import MethodGraphBert
from transformers.models.bert.modeling_bert import BertPreTrainedModel

class MethodGraphBertNodeConstruct(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.0001
    weight_decay = 5e-4
    max_epoch = 200
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertNodeConstruct, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.x_size)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        x_hat = self.cls_y(sequence_output)

        return x_hat