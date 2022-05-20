from turtle import forward
from boto import config
import torch
from torch import embedding, nn
from transformers import BertConfig, BertModel, BertForSequenceClassification, RobertaConfig, RobertaModel, RobertaForSequenceClassification, ElectraConfig, ElectraModel, ElectraForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

def get_pooled_embedding(feature, attention_mask, method):
    if method == 'cls_with_pooler':
        return feature.pooler_output
    feature = feature.hidden_states
    if method == 'first_last_avg':
        feature = feature[-1] + feature[-0]
    elif method == 'last_avg':
        feature = feature[-1]
    elif method == 'last_2_avg':
        feature = feature[-1] + feature[-2]
    elif method == 'cls':
        return feature[-1][:, 0, :]
    else:
        raise Exception("Unknown pooling method {}".format(method))
    
    if attention_mask is None:
        return torch.mean(feature, dim=1)
    else:
        return torch.sum(feature * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)

class XQBert(nn.Module):
    def __init__(self, backbone="bert", num_labels=2, dropout=None, embedding_method='cls_with_pooler', reinit_layers=0):
        super(XQBert, self).__init__()
        if backbone == "bert":
            self.config = BertConfig(
                num_labels=num_labels,
                classifier_dropout=dropout
            )
            self.backbone = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=self.config)
        elif backbone == "roberta":
            self.config = RobertaConfig.from_pretrained(
                "roberta-base",
                num_labels=num_labels,
                classifier_dropout=dropout
            )
            self.backbone = RobertaForSequenceClassification.from_pretrained("roberta-base", config=self.config)
        elif backbone == "electra":
            self.config = ElectraConfig.from_pretrained(
                "google/electra-base-discriminator",
                num_labels=num_labels,
                classifier_dropout=dropout
            )
            self.backbone = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator", config=self.config)
        else:
            raise ValueError("Unsupported backbone model {}".format(backbone))

        self.reinit_layers = reinit_layers
        if self.reinit_layers > 0:
            self._reinit(backbone)
        # self.backbone_config = BertConfig(
        #     output_hidden_states=True
        # )
        # self.num_labels = num_labels
        # self.backbone = BertModel.from_pretrained("bert-base-uncased", config=self.backbone_config)
        # self.embedding_method = embedding_method
        # self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.backbone_config.hidden_size, num_labels)
        # self.loss_func = nn.CrossEntropyLoss()

        # for param in self.backbone.parameters():
        #     param.requires_grad = True
    def _reinit(self, backbone):
        if backbone == "bert":
            if self.backbone.bert.pooler is not None:
                self.backbone.bert.pooler.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                self.backbone.bert.pooler.dense.bias.data.zero_()
                for param in self.backbone.bert.pooler.parameters():
                    param.requires_grad = True
            for n in range(self.reinit_layers):
                self.backbone.bert.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
        elif backbone == "roberta":
            if self.backbone.roberta.pooler is not None:
                self.backbone.roberta.pooler.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                self.backbone.roberta.pooler.dense.bias.data.zero_()
                for param in self.backbone.roberta.pooler.parameters():
                    param.requires_grad = True
            for n in range(self.reinit_layers):
                self.backbone.roberta.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
        elif backbone == "electra":
            if self.backbone.electra.pooler is not None:
                self.backbone.electra.pooler.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                self.backbone.electra.pooler.dense.bias.data.zero_()
                for param in self.backbone.electra.pooler.parameters():
                    param.requires_grad = True
            for n in range(self.reinit_layers):
                self.backbone.electra.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
        else:
            raise ValueError("Unsupported backbone model {}".format(backbone))
    
    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) 

    def forward(self, x):
        return self.backbone(**x)
        # output = self.backbone(
        #     input_ids=x['input_ids'],
        #     attention_mask=x["attention_mask"],
        #     token_type_ids=x["token_type_ids"]
        # )
        # embedding = get_pooled_embedding(output, x["attention_mask"], self.embedding_method)
        # logits = self.classifier(self.dropout(embedding))
        # if "labels" in x.keys():
        #     loss = self.loss_func(logits.view(-1, self.num_labels), x["labels"].view(-1))
        # else:
        #     loss = None
        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=None,
        #     attentions=None
        # )

class XQSBert(nn.Module):
    def __init__(self, backbone="bert", num_labels=2, dropout=None, embedding_method='cls_with_pooler'):
        super(XQSBert, self).__init__()
        if backbone == "bert":
            self.config = BertConfig(
                output_hidden_states=True
            )
            self.backbone = BertModel.from_pretrained("bert-base-uncased", config=self.config)
        elif backbone == "roberta":
            self.config = RobertaConfig.from_pretrained(
                "roberta-base",
                output_hidden_states=True
            )
            self.backbone = RobertaModel.from_pretrained("roberta-base", config=self.config)          
        elif backbone == "electra":
            self.config = ElectraConfig(
                output_hidden_states=True
            )
            self.backbone = ElectraModel.from_pretrained("google/electra-base-discriminator", config=self.config)
        else:
            raise ValueError("Unsupported backbone model {}".format(backbone))

        self.classifier = nn.Linear(3 * self.config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.embedding_method = embedding_method
        if dropout is None:
            dropout = self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        output_a = self.backbone(
            input_ids=x['input_ids_a'],
            attention_mask=x["attention_mask_a"],
            token_type_ids=x["token_type_ids_a"]
        )
        output_b = self.backbone(
            input_ids=x["input_ids_b"],
            attention_mask=x["attention_mask_b"],
            token_type_ids=x["token_type_ids_b"]
        )
        embedding_a = get_pooled_embedding(output_a, x["attention_mask_a"], self.embedding_method)
        embedding_b = get_pooled_embedding(output_b, x["attention_mask_b"], self.embedding_method)
        embeddings = torch.concat([embedding_a, embedding_b, torch.abs(embedding_a - embedding_b)], dim=-1)
        logits = self.classifier(self.dropout(embeddings))
        if "labels" in x.keys():
            loss = self.loss_func(logits.view(-1, self.num_labels), x["labels"].view(-1))
        else:
            loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )