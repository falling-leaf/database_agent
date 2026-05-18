import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

from morphingdb_test.config import (
    reasoning_deberta_model_path,
    reasoning_flant5_model_path
)


class DebertaQAWrapper(torch.nn.Module):

    def __init__(self, model_path=None):
        super().__init__()
        if model_path is None:
            model_path = reasoning_deberta_model_path
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.eval()
        self.model.cpu()

    def forward(self, inputs):
        device = next(self.model.parameters()).device

        input_ids = inputs[:, 0, :].long().to(device)
        attention_mask = inputs[:, 1, :].long().to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        return start_logits, end_logits


class FlanT5Wrapper(torch.nn.Module):

    def __init__(self, model_path=None, device=None):
        super().__init__()
        if model_path is None:
            model_path = reasoning_flant5_model_path
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)

    def forward(self, inputs):
        model_device = next(self.model.parameters()).device
        inputs = inputs.to(model_device)

        input_ids = inputs[:, 0, :].long()
        attention_mask = inputs[:, 1, :].long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids
        )

        logits = outputs.logits
        return logits.cpu()
