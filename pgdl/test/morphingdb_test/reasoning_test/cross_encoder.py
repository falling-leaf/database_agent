import torch
from sentence_transformers import CrossEncoder

from morphingdb_test.config import reasoning_cross_encoder_model_path


class CrossEncoderWrapper(torch.nn.Module):

    def __init__(self, model_path=None):
        super().__init__()
        if model_path is None:
            model_path = reasoning_cross_encoder_model_path
        cross_model = CrossEncoder(model_path)
        self.model = cross_model.model
        self.model.eval()
        self.model.cpu()

    def forward(self, inputs):
        input_ids = inputs[:, 0, :].long()
        attention_mask = inputs[:, 1, :].long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs.logits
