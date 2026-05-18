import torch
import os

from morphingdb_test.config import (
    reasoning_cross_encoder_output_path,
    reasoning_deberta_output_path,
    reasoning_flant5_output_path
)
from .cross_encoder import CrossEncoderWrapper
from .model_wrappers import DebertaQAWrapper, FlanT5Wrapper


def export_cross_encoder():
    wrapper = CrossEncoderWrapper()
    wrapper.eval()

    dummy = torch.zeros((32, 2, 128), dtype=torch.float32)

    traced_model = torch.jit.trace(wrapper, dummy)
    traced_model.save(reasoning_cross_encoder_output_path)

    print(f"TorchScript CrossEncoder model saved to {reasoning_cross_encoder_output_path}")


def export_deberta_qa():
    wrapper = DebertaQAWrapper()
    wrapper.eval()

    dummy = torch.zeros((32, 2, 256), dtype=torch.float32).cpu()

    traced_model = torch.jit.trace(wrapper, dummy)
    traced_model.save(reasoning_deberta_output_path)

    print(f"TorchScript DeBERTa QA model saved to {reasoning_deberta_output_path}")


def export_flan_t5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = FlanT5Wrapper(device=device)
    wrapper.eval()

    dummy = torch.zeros((32, 2, 256), dtype=torch.float32).to(device)

    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, dummy)

    traced_model.save(reasoning_flant5_output_path)

    print(f"TorchScript FLAN-T5 model saved to {reasoning_flant5_output_path}")


def export_all_models():
    os.makedirs(os.path.dirname(reasoning_cross_encoder_output_path), exist_ok=True)
    export_cross_encoder()
    export_deberta_qa()
    export_flan_t5()


if __name__ == "__main__":
    export_all_models()
