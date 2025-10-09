import torch
from transformers import RTDetrV2ForObjectDetection

MODEL_ID = "PekingU/rtdetr_v2_r18vd"
class RTDETRWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state


print("Loading model...")
model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID)
model.eval()

print("Tracing model...")
# Dummy input: 1x3x640x640 RGB image
example = torch.rand(1, 3, 640, 640)
wrapped = RTDETRWrapper(model)
traced = torch.jit.trace(wrapped, example)
print("Saving...")
traced.save("rtdetr_v2_traced.pt")
print("✅ Saved TorchScript model: rtdetr_v2_traced.pt")
