import torch

ckpt = torch.load("trained_transformer_jeff.pth", map_location="cpu")
torch.save(ckpt["model_state_dict"], "trained_transformer_jeff_weights.pth")
print("Done!")