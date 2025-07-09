from transformers import AutoModel
model = AutoModel.from_pretrained(
                "GoodBaiBai88/M3D-CLIP",
                trust_remote_code=True
            )