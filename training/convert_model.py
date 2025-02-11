import torch
from CNN import CNN

def load_model(fold):
    model_path = f"checkpoints/{fold}/best-checkpoint.ckpt"
    model = CNN.load_from_checkpoint(
        f"{model_path}",
        input_dim=60,
        fold=fold,
        classes_names=["mescolare", "tagliare", "stare_fermo"]
    ).to("cpu")
    model.eval()
    return model

def torch_to_onnx(model, dummy_input):
    torch.onnx.export(
        model,
        dummy_input,
        f"{model.id()}.onnx",
        export_params=True,
        verbose=True,
        opset_version=10
    )

def main():
    info = "fold_1"
    dummy_input = torch.randn(1, 60, 6)
    model = load_model(info)
    torch_to_onnx(model, dummy_input)

if __name__ == '__main__':
    main()

