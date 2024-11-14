import torch

def test(model, data_loader, threshold=0.5):
    model.eval()
    results = []
    with torch.no_grad():
        for images in data_loader:
            outputs = model(images)
            # Process outputs
            results.append(outputs)
    return results