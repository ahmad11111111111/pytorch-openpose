import torch

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        backend = 'CUDA'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
        backend = 'MPS'
    else:
        device = torch.device('cpu')
        backend = 'CPU'

    print(f"Selected backend: {backend}")
    x = torch.randn(1, 3, 64, 64, device=device)
    model = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device).eval()
    with torch.no_grad():
        y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == '__main__':
    main()
