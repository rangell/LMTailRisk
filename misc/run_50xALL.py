import torch
import time

torch.cuda.init()
device = torch.device("cuda")

model = torch.nn.Linear(1024 * 8, 1024)
optimizer = torch.optim.Adam(model.parameters())

print(
    "Let's keep all ", torch.cuda.device_count(), "GPUs busy!", end="\r\n", flush=True
)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

batch_size = 10 * 1024 * torch.cuda.device_count()

while True:
    inputs = torch.randn(batch_size, 1024 * 8, device=device)
    targets = torch.randn(batch_size, 1024, device=device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()

    time.sleep(0.0001)
