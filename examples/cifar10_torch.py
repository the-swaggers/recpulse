import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

torch.manual_seed(42)
DEVICE = 'cuda'
BS = 64
NUM_EPOCHS = 100

print("Loading CIFAR-10...")
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='../data/cifar10_torch', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data/cifar10_torch', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)

class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.25)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.25)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        h = torch.relu(self.conv1a(x))
        h = torch.relu(self.conv1b(h))
        h = self.drop1(self.pool1(h))

        h = torch.relu(self.conv2a(h))
        h = torch.relu(self.conv2b(h))
        h = self.drop2(self.pool2(h))

        h = torch.relu(self.conv3a(h))
        h = torch.relu(self.conv3b(h))
        h = self.drop3(self.pool3(h))

        h = h.reshape(x.shape[0], 256 * 4 * 4)
        h = torch.relu(self.fc1(h))
        h = self.drop4(h)
        return self.fc2(h)

model = VGGNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
criterion = nn.CrossEntropyLoss()

num_params = sum(p.numel() for p in model.parameters())
print(f"Params: {num_params:,}")
print(f"Training: {NUM_EPOCHS} epochs, batch_size={BS}, device={DEVICE}")
print(f"{'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'LR':>10}  {'Time':>6}")
print("-" * 50)

best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    tl = 0.0; n = 0
    start = time.time()

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        tl += loss.item(); n += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

    elapsed = time.time() - start
    avg_loss = tl / n

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct / total

    scheduler.step(avg_loss)
    lr = optimizer.param_groups[0]['lr']

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../data/cifar10_best_torch.pt")

    print(f"{epoch+1:5d}  {avg_loss:8.4f}  {acc:7.2%}  {lr:10.6f}  {elapsed:5.0f}s", flush=True)

print(f"\nBest accuracy: {best_acc:.2%}")
