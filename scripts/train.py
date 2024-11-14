import torch
from tqdm import tqdm
from config import NUM_EPOCHS  # NUM_EPOCHS를 config에서 임포트

def train(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()