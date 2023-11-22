import torch
import numpy as np
from torchvision.transforms import ToTensor


class Trainer():
    def __init__(self, model, loss, optimizer, cfg):
        self.model = model
        self.model.train()
        self.loss = loss
        self.optim = optimizer
        self.cfg = cfg

    def train(self, dataloader):
        epochs = self.cfg['epochs']
        for epoch in range(epochs):
            inputs, labels = dataloader.get_items(0, len(dataloader)-1)
            img_to_tensor = ToTensor()
            for n, input in enumerate(inputs):
                inputs[n] = img_to_tensor(input).unsqueeze(0)
            inputs = torch.cat(inputs)

            running_loss = 0.0
            outputs = self.model.forward(inputs)
            self.optim.zero_grad()
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optim.step()
            running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

        print("Training complete.")



