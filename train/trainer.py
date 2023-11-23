import torch
import numpy as np
from torchvision.transforms import ToTensor


class Trainer():
    def __init__(self, model, loss, optimizer, cfg):
        self.model = model.to('cuda')
        self.model.train()
        self.loss = loss
        self.optim = optimizer
        self.cfg = cfg

    def train(self, dataloader):
        epochs = self.cfg['epochs']
        for epoch in range(epochs):
            inputs, labels = dataloader.get_items(0, len(dataloader))
            to_tensor = ToTensor()
            for n, input in enumerate(inputs):
                inputs[n] = to_tensor(input).unsqueeze(0)
            # inputs = torch.cat(inputs)
            # labels = torch.Tensor(labels).long()
            inputs = torch.cat(inputs).to('cuda')
            labels = torch.Tensor(labels).long().to('cuda')

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



