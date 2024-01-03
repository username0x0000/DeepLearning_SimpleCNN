import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm


class Trainer():
    def __init__(self, model, loss, optimizer, cfg):
        self.model = model.to('cuda')
        # self.model.train()
        self.loss = loss
        self.optim = optimizer
        self.cfg = cfg

    def train(self, dataloader):
        epochs = self.cfg['epochs']
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(dataloader, desc='Processing', unit='Iteration'):
            # for i in range(int(len(dataloader)/64)):
            #     inputs, labels = next(iter(dataloader))
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to('cuda').float()
                labels = labels.to('cuda')

                outputs = self.model.forward(inputs)
                self.optim.zero_grad()
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

        print("Training complete.")
