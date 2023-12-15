import torch
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
            running_loss = 0.0
            for i in range(int(len(dataloader)/32)):
                inputs, labels = dataloader.get_items(i*32, (i+1)*32)
                to_tensor = ToTensor()
                for n, input in enumerate(inputs):
                    inputs[n] = to_tensor(input).unsqueeze(0)
                inputs = torch.cat(inputs).to('cuda')
                labels = torch.Tensor(labels).long().to('cuda')

                outputs = self.model.forward(inputs)
                self.optim.zero_grad()
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}', i)

        print("Training complete.")



