import torch
from torchvision.transforms import ToTensor


class Validatater():
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        model.eval()

    def eval(self, dataloader):
        with torch.no_grad():
            inputs, labels = dataloader.get_items(0, len(dataloader.test_data))
            to_tensor = ToTensor()
            for n, input in enumerate(inputs):
                inputs[n] = to_tensor(input).unsqueeze(0)
            inputs = torch.cat(inputs).to('cuda')
            labels = torch.Tensor(labels).long().to('cuda')
            
            outputs = self.model.forward(inputs)
            
            loss = self.loss(outputs, labels)

            # 정확도 계산
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += lables.size(0)
            
            accuracy = correct_predictions / total_samples
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')