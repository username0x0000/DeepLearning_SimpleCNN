import torch
from torchvision.transforms import ToTensor


class Validatater():
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        model.eval()

    def eval(self, dataloader):
        with torch.no_grad():
            accuracy = 0
            correct_predictions, total_samples = 0,0
            for inputs, labels in dataloader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to('cuda').float()
                labels = labels.to('cuda')
            
                outputs = self.model.forward(inputs)
                
                loss = self.loss(outputs, labels)

                # 정확도 계산
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                
                accuracy = correct_predictions / total_samples
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')
        return accuracy
