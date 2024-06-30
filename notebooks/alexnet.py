import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1, input_size=[1, 480, 640]):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.input_size = input_size
        self.final_feature_size = self.calculate_final_feature_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.final_feature_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )

    def calculate_final_feature_size(self):
        with torch.no_grad():
            input_tensor = torch.zeros(1, *self.input_size)
            output_tensor = self.features(input_tensor)
            return output_tensor.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet(num_classes=1, input_size=[1, 480, 640]):
    return AlexNet(num_classes=num_classes, input_size=input_size)
