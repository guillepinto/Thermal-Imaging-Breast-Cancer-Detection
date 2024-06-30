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
        size_h, size_w = self.input_size[1:]
        size_h = self.conv_output_size(size_h, 11, 4, 2) // 2
        size_w = self.conv_output_size(size_w, 11, 4, 2) // 2
        size_h = self.conv_output_size(size_h, 5, 1, 2) // 2
        size_w = self.conv_output_size(size_w, 5, 1, 2) // 2
        size_h = self.conv_output_size(size_h, 3, 1, 1)
        size_w = self.conv_output_size(size_w, 3, 1, 1)
        size_h = self.conv_output_size(size_h, 3, 1, 1)
        size_w = self.conv_output_size(size_w, 3, 1, 1)
        size_h = self.conv_output_size(size_h, 3, 1, 1) // 2
        size_w = self.conv_output_size(size_w, 3, 1, 1) // 2
        return size_h * size_w * 256

    def conv_output_size(self, size, kernel_size, stride, padding):
        return (size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet(num_classes=1, input_size=[1, 480, 640]):
    return AlexNet(num_classes=num_classes, input_size=input_size)
