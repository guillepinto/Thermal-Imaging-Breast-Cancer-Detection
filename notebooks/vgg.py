import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, num_classes=1, input_size=[1, 250, 333]):
        super(VGGNet, self).__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.input_size = input_size
        self.final_feature_size = self.calculate_final_feature_size()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            #nn.Linear(self.final_feature_size, 4096),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def calculate_final_feature_size(self):
        size_h, size_w = self.input_size[1:]
        size_h = self.conv_output_size(size_h, 3, 1, 1) // 2  # bloque 1
        size_w = self.conv_output_size(size_w, 3, 1, 1) // 2  # bloque 1
        size_h = self.conv_output_size(size_h, 3, 1, 1) // 2  # bloque 2
        size_w = self.conv_output_size(size_w, 3, 1, 1) // 2  # bloque 2
        size_h = self.conv_output_size(size_h, 3, 1, 1) // 2  # bloque 3
        size_w = self.conv_output_size(size_w, 3, 1, 1) // 2  # bloque 3
        size_h = self.conv_output_size(size_h, 3, 1, 1) // 2  # bloque 4
        size_w = self.conv_output_size(size_w, 3, 1, 1) // 2  # bloque 4
        size_h = self.conv_output_size(size_h, 3, 1, 1) // 2  # bloque 5
        size_w = self.conv_output_size(size_w, 3, 1, 1) // 2  # bloque 5

        return size_h * size_w * 512

    def conv_output_size(self, size, kernel_size, stride, padding):
        return (size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

def vgg(num_classes=1, input_size=[1, 250, 333]):
    return VGGNet(num_classes=num_classes, input_size=input_size)
