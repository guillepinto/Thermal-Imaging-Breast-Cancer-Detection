""" 
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
"""

# Pytorch essentials
import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
     super().__init__()
    # Doble bloque convolucional al inicio de Xception
     self.double_conv = nn.Sequential(
         nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=2, bias=False),
         nn.BatchNorm2d(out_channels//2),
         nn.ReLU(inplace=True),
         nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, bias=False),
         nn.BatchNorm2d(out_channels),
         nn.ReLU(inplace=True)
      )

  def forward(self, x):
    return self.double_conv(x)
  
class SeparableConv2d(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()

    self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=False)
    self.one_by_one = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

  def forward(self, x):
    x = self.depth_wise_conv(x)
    x = self.one_by_one(x)
    return x

class XceptionModule(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, relu_at_start=True):
    super().__init__()

    # first one by one
    self.one_by_one = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
      nn.BatchNorm2d(out_channels)
    )

    if relu_at_start:
      self.double_depth_wise_conv = nn.Sequential(
        nn.ReLU(inplace=False),
        SeparableConv2d(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
        SeparableConv2d(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      )
    else:
      self.double_depth_wise_conv = nn.Sequential(
        SeparableConv2d(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
        SeparableConv2d(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      )

  def forward(self, x):
    x1 = self.one_by_one(x)
    x2 =self.double_depth_wise_conv(x)
    x = torch.add(x1, x2)
    return x

class EntryFlowModule(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()

    # 2d double convolution at start
    self.double_conv = DoubleConvBlock(in_channels, 64)

    self.block1 = XceptionModule(64, 128, relu_at_start=False)
    self.block2 = XceptionModule(128, 256)
    self.block3 = XceptionModule(256, out_channels)
  
  def forward(self, x):
    x = self.double_conv(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    return x

class XceptionMiddleModule(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()

    # triple separable conv 2d
    self.triple_depth_wise_conv = nn.Sequential(
      nn.ReLU(inplace=True),
      SeparableConv2d(in_channels, out_channels),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      SeparableConv2d(out_channels, out_channels),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      SeparableConv2d(out_channels, out_channels),
      nn.BatchNorm2d(out_channels),
    )
  
  def forward(self, x):
    x = self.triple_depth_wise_conv(x)
    return x

class MiddleFlowModule(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.middle_flow = nn.Sequential()
    for _ in range(8):
      self.middle_flow.append(XceptionMiddleModule(in_channels, out_channels))

  def forward(self, x):
    x = self.middle_flow(x)
    return x
  
class XceptionExitModule(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()

    # first one by one
    self.one_by_one = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
      nn.BatchNorm2d(out_channels)
    )

    self.double_depth_wise_conv = nn.Sequential(
      nn.ReLU(inplace=False),
      SeparableConv2d(in_channels, in_channels),
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=False),
      SeparableConv2d(in_channels, out_channels),
      nn.BatchNorm2d(out_channels),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

  def forward(self, x):
    x1 = self.one_by_one(x)
    x2 =self.double_depth_wise_conv(x)
    x = torch.add(x1, x2)
    return x

class ExitFlowModule(nn.Module):
  def __init__(self, in_channels, n_classes):
    super().__init__()

    self.block1 = XceptionExitModule(in_channels, 1024)
    self.block2 = nn.Sequential(
      SeparableConv2d(1024, 1536),
      nn.BatchNorm2d(1536),
      nn.ReLU(inplace=True),
      SeparableConv2d(1536, 2048),
      nn.BatchNorm2d(2048),
      nn.ReLU(inplace=True),
    )

    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.last_fc = nn.Linear(2048, n_classes)
  
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.gap(x)
    x = x.view(x.size(0), -1)
    x = self.last_fc(x)
    return x
  
class Xception(nn.Module):
  def __init__(self, n_channels, n_classes):
    super().__init__()
    self.entry_flow = EntryFlowModule(n_channels, 728)
    self.middle_flow = MiddleFlowModule(728, 728)
    self.exit_flow = ExitFlowModule(728, n_classes)

  def forward(self, x):
    x = self.entry_flow(x)
    x = self.middle_flow(x)
    x = self.exit_flow(x)
    return x
  
def xception(n_channels: int, n_classes: int):
    """
    Construct Xception.
    """

    model = Xception(n_channels=n_channels, n_classes=n_classes)
    return model

# Test the model to see if it gives the expected result.

# input_image = torch.rand([2, 1, 299, 299])
# print(f"Entrada: {input_image.size(), input_image.dtype}")
# model = Xception(n_channels=1, n_classes=1)
# ouput = model(input_image)
# print(f"Salida: {ouput.size(), ouput.dtype}")