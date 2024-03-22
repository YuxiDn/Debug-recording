# RunTimeError: inplace operation

When I train my RNN model, there is a bug as follows:

RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [128, 512, 7, 7]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!

- I try to add 'torch.autograd.set_detect_anomaly(True)' to find where the bug is. 

- I also ask copilot, it advise me that I should set the variable 'inplace = False' in nn.ReLU . 

  class Resnet_module(nn.Module):

  ​    def __init__(self):

  ​        super(Resnet_module, self).__init__()

  ​        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

  ​        self.bn1 = nn.BatchNorm2d(64)

  ​        self.relu = nn.ReLU(inplace=False)  # 将 inplace 设置为 False

  ​    def forward(self, x):

  ​        x = self.conv1(x)

  ​        x = self.bn1(x)

  ​        x = self.relu(x)  # 现在这里不再是就地操作

  ​        return x

  - The problem is still not solved.

- I also tried to set the variable 'retain_graph = True' in loss.backward method.

  - loss.backward(retain_graph=True)

- **what help me at last**:

  - for the parameters in your network: '+=' operator is not permitted
  - Such as: x+=1, it should be changed as x = x +1