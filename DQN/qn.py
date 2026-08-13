import torch.nn as nn               # Εισαγωγή των layers του PyTorch
import torch.optim as optimizer     # Εισαγωγή του optimizer module
import torch.nn.functional as F 

class QN(nn.Module):

    def __init__(self,Size,Hidden2,Hidden3,Actions):
        super(QN,self).__init__()

        self.fc1 = nn.Linear(Size,Hidden2)
        self.fc2 = nn.Linear(Hidden2,Hidden3)
        self.fc3 = nn.Linear(Hidden3,Actions)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def get_weight(self):

        weights1 = self.fc1.weight
        weights2 = self.fc2.weight
        weights3 = self.fc3.weight

        bias1 = self.fc1.bias
        bias2 = self.fc2.bias
        bias3 = self.fc3.bias

        return weights1,weights2,weights3,bias1,bias2,bias3



