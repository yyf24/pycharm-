import torch
import numpy as np
import torch.nn as nn
x_values=[i for i in range(11)]
x_train=np.array(x_values,dtype=np.float32)
x_train=x_train.reshape(-1,1)

y_values=[2*i+1 for i in x_values]
y_train=np.array(y_values,dtype=np.float32)
y_train=y_train.reshape(-1,1)

class linemodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        out=self.linear(x)
        return out
model=linemodel()

print(model)

epochs=1000
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),learning_rate)
Loss=nn.MSELoss()

for i in range(epochs):
    i+=1
    inputs=torch.from_numpy(x_train)
    labels=torch.from_numpy(y_train)

    optimizer.zero_grad()
    outputs=model(inputs)

    loss=Loss(outputs,labels)

    loss.backward()

    optimizer.step()

print(model(torch.from_numpy(x_train)).data.numpy())