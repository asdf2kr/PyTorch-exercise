import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegrssion, self).__init__()
        # Hyper-parametrs
        self.input_size = 1
        self.output_size = 1
        self.linear = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

epochs = 500
lr = 0.0001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

net = linearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = lr)

for epoch in range(epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = net(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

# Plot the graph
predicted = net(torch.from_numpy(x_train)).detach().numpy()
print(predicted)
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
