import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classess):
        super().__init__()
        # don't need to flatten the input tensor as it is already in the correct shape to be passed to a classification model.
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classess)
            # no activation and no softmax at the end
            # due to using nn.CrossEntropyLoss (nn.LogSoftMax + nn.NLLLoss)
        )

    def forward(self, X):
        # X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits
