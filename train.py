import json
import numpy as np
from torch.utils.data import DataLoader
import torch
from nltk_utils import bag_of_words, tokenize, stem, cv
from chat_dataset import ChatDataset
from model import NeuralNet
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
FILE = "data.pth"

with open('intents.json', 'r') as f:
    jsonData = json.load(f)

tags = []
ignore_words = ['?', '.', '!']
corpus = []
y_train = []
for idx, intent in enumerate(jsonData['intents']):
    label = idx
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        pattern = tokenize(pattern)
        pattern = [stem(word) for word in pattern if word not in ignore_words]
        sentence = ' '.join(pattern)
        corpus.append(sentence)
        # y: PyTorch CrossEntropyLoss needs only class labels, not One-hot encoded
        y_train.append(label)

X_train = bag_of_words(corpus)
y_train = np.array(y_train)
# print(cv.get_feature_names_out())

# Hyperparameters
epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = NeuralNet(input_size, hidden_size, output_size).to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(dtype=torch.float).to(DEVICE)
        labels = labels.to(dtype=torch.long).to(DEVICE)

        # Forward pass
        outputs = model(words)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "intents": jsonData["intents"],
    "tags": tags,
}

torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
