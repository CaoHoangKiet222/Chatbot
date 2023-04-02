import torch
import random
from train import DEVICE, FILE, ignore_words
from model import NeuralNet
from nltk_utils import stem, tokenize, apply_bag_of_words

data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data['tags']
intents = data['intents']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(DEVICE)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    sentence = [stem(word) for word in sentence if word not in ignore_words]
    sentence = ' '.join(sentence)
    X = apply_bag_of_words([sentence])
    X = torch.from_numpy(X).to(dtype=torch.float).to(DEVICE)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    probs = torch.softmax(output, dim=1)

    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents:
            if tags[predicted.item()] == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
