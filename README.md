# Chatbot

The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers and 1 output layer.

## Built With

[![PyTorch][PyTorch-logo]][PyTorch-url]

## 📦 Installation

### Create an environment

```console
$ git clone https://github.com/CaoHoangKiet222/Chatbot
$ cd Chatbot
$ python3 -m venv venv
```

### Activate it

Mac / Linux:

```console
. venv/bin/activate
```

Windows:

```console
venv\Scripts\activate
```

### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:

```console
pip install nltk
```

## 🚀 Usage

```console
python3 chat.py
```

## ⚙ Customize

Have a look at [intents.json](./json/intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.

```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```

# :books: Bibliography

---

**[1]** Wikipedia, "FeedForward Neural Network," Wikipedia, [Online]. Available: https://en.wikipedia.org/wiki/Feedforward_neural_network.

**[2]** T. Wood, "Activation Function," DeepAI, [Online]. Available: https://deepai.org/machine-learning-glossary-and-terms/activation-function.

**[3]** PyTorch, "Neural Network" [Online]. Available: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

**[4]** PyTorch, "Optimizing Model Parameters" [Online]. Available: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

<h1 align="center">🌟 Good Luck and Cheers! 🌟</h1>

[PyTorch-logo]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
