# Experiment

<p align="center">
  <img src="https://github.com/andrew-m-holmes/transformer/blob/main/experiment/metrics.jpg" alt="Metrics Graph" width=648 height=432>
</p>

---

## Introduction

This page of the repository provides an overview of an English to German language translation experiment, the results of which are depicted in the metrics graph at the top of this page. On this page, you can learn how to use different tools and modules to interact with the files contained in the experiment.

This guide will cover:

- Reading the `config` into a dictionary.

- Loading the `Tokenizer` and `DataLoader` modules.
  
- Loading a `Checkpoint` for retraining.
  
- Prompting a `Transformer` model from model weights.
  
> __Note__: _If you haven't already, please make sure you view the outline of this repository on the [homepage](https://github.com/andrew-m-holmes/Transformer/) for general information._

---

This section contains several important files:

- `config.txt`: This is the configuration file for this experiment. It defines hyperparameters and other settings used in the experiment.

- `dataloader.pt` and `testloader.pt`: These files store the `DataLoader` objects for this experiment. Once loaded, they can be used for training and testing a `Transformer` model.

- `tokenizer.pt`: This file contains the `Tokenizer` object used for tokenization in this experiment. Tokenization is needed for any process that requires converting a sequence of text into token IDs.

- `checkpoint.pt`: This file stores the `Checkpoint` object which includes modules like the state of the `Transformer` model and metrics like the BLEU scores. Once loaded, you can use it for retraining or retrieve its modules for other purposes.

- `model.pt`: This file stores the weights of the trained `Transformer` model. You can load these weights to use the trained model for prompting or other similar tasks.

- `log.txt`: This file contains logging information recorded during the training of this experiment. This can be useful for troubleshooting your own experiment or as a benchmark.

- `metrics.jpg`: This file contains two graphs that plot the training and testing losses against the epochs, and the BLEU scores against the epochs. The graph offers a visualization of how the model performed during training and may be used as a reference for your own experiment or as a benchmark. 

> **Note**: _Understanding these files can be extremely helpful for utilizations not outlined in the tutorials below._

--- 

## Getting Started

In order to fully utilize the contents of this page, you must be familiar with the process of deserializing the serialized files. This section will guide you through the process of bringing these modules back into python. On top of that, it will give you an idea of how to use certain functions and/or modules!

As you progress through each section, please be aware that each tutorial is designed to build upon the previous one. To ensure coherent tutorials and a trouble free experience, it's crucial to complete the tutorials in the order they're presented.

---

### Parsing Config

First, we will parse the `config.txt` file that stores important parameters to pass to functions and modules. You can parse the config using `parse_config()` which stores the parameters and their values into a dictionary. 

```python3
# path/to/Transformer/
from utils.functional import parse_config

config = parse_config("experiment/config.txt", verbose=True)
maxlen = config["maxlen"]
print(f"Maxlen: {maxlen}")
```

__Output__:

```
Config parsed
Maxlen: 256
```

---

### Loading Tokenizer and DataLoader

With your ability to parse `config.txt`, we can now move on to loading the `Tokenizer` and `DataLoader` objects. These can both be loaded by using the `load_module()` function.

```python3
from utils.functional import load_module

tokenizer = load_module("experiment/tokenizer.pt", verbose=True)
dataloader = load_module("experiment/dataloader.pt", verbose=True)
```

__Output__:

```
Module loaded
Module loaded
```

---

### Loading Checkpoint

Now that we have our configuration from `config.txt`, a `Tokenizer`, and `Dataloader`, we can use their attributes and the `load_checkpoint()` function to load stored modules into a fresh `Checkpoint`.

```python3
import torch
from utils.checkpoint import Checkpoint
from model.transformer import Transformer

device = "cuda" if torch.cuda.is_available() else "cpu" # use correct device

# create required modules for checkpoint
vocab_english, vocab_german = tokenizer.vocab_size()
pad_id = tokenizer.getitem("<pad>", module="source")
model = Transformer(vocab_english, vocab_german, maxlen, pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

checkpoint = Checkpoint(model, optimizer)
checkpoint.load_checkpoint(path="experiment/checkpoint.pt", verbose=True, device=device)
```

__Output__:

```
Checkpoint loaded
```

---

### Retraining

After getting the `Checkpoint` and access to other parameters and modules that are used for retraining, we can combine them to pass to the `retrain()` function that retrains the model.

```python3
from utils.train import retrain

retrain(dataloader, checkpoint, epochs=1000, warmups=100, clip=4.0, verbose=True, device=device)
```

__Output__:

```
----------------------------------------------------------------------
|                          Training Resumed                          |
----------------------------------------------------------------------
-------------------------------------------------------------------
|                            Epoch 701                            |
-------------------------------------------------------------------
|        Epoch Duration: 00:00:51 | Elapsed Time: 10:40:29        |
|       Train Loss: 0.9044 | Test Loss: 2.3045 | BLEU: 25.0       |
|          Gradient Norm: 4.0 | Learning Rate: 5.815e-07          |
|           Warmup Step: False | Checkpoint Saved: False          |
-------------------------------------------------------------------
```
> **Note**: _The following output was retrieved from the actual experiment and does not reflect the expected output from the example above._

---

### Prompting

Assuming training is complete and the `Transformer` model has achieved an acceptable performance level, we can prompt the model to do English to German language translation.  

In order to prompt, we need to grab our `Tokenizer`, our trained `Transformer` model, and we'll have to build a `DecoderSearch` (e.g. `Beam` or `Greedy`). We can retrieve our model from our `Checkpoint` or by loading the weights using `model.pt` and the `load_model()` function.

```python3
model = checkpoint["model"]
```

___or...___

```python3
from utils.functional import load_model

model = Transformer(vocab_english, vocab_german, maxlen, pad_id)
load_model(model, path="experiment/model.pt", verbose=True, device=device)
```

__Output__:

```
Model params loaded
```

> **Note**: _The reason we re-initialize our `Transformer` model is because we've already loaded the weights in the 'Loading Checkpoint' tutorial. By doing this, we ensure the model weights loaded are derived from `model.pt` solely. Since both weights are identical in this experiment, this is purely for demonstration purposes._

___With that out of the way...___

```python3
from utils.search import Beam
from utils.test import prompt

# use tokenizer and its special tokens for DecoderSearch
sos, eos = tokenizer.getitem("<sos>", module="source"), tokenizer.getitem("<eos>", module="source")
search = Beam(sos, eos, maxlen, width=3, fast=False)
sequence = input("Enter in the sequence of text: ")
translation = prompt(sequence, model, tokenizer, search, device=device)
print(f"Translation: {output}")
```

__Output__:

```
Enter in the sequence of text: Hello transformer
Translation: ein feuerwehrmann, der sich an der arbeit unter freiem himmel
```
Clearly it needs some fine tuning haha...

---

### One last thing, Quantization!

Using a fully trained `Transformer` without the necessary compute like a GPU, can cause some computational overhead. Thankfully, we can compress the model, similar to how we compress photos to make them more exportable online. This can be achieved through quantization; with it we can change the percision of the `Embedding` weights and the `Linear` layer weights and biases to a smaller percision (e.g. 16-bit float, or 8-bit inetger).

```python3
from utils.quantize import quantize_model

model.to("cpu") # must be on CPU

model_int8 = quantize_model(model, dtype=torch.qint8, inplace=False)

print(f"Size of normal model: {model_size(model):.1f}MB, Number of Parameters: {parameter_count(model):.1f}M")
print("vs.")
print(f"Size of quantized model: {model_size(model_int8):.1f}MB, Number of Parameters: {parameter_count(model_int8):.1f}M")
```

__Output__:

```
Size of normal model: 259.1MB, Number of Parameters: 64.5M
vs.
Size of quantized model: 66.2MB, Number of Parameters: 64.5M
```

Once quantized, the size your model takes in your memory is much smaller and the model will run much faster on your CPU, all while maintaining relatievely the same performance as the original model.

```python3
import time

text = (
        """
        Self-driving cars are revolutionizing the way we travel. By utilizing advanced algorithms and sensors,
        they can detect obstacles, traffic, and road conditions in real-time. 
        """
        )

# basic model test
start = time.time()
output = prompt(text, model, tokenizer, beam, early_stop=True, device=None)
end = time.time()
print(output)
print(f"Total time in seconds for Normal Transformer (On CPU): {(end - start):.2f} s")

# quantized model test
start = time.time()
output = prompt(text, model_int8, tokenizer, beam, early_stop=True, device=None)
end = time.time()
print(output)
print(f"Total time in seconds for Quantized Transformer (On CPU): {(end - start):.2f} s")
```

__Output__:

```
man mit autos - kostümen und der es schneit auf einem mann setzt sich hin, dass der ob sie die meisten der junge frau und die meisten davon aussieht.
Total time in seconds for Normal Transformer (On CPU): 3.40 s
man mit autos - kostümen und der es schneit auf diesem mädchen kommt an, wobei eine frau und die gerade einen spaziergang geworfen hat.
Total time in seconds for Quantized Transformer (On CPU): 1.83 s
```

## Conclusion

Congratulations! You've successfully completed all the tutorials that help you utilize this experiment. Like before, please contribute using [_Issues_](https://github.com/andrew-m-holmes/Transformer/issues), [_Pull Requests_](https://github.com/andrew-m-holmes/Transformer/pulls), or the [_Discussions_](https://github.com/andrew-m-holmes/Transformer/discussions/1) section. Also, please reach out to me on [LinkedIn](https://www.linkedin.com/in/andrewmicholmes/) and check out any articles I have on [Medium](https://medium.com/@andmholm).

Thank you!
