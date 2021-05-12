# Text Generator

Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code, or quickly train on a text using a pretrained model.

text-generator is a Python 3 module on top of [jjmagicsolver](https://github.com/jjmagicsolver/text-generator) for creating many cool features:

* A modern neural network architecture which utilizes new techniques as attention-weighting and skip-embedding to accelerate training and improve model quality.
* Train on and generate text at either the character-level or word-level.
* Configure RNN size, the number of RNN layers, and whether to use bidirectional RNNs.
* Train on any generic input text file, including large files.
* Train models on a GPU and then use them to generate text with a CPU.
* Utilize a powerful CuDNN implementation of RNNs when trained on the GPU, which massively speeds up training time as opposed to typical LSTM implementations.
* Train the model using contextual labels, allowing it to learn faster and produce better results in some cases.

## Examples

```python
from textgenerator import textgenerator

textgen = textgenerator()
textgen.generate()
```

```text
[Spoiler] Anyone else find this post and their person that was a little more than I really like the Star Wars in the fire or health and posting a personal house of the 2016 Letter for the game in a report of my backyard.
```

The included model can easily be trained on new texts, and can generate appropriate text *even after a single pass of the input data*.

```python
textgen.train_from_file('hacker_news_2000.txt', num_epochs=1)
textgen.generate()
```

```text
Project State Project Firefox
```

The model weights are relatively small (2 MB on disk), and they can easily be saved and loaded into a new textgenerator instance. As a result, you can play with models which have been trained on hundreds of passes through the data. (in fact, textgenerator learns *so well* that you have to increase the temperature significantly for creative output!)

```python
textgen_2 = textgenerator('/weights/hacker_news.hdf5')
textgen_2.generate(3, temperature=1.0)
```

```text
Why we got money “regular alter”

Urburg to Firefox acquires Nelf Multi Shamn

Kubernetes by Google’s Bern
```

You can also train a new model, with support for word level embeddings and bidirectional RNN layers by adding `new_model=True` to any train function.

## Interactive Mode

It's also possible to get involved in how the output unfolds, step by step. Interactive mode will suggest you the *top N* options for the next char/word, and allows you to pick one.  
  
When running textgenerator in the terminal, pass `interactive=True` and `top=N` to `generate`. N defaults to 3.

```python
from textgenerator import textgenerator

textgen = textgenerator()
textgen.generate(interactive=True, top_n=5)
```

![word_level_demo](/docs/textgenerator_interactive.gif)
  
This can add a *human touch* to the output; it feels like you're the writer! ([reference](https://fivethirtyeight.com/features/some-like-it-bot/))
  
## Usage

textgenerator can be installed [from pypi](https://pypi.python.org/pypi/textgenerator) via `pip`:

```sh
pip3 install textgenerator
```

For the latest textgenerator, *you must have a minimum TensorFlow version of 2.1.0*.

You can view a demo of common features and model configuration options in [this Jupyter Notebook](/docs/textgenerator-demo.ipynb).

`/datasets` contains example datasets using Hacker News/Reddit data for training textgenerator.

`/weights` contains further-pretrained models on the aforementioned datasets which can be loaded into textgenerator.

`/outputs` contains examples of text generated from the above pretrained models.

## Neural Network Architecture and Implementation

textgenerator is based off of the [char-rnn](https://github.com/karpathy/char-rnn) project by [Andrej Karpathy](https://twitter.com/karpathy) with a few modern optimizations, such as the ability to work with very small text sequences.

![default model](/docs/default_model.png)

The included pretrained-model follows a [neural network architecture](https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/model_def.py) inspired by [DeepMoji](https://github.com/bfelbo/DeepMoji). For the default model, textgenerator takes in an input of up to 40 characters, converts each character to a 100-D character embedding vector, and feeds those into a 128-cell long-short-term-memory (LSTM) recurrent layer. Those outputs are then fed into *another* 128-cell LSTM. All three layers are then fed into an Attention layer to weight the most important temporal features and average them together (and since the embeddings + 1st LSTM are skip-connected into the attention layer, the model updates can backpropagate to them more easily and prevent vanishing gradients). That output is mapped to probabilities for up to [394 different characters](/textgenerator/textgenerator_vocab.json) that they are the next character in the sequence, including uppercase characters, lowercase, punctuation, and emoji. (if training a new model on a new dataset, all of the numeric parameters above can be configured)

![context model](/docs/context_model.png)

Alternatively, if context labels are provided with each text document, the model can be trained in a contextual mode, where the model learns the text *given the context* so the recurrent layers learn the *decontextualized* language. The text-only path can piggy-back off the decontextualized layers; in all, this results in much faster training and better quantitative and qualitative model performance than just training the model gien the text alone.

The model weights included with the package are trained on hundreds of thousands of text documents from Reddit submissions ([via BigQuery](http://minimaxir.com/2015/10/reddit-bigquery/)), from a very *diverse* variety of subreddits. The network was also trained using the decontextual approach noted above in order to both improve training performance and mitigate authorial bias.

When fine-tuning the model on a new dataset of texts using textgenerator, all layers are retrained. However, since the original pretrained network has a much more robust "knowledge" initially, the new textgenerator trains faster and more accurately in the end, and can potentially learn new relationships not present in the original dataset (e.g. the [pretrained character embeddings](http://minimaxir.com/2017/04/char-embeddings/) include the context for the character for all possible types of modern internet grammar).

Additionally, the retraining is done with a momentum-based optimizer and a linearly decaying learning rate, both of which prevent exploding gradients and makes it much less likely that the model diverges after training for a long time.

## Notes

* **You will not get quality generated text 100% of the time**, even with a heavily-trained neural network. That's the primary reason viral [blog posts](http://aiweirdness.com/post/170685749687/candy-heart-messages-written-by-a-neural-network)/[Twitter tweets](https://twitter.com/botnikstudios/status/955870327652970496) utilizing NN text generation often generate lots of texts and curate/edit the best ones afterward.

* **Results will vary greatly between datasets**. Because the pretrained neural network is relatively small, it cannot store as much data as RNNs typically flaunted in blog posts. For best results, use a dataset with at least 2,000-5,000 documents. If a dataset is smaller, you'll need to train it for longer by setting `num_epochs` higher when calling a training method and/or training a new model from scratch. Even then, there is currently no good heuristic for determining a "good" model.

* A GPU is not required to retrain textgenerator, but it will take much longer to train on a CPU. If you do use a GPU, I recommend increasing the `batch_size` parameter for better hardware utilization.

## Future Plans for textgenerator

* More formal documentation

* A web-based implementation using tensorflow.js (works especially well due to the network's small size)

* A way to visualize the attention-layer outputs to see how the network "learns."

* A mode to allow the model architecture to be used for chatbot conversations (may be released as a separate project)

* More depth toward context (positional context + allowing multiple context labels)

* A larger pretrained network which can accommodate longer character sequences and a more indepth understanding of language, creating better generated sentences.

* Hierarchical softmax activation for word-level models (once Keras has good support for it).

* FP16 for superfast training on Volta/TPUs (once Keras has good support for it).
