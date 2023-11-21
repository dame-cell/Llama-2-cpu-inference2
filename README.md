# MPT 30B inference code using CPU

Run inference on the latest llama2 model using your CPU. This inference code uses a [ggml](https://github.com/ggerganov/ggml) quantized model. To run the model we'll use a library called [ctransformers](https://github.com/marella/ctransformers) that has bindings to ggml in python.



## Requirements

I recommend you use docker for this model, it will make everything easier for you. Minimum specs system with 5.37 GB of ram. Recommend to use `python 3.10`.


## Setup

First create a venv.

```sh
python -m venv env && source env/bin/activate
```

Next install dependencies.

```sh
pip install -r requirements.txt
```

Next download the quantized model weights (about 19GB).

```sh
python download_model.py
```

Ready to rock, run inference.

```sh
python inference.py
```

To try out the RAG code you can run this code in the terminal.

```sh
python rag.py
```

The idea for this code was inspired by - [https://github.com/abacaj/mpt-30B-inference](https://github.com/abacaj/mpt-30B-inference)
