# Newtral tech test

Fine-tune a BERT model on a spanish collection for fact checking.

### Reproduce

To reproduce the experiment locally, you will need a GPU and enough disk space (\~5GB). The following steps will:
* download, prepare and featurize data
* search hyperparameters
* fine-tune the model and validate the results by doing it 10 times and averaging results, saving the best model.


```bash
python src/dl_data.py https://ml-coding-test.s3.eu-west-1.amazonaws.com/ml_test_data.csv -o data
python src/prepare_data.py data/ml_test_data.csv --dev_size 0.1 --test_size 0.2 --random_state 42
python src/featurize.py data --splits train dev test
python src/modeling.py data --output_dir models --hypersearch validation_steps=10
```

`config.yaml` contains the already searched hyper-parameters, if you just want to train with those issue:
```bash
python src/modeling.py data --output_dir models --train --evaluate
```

The default model is [BERTin](https://huggingface.co/bertin-project/bertin-roberta-base-spanish), a spanish pre-trained model, based on RoBERTa, if you want to use other model, just pass `--model_name` to the necessary steps (featurize and modeling). The fine-tuned model is [here](https://drive.google.com/drive/folders/1zQxgsLKcudnZCDWpcXU94OSGrw5szhmA?usp=sharing), it was trained with two Nvidia GeForce RTX 2080 TI with 12GB each, python 3.8.3 and CUDA 10.2

### Results

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99       925
               1       0.92      0.81      0.87        75
    
        accuracy                           0.98      1000
       macro avg       0.95      0.90      0.93      1000
    weighted avg       0.98      0.98      0.98      1000

    Global
         f1-score   0.86
         accuracy   0.98
        micro avg   0.98
        macro avg   0.92

### Notebook

You can also run this in a Colab Notebook (__only evaluation, not training__), to do so, [pick the notebook](https://github.com/geblanco/newtral_technical_test/blob/master/Newtral_tech_test_Guillermo_E_.ipynb) and run it in Colab. You will need a fine-tuned model present in `/content/drive/My Drive/Colab Notebooks/models`, you can download it [here](https://drive.google.com/drive/folders/1zQxgsLKcudnZCDWpcXU94OSGrw5szhmA?usp=sharing).

# LICENCE
BSD 3-Clause License

Copyright (c) 2021, Guillermo Blanco
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.