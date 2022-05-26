# QualityAdapt

This repo implements the Paper [*QualityAdapt: an Automatic Dialogue Quality Estimation Framework*](). 

QualityAdapt composes dialogue subqualities using adapter fusion for the task of overall dialogue quality estimation. Unlike similar approaches, QualityAdapt only requires a single forward pass on a Language Model to produce predictions for overall quality, thus reducing computational complexity.


| Section |
|-|
| [Abstract](#abstract) |
| [Apply QualityAdapt to your own dialogues](#apply-qualityadapt-to-your-own-dialogues) |
| [Training from scratch](#training-from-scratch) |
| [Implementation Details](#implementation-details) |
| [Citation](#citation) |

## Abstract 

> *Despite considerable advances in open-domain neural dialogue systems, their evaluation remains a bottleneck. Several automated metrics have been proposed to evaluate these systems, however, they mostly focus on a single notion of quality, or, when they do combine several sub-metrics, they are computationally expensive. This paper attempts to solve the latter: **QualityAdapt** leverages the Adapter framework for the task of Dialogue Quality Estimation. Using well defined semi-supervised tasks, we train adapters for different subqualities and score generated responses with AdapterFusion. This compositionality provides an easy to adapt metric to the task at hand that incorporates multiple subqualities. It also reduces computational costs as individual predictions of all subqualities are obtained in a single forward pass. This approach achieves comparable results to state-of-the-art metrics on several datasets, whilst keeping the previously mentioned advantages.*

## Apply QualityAdapt to your own dialogues

### 1. Download adapters

* Obtain the trained adapter and adapter fusion from AdapterHub or directly from [Drive]().

### 2. Run prediction code

* Adjust the example script `predict.py` to your requirements. You can choose from Parallel prediction of all subqualities or simply Overall Quality.

## Training from scratch
Here we use an example to illustrate how to reproduce the main results of the paper.

### 1. Download data

* Download the [preprocessed data]() you wish to train and/or evaluate on and insert it somewhere in your workspace.

### 2. Subquality Adapter training

* Adjust and run `train_u` and `train_s` to train the **U**nderstandability and **S**ensibleness Adapters, respectively.

### 3. Overall Quality Adapter Fusion training

* Adjust and run `train_h` to train the Adapter Fusion module on *Overall Quality* annotations.


## Implementation Details

### Tokenization

Raw dialog corpora may have very different data structures, so we leave to the user to convert their own data to QualityAdapt format.

The format is straightforward:

* The tokenizer receives as input `res` and optionally `ctx` (context is needed to evaluate context dependent metrics such as Sensibleness and Overall Quality). 
* `ctx` can be multi-turn, the only limitation relates to `max_length=124`. 
* Who said what is determined by appending the speaker token at the start of the sentence.

~~~
A: Gosh, you took all the word right out of my mouth. Let's go out and get crazy tonight.
B: Let's go to the new club on West Street .
A: I'm afraid I can't.


ctx = "<speaker1>Gosh , you took all the word right out of my mouth . Let's go out and get crazy tonight . <speaker2>Let's go to the new club on West Street ."
res = "<speaker1>I ' m afraid I can ' t ."
~~~


## Citation

If you use this work, please consider citing:

~~~

~~~