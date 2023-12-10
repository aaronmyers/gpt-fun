# GPT fun project.

This repo is dedicated to building different GPT models and having fun with their output. This may result in a blog article :)

Some of the basics:
We use a vocabulary size of 65, this is mostly capital letters, lowercase letters, and some symbols. We do not do any tokenizing here, we're keeping it simple.

The model we use is the basic transformer architecture and we will primarily vary the number of attention heads and layers as a means of adjusting the parameter size across experiments.

We keep some variables the same throughout: droput rate (0.2), learning rate (0.0003), block size (256), and batch size (64).

The source text is from karapathy tinyshakespeare, we also leverage his code (with modifications) for the base GPT model in python using PyTorch.

This repo also includes some basic MNIST model in a single file, using both non-torch models and torch models.


## Experiments:

Each experiment generates 500 new tokens after being trained on Shakespeare data.


1. 1.9 million parameters, 100 training steps

Parameter Input:

```
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 500
learning_rate = 3e-4
device = 'mps' #running on a mac M1 chip and want to use gpu
eval_iters = 200
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
```

### Output:

```
1.922369 M parameters
step 0: train loss 4.2132, val loss 4.2179
step 99: train loss 2.4696, val loss 2.4846

AUET:
! zantorothafis the mede ith manyee odKirFofu plit

Then we-
GADUSTESBucr trss RINWhell y str m'lone&s sthte lld funced
SOROF blryss ltr, int,

Tigru
HEd iatatheinO:
TIYombe:
Wh I mo bes,oroxcr, monk:
A breswut rfousthin, hil:
Th Her ke his e it what
Pav' 'troatheerr, ain ff acofials, lllo y t llo hy beeends, that!
Whe;
My he.
Hou sisut,
NGof OMave moth ot rsldatherthe BEghaipaticole: he mberrtethowad h arereomys cary ihana t tothats k.
He helo my I bre bly s:
He

QBune an td t fouser:

I
```
It is clear that there are no real words in here, but it does somewhat capture the format of a play script. Let's crank it up.

2. 1.9 Million parameters, 5000 training epochs:

Parameter Input:

```
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # training steps
eval_interval = 500
learning_rate = 3e-4
device = 'mps' #running on a mac M1 chip and want to use gpu
eval_iters = 200
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
```

### Output:

```
1.922369 M parameters
step 0: train loss 4.2132, val loss 4.2179
step 500: train loss 1.7942, val loss 1.9369
step 1000: train loss 1.6637, val loss 1.8477
step 1500: train loss 1.6101, val loss 1.8098
step 2000: train loss 1.5770, val loss 1.7859
step 2500: train loss 1.5680, val loss 1.7839
step 3000: train loss 1.5558, val loss 1.7818
step 3500: train loss 1.5387, val loss 1.7602
step 4000: train loss 1.5357, val loss 1.7708
step 4500: train loss 1.5278, val loss 1.7597
step 4999: train loss 1.5258, val loss 1.7574

Ayong she in wick'd din clains the frig to Plifamon I;
Is ay, viry the you nof that though timeltomon her thour,
As narrant:
How had it to me his wurpowerds, God lipatory trunkenct the our out and it slewls eveat the and thine blood', free dear my goes bortunden.

LUCLEOffit thy.
O, 'le gracioun.

MENENIUS:
Stay, in or hope for Hen IOLKDow:
And its morch!--
O, let, wity, or soular, he cup,
Thy y shave thunk our us not charden him, bed;
No bot gette, who irth thouse crown bid;
And Sa XI:
Coment t
```

Well, this is...better? It is better at grasping the script format and there are SOME real words in there. Let's increase the parameter count.


3. 5 million parameters:

```

```


