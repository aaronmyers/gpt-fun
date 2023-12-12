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


3. 3.6 million parameters, 5000 training steps:

### Input:
```
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' #running on a mac M1 chip and want to use gpu
eval_iters = 200
n_embd = 384
n_head = 2
n_layer = 2
dropout = 0.2

```


### Output:

```
3.695681 M parameters
step 0: train loss 4.3467, val loss 4.3441
step 500: train loss 1.7172, val loss 1.8691
step 1000: train loss 1.5225, val loss 1.7265
step 1500: train loss 1.4380, val loss 1.6635
step 2000: train loss 1.3804, val loss 1.6276
step 2500: train loss 1.3329, val loss 1.5926
step 3000: train loss 1.3032, val loss 1.5715
step 3500: train loss 1.2868, val loss 1.5762
step 4000: train loss 1.2734, val loss 1.5691
step 4500: train loss 1.2542, val loss 1.5705
step 4999: train loss 1.2383, val loss 1.5605

Senon't here own.


QUEEN MARGARET:
Fear me this mine own. This and bud arms of you
Beforth.

GLOUCESTER:
Why, we'll matter I adventant you misgursed fore?

KING EDWARD IV:
Therefof York.

KIs Oxford, to your heir lordshire's shall not leight with wad.
Thou bloody death not was,, the queen is a full yarr.

GLOUCESTER:
Man him; for look there very
Matuntion ben timelecters us, bred nerieve to to his he,
Are to Clarence out heirs. Has, my sorrows
But banish blame and let the royal welcome here,
Th

```

4. 10.7 million parameters, 5,000 training steps:

## Input:
```
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' #running on a mac M1 chip and want to use gpu
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

```


## Output:
```
10.788929 M parameters
step 0: train loss 4.2221, val loss 4.2306
step 500: train loss 1.7449, val loss 1.9065
step 1000: train loss 1.3934, val loss 1.6012
step 1500: train loss 1.2631, val loss 1.5233
step 2000: train loss 1.1869, val loss 1.5130
step 2500: train loss 1.1197, val loss 1.4873
step 3000: train loss 1.0710, val loss 1.4810
step 3500: train loss 1.0172, val loss 1.4981
step 4000: train loss 0.9627, val loss 1.5118
step 4500: train loss 0.9142, val loss 1.5426
step 4999: train loss 0.8590, val loss 1.5666

a lie one another thing there was to be spoken;
On what by-father's coffinds I saw yours,
That's full of sour ride?

COMINIUS:
It may be.
How are the modest senate's corn;
The voice of you read, and you content:
A present to private your person which finds
Live us in a crotten from fasting go to-dinner;
Commidiation when they were enough bound,
And thou deputy forth? Once bulishre,
In earth, whose littes of contomplace care
Into simputatishing of the king suit
And whose absence refer the dire du

```

5. 21.4 million parameters, 5,000 training steps:

## Input:
```
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' #running on a mac M1 chip and want to use gpu
eval_iters = 200
n_embd = 384
n_head = 12
n_layer = 12
dropout = 0.2
```


## Output:
```
21.428801 M parameters
step 0: train loss 4.3112, val loss 4.3210
step 500: train loss 1.9159, val loss 2.0204
step 1000: train loss 1.4057, val loss 1.6085
step 1500: train loss 1.2402, val loss 1.5061
step 2000: train loss 1.1399, val loss 1.4772
step 2500: train loss 1.0533, val loss 1.4919
step 3000: train loss 0.9591, val loss 1.5144
step 3500: train loss 0.8595, val loss 1.5688
step 4000: train loss 0.7581, val loss 1.6402
step 4500: train loss 0.6597, val loss 1.7168
step 4999: train loss 0.5574, val loss 1.8041


RIVERS:
Romeo! madam, lay'st thou lake not where.

QUEEN:
But I have forgot that heart's sake.
How now, sweet reason, that I should not stay;
For by any hours ear minister.

runage:
The senators and women's fire joints are taken
Makes Lovel amazable two why I that your spotted lady.

Lord:
Amen.

Second Murderer:
I saw you him.

CATESBY:
A little man have at your country and the people,
colourer; a match, being feast, I would there's beats.

First Murderer:
My believes is not so.

SLY:
Here is

```

6. 
