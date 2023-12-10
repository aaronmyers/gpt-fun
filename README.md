# GPT fun project.

This repo is dedicated to building different GPT models and having fun with their output. This may result in a blog article :)


## Experiments:

Each experiment generates 500 new tokens after being trained on Shakespeare data.


1. 1.9 million parameters

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



2. 5 million parameters:

```

```


