# MDM - Mixed Distribution Model

## Discrete Distribution

Let there be a distribution of $k$ possibilites. Let there also be $n$ distributions $X_1$, $X_2$,...,$X_n$. Then, a mixed distribution, M, is a linear combination of the distributions:

$$M = \sum_{i=1}^{n} p_i X_i$$

where $\sum_{i=1}^n p_i = 1$

Then, then given the counts of observations $a_1$, $a_2$,...,$a_k$, MDM_softmax find a p such maximises:

$$ Pr(a_1,a_2,...,a_k | X, p) $$ 
