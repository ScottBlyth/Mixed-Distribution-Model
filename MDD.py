# MDM -> mixed distribution model (discrete)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

def MDM_softmax(distributions, observations, iterations=1000, alpha=0.1,epsilon = 10e-5,
                    prob_plot=[], prob_history=[], plot = True):
    X = np.array(distributions).T
    k,n = X.shape 
    q = np.random.uniform(-1,1, size=n)
    p = np.exp(q)
    p = p / np.sum(p)
    a = observations / np.sum(observations)
    I = np.ones(n)
    knack_vec = np.zeros((k,n))
    for j in range(min(k,n)):
        knack_vec[j,j] = 1
    seq_to_n = np.arange(n)
    for _ in range(iterations):
        pred = X @ p
        a_Q_X = (a/pred) @ X
        f = np.vectorize(lambda j : (knack_vec[j] - p[j]) * a_Q_X @ p)
        s = f(seq_to_n)
        q = q + alpha*s
        p_prev = p.copy()
        p = np.exp(q)
        S = np.sum(p)
        p = p / S
        diff = p-p_prev
        if plot:
            log_prob = 0
            for i in range(k):
                log_prob += observations[i] * np.log(X[i] @ p)
            prob_plot.append(log_prob)
            prob_history.append(p.copy())
        if np.sum(s*s) <= epsilon*epsilon:
            return p
    return p

if __name__ == "__main__":
    x1 = np.array([0.1, 0.8, 0.1])
    x2 = np.array([0.1, 0.1, 0.8])
    x3 = np.array([0.4, 0.4, 0.2])
    X_t = np.array([x1, x2, x3])
    obs = np.array([50, 100, 74])
    prob_plot = []
    prob_history = []
    p = MDM_softmax(X_t, obs, iterations=10**4, alpha=0.01, prob_plot=prob_plot,
                                    prob_history=prob_history)
    print(p)
    print(X_t.T @ p * np.sum(obs))

    print(X_t.T @ p)

    plt.bar(np.arange(X_t.T.shape[0]),X_t.T @ p * np.sum(obs), 
                    label='Fit', width=0.25, edgecolor='black')
    plt.bar(np.arange(X_t.T.shape[0])+0.25, obs, label='Observed', width=0.25,
                                        edgecolor='black')
    plt.xticks(np.arange(X_t.T.shape[0]))
    plt.legend(['Fit', 'Observed'])
    plt.ylim(0, 1.25*max(obs))
    plt.show()
    plt.plot(prob_plot)
    plt.show()
    prob_history = np.array(prob_history)
    plt.scatter(prob_history.T[0], prob_history.T[1], s=1)
    plt.show()
