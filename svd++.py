#!/usr/bin/python3

import numpy as np
import pandas as pd

"""
    COMP9417 Assignment 2
    SVD++ implementation
"""

"""
    Training function
    Args:
        trainset    the trainset used to train
        mu          movies rated by each user
        n_users     number of users
        n_movies    number of movies
        n_genre     number of genres
        global_mean average of all ratings
        lr          learning rate
        reg         regulizing rate
        bu          user bias
        bm          movie bias
        fu          user factor
        fm          movie factor
        impn        implicit movie factor
"""

def sgd(trainset, mu, n_users, n_movies, n_genre, global_mean, lr, reg, bu = None, bm = None, fu = None, fm = None, impn = None):
    # initiate model if not exist
    randgen = np.random.mtrand._rand
    bu = np.zeros(n_users, np.double) if bu is None else bu
    bm = np.zeros(n_movies, np.double) if bm is None else bm
    fu = randgen.normal(0, .1, (n_users, n_genre)) if fu is None else fu
    fm = randgen.normal(0, .1, (n_movies, n_genre)) if fm is None else fm
    impn = randgen.normal(0, .1, (n_movies, n_genre)) if impn is None else impn

    lastu = -1
    Mu = []
    sqrt_Mu = 0
    for u, m, r in trainset:

        # get movies rated by u
        if (lastu != u):
            Mu = np.array(mu[u])
            lastu = u
            sqrt_Mu = np.sqrt(len(Mu))

        # compute user implicit feedback
        u_impl_fdb = (impn[[Mu]] / sqrt_Mu).sum(0)

        # compute current error
        dot = (fm[m] * (fu[u] + u_impl_fdb)).sum(0)
        err = r - (global_mean + bu[u] + bm[m] + dot)

        # update biases
        bu[u] += lr * (err - reg * bu[u])
        bm[m] += lr * (err - reg * bm[m])

        # update factors
        fuf = fu[u]
        fmf = fm[m]
        fu[u] += lr * (err * fmf - reg * fuf)
        fm[m] += lr * (err * (fuf + u_impl_fdb) - reg * fmf)
        impn[Mu] += lr * (err * fmf / sqrt_Mu - reg * impn[Mu])
    # print(bu.size, bm.size, fu.size, fm.size, impn.size)
    # exit()
    return bu, bm, fu, fm, impn

"""
    Test function
    Args:
        testset    the trainset used to train
        mu          movies rated by each user
        n_movies    number of movies
        bu          user bmas
        bm          movie bmas
        fu          user factor
        fm          movie factor
        impn          implicit movie factor
"""

def estimate(testset, n_movies, mu, bu, bm, fu, fm, impn):
    rret = np.zeros(len(testset))
    counter = 0

    for u, m in testset:
        est = global_mean
        doU = u<mu.size
        doM = m<n_movies

        # add user bias if have
        if doU:
            est += bu[u]

        # add movie bias if have
        if doM:
            est += bm[m]

        # add multiplied factors if have both
        if doU and doM:
            u_impl_feedback = (impn[mu[u]]).sum(0) / np.sqrt(len(mu[u]))
            # print(u,m,u_impl_feedback, n_movies)
            est += np.dot(fm[m], fu[u] + u_impl_feedback)

        # add result to return results
        rret[counter] = est
        counter += 1
    return rret


# helper function to read data
def get_dataset(file_name):
    data = pd.read_table(file_name,
                         delimiter='\t', header=None,
                         names=['UserID', 'MioveID', 'Rating', 'Timestamp'],
                         engine='python', encoding='latin-1')

    return list(zip(data.UserID, data.MioveID, data.Rating)), max(list(data.UserID))+1, max(list(data.MioveID))+1

n_genre = 20
test_loss_mean = 0
training_file = ['ml-100k/u1.base', 'ml-100k/u2.base', 'ml-100k/u3.base', 'ml-100k/u4.base', 'ml-100k/u5.base']
test_file = ['ml-100k/u1.test','ml-100k/u2.test','ml-100k/u3.test','ml-100k/u4.test','ml-100k/u5.test']

# cross-validation
for traning_f, test_f in zip(training_file, test_file):
    trainset, n_users, n_movies = get_dataset(traning_f)
    testset, _, _ = get_dataset(test_f)

    # initialize all args
    mu = [[] for _ in range(n_users)]
    global_mean = 0
    for u, m, r in trainset:
        mu[u].append(m)
        global_mean += r
    mu = np.array(mu)
    global_mean /= len(trainset)

    bu, bm, fu, fm, impn = [None]*5
    tu, tm, tr = zip(*testset)
    testset = list(zip(tu, tm))
    lastrmse = 10
    tname = test_f.split('/')[1].split('.')[0]
    tr = np.array(tr)

    # train & test
    # a good balance between training time and training outcome:
    # lr at 0.005
    # reg at 0.12
    for epoch in range(1000):
        bu, bm, fu, fm, impn = sgd(trainset, mu, n_users, n_movies, n_genre, global_mean, .005, .12, bu, bm, fu, fm, impn)
        er = estimate(testset, n_movies, mu, bu, bm, fu, fm, impn)
        rmse = np.sqrt(np.mean((er - tr) ** 2))
        print (tname, format(epoch, '03'), round(rmse, 6))
        if (lastrmse < rmse):
            print(tname, 'converges at', lastrmse)
            break
        lastrmse = rmse

    test_loss_mean += lastrmse

print("final test loss: ", test_loss_mean/5)