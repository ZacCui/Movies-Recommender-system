import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
import torch.optim as optimizer
from Model import Model
import torch

# get data from file
def get_data(filename):
    data = pd.read_table(filename, delimiter='\t', header=None,
                         names=['UserID', 'Movie', 'Rating', 'Timestamp'],
                         engine='python', encoding='latin-1')
    data.drop('Timestamp', axis=1, inplace=True)
    total = data.values.shape[0]
    return data.values, total
    


num_user = 943 + 1
num_movie = 1682 + 1
num_feature = 20

training_file = ['./ml-100k/u1.base','./ml-100k/u2.base','./ml-100k/u3.base','./ml-100k/u4.base','./ml-100k/u5.base',]
test_file = ['./ml-100k/u1.test','./ml-100k/u2.test','./ml-100k/u3.test','./ml-100k/u4.test','./ml-100k/u5.test']

train_mean = 0
test_mean = 0

# write result to file
file = open("embedding_result.txt","w")
for traning_f, test_f in zip(training_file, test_file):
    # the model has been described in the report
    # implementation section ----> neural network with embedding leayer 
    model = Model(num_movie, num_user, num_feature)
    
    # I choose Stochastic gradient descent with learning rate 0.005
    op = optimizer.SGD(model.parameters(), lr = 0.005)

    # I choose mean squre error as the loss function
    loss_func = nn.MSELoss()

    # get the traning data
    data, data_total = get_data(traning_f)
    # get the test data
    test, test_total = get_data(test_f)

    cur_train_loss = 10
    cur_test_loss = 10
    for e in range(20):
        error = 0
        round = 1

        # shuffle the traning data in each epoch
        np.random.shuffle(data)
        for i, j, k in data:
            k -=3
            model.zero_grad()
            u = Variable(torch.LongTensor([i]))
            m = Variable(torch.LongTensor([j]))
            r = Variable(torch.FloatTensor([k]))
            
            # prediction
            predict = model(m,u)

            # calculate the loss
            # since we just train one data every time
            # the mean squre error is equal to the squre error of this data
            loss = loss_func(predict, r)

            # add error
            error += loss.item()

            # backpropagation
            loss.backward()
            op.step()
            if round % 2000 == 0:
                print("epoch: "+str(e)+"  round "+str(round)+"   traning loss: "+str(np.sqrt(error/round)))
            round += 1
        print("train loss:", np.sqrt(error/data_total))

        if np.sqrt(error/data_total) < cur_train_loss:
            cur_train_loss = np.sqrt(error/data_total)

        error = 0
        round = 1
        for i, j, k in test:
            model.zero_grad()
            k -= 3
            u = Variable(torch.LongTensor([i]))
            m = Variable(torch.LongTensor([j]))
            predict = model(m,u)
            loss = abs(predict.item() - k) ** 2
            error += loss
            if round % 2000 == 0:
                print("epoch: "+str(e)+"  round "+str(round)+"   test loss: "+str(np.sqrt(error/round)))
            round += 1
        print("test loss:", np.sqrt(error/test_total))

        if np.sqrt(error/test_total) < cur_test_loss:
            cur_test_loss = np.sqrt(error/test_total)
        
    train_mean += cur_train_loss
    test_mean += cur_test_loss
    file.write(traning_f+" loss: "+str(cur_train_loss) +"  |  "+ test_f+" loss: "+str(cur_test_loss)+"\n")

file.write("overall train loss: "+str(train_mean/5) + "   overall test loss: " + str(test_mean/5))
file.close()
print("overall train loss: "+str(train_mean/5) + "   overall test loss: " + str(test_mean/5))
