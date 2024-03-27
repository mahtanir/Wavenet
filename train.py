from wavenet import *  
import datetime
from dataloader import * 
import numpy as np
from mxnet import ndarray

class Train():

    def __init__(self, config) -> None:
        self.layer_size = config.layer_size 
        self.stack_size = config.stack_size #repeat 

        self.res_channels = config.res_channels
        self.skip_channels = config.skip_channels

        self.mu = 256 
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        self.seq_length = config.seq_length 
    
    def save_params(model):
        model.save_params('models/best_perf/' + datetime.datetime.now())

    def build_model(self):
        net = Wavenet(self.res_channels, self.mu, self.skip_channels, 2, self.layer_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        n_steps = self.batch_size
        fs, data = load_music('data_parametric-2')
        minLoss = None 
        data_generator = data_generation(data, fs, self.seq_length, self.mu, None)
        for i in range(self.epochs):
            loss = 0 
            for j in range(self.batch_size): #assuming that the batch size is full training set, stochastic gradient descent 
                sample = next(data_generator)
                #Forward Pass
                x = data_generator[:-1] #one behind 
                y = data_generation[-x.shape[0]:] #normal (effectively one forward)
                y_hat = net(x)
                loss = loss + criterion(y_hat, y)
                print(loss.shape)
                loss.backward() #predicts loss across each step for all categories. Only true cat matters though.I.e loss is (1, sample)
                optimizer.step()
                optimizer.zero_grad() #stochastic 
            agg_loss = torch.sum(minLoss) / self.batch_size
            # ndarray.sum(loss).asscalar()
            if (loss is None or loss < minLoss): #stochastic volative, so look per batch which is best
                    minLoss = loss
                    self.save_params(net)
    
    def generate_slow(self, x, model, n, dilation_depth, n_repeat):
         dilations = [2*i for i in  range(dilation_depth)] * n_repeat
         reference_window = sum(dilations)
         x_generated = x.copy()
         for i in range(n):
              y = model(x_generated)
            #   y = model(x_generated[-reference_window -1: ]) ALR
              y_next = np.array(y.argmax(1))[-1] #n, c, samples
              x_generated.append(y_next) 
              #similar to LSTM logic but now instead of reference window of 1, you have reference window of dilations 
              #still add the prev output! 
         return x_generated
    

    
    # LOGIC: Note that for the last output, elements beyond the reference window don't affect the subsequent output. 
    # as such we simply need to consider the reference window only. Reference window is the sum of the dilations for last point. 
    #i.e always consider left most point contributing towards it! Always + dilation for reference window. 
    #But what is x too small ie starts from 0? Not sure about - 1 also but still works 
    
    # def generate_slower(self, x, models, dilation_depth, n_repeat, ctx, n):
    #     dilations = [2**i for i in range(dilation_depth)] * n_repeat 
    #     res = list(x.asnumpy())
    #     for _ in trange(n):
    #         x = nd.array(res[-sum(dilations)-1:],ctx=ctx) i.e losing (k - 1)*dilation every time. So we sum all --> here only looking at 1 skip conn output
    #         y = models(x)
    #         res.append(y.argmax(1).asnumpy()[-1])
    #     return res
         

        







        
        
