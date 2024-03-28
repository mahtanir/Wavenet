from wavenet import *  
import datetime
from dataloader import * 
import numpy as np
from mxnet import ndarray
from tqdm import tqdm
from scipy.io.wavfile import write
from IPython.display import Audio
import sounddevice as sd




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
    
    def save_params(self, model):
        torch.save(model.state_dict(), 'models/best2/wavenet.pth')
        # model.save_params('models/best_perf/' + datetime.datetime.now())

    def train(self):
        print('at train')
        net = Wavenet(self.res_channels, self.mu, self.skip_channels, 2, self.stack_size, self.layer_size)
        self.net = net 
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        n_steps = self.batch_size
        print('loading music...')
        fs, data = load_music('data_parametric-2')
        minLoss = None 
        data_generator = data_generation(data, fs, self.seq_length, self.mu, None) #generate training data lazilly.
        for i in tqdm(range(self.epochs)):
            loss = 0 
            for j in tqdm(range(self.batch_size), leave=False): #assuming that the batch size is full training set, stochastic gradient descent 
                # print('epoch: ', i, '\n sample: ', j, '\n')
                sample = next(data_generator)
                #Forward Pass
                x = sample[:-1] #one behind 
                # x = x.astype('float64')
                # print('type x', type(x[0]))
                y = sample[-x.shape[0]:] #normal (effectively one forward)

                y = one_hot_utils(y)

                y_hat = net(x) #converted to one_hot already but in the right format for conv 
                # print('model_output: ', y_hat, '\n model shape:', y_hat.shape,
                    #    '\n test output: ', y, '\n test output shape', y.shape)
                # print('shape check', y_hat.shape, y.shape)
                tf_shape = (0, 2, 1) #so rows actually are the points! I THINK! but then the way the conv channel works is weird... But image also shows like this 
                y_hat = torch.permute(y_hat, tf_shape)


                loss_criterion = criterion(y_hat, y)
                loss = loss + loss_criterion.item()
                # print('loss criterion: ', loss_criterion, '\nloss: ', loss)
                loss_criterion.backward() #predicts loss across each step for all categories. Only true cat matters though.I.e loss is (1, sample) -> Actually loss criterion avg across samples 
                optimizer.step()
                optimizer.zero_grad() #stochastic 

            with torch.no_grad():
                 agg_loss = loss / self.batch_size
                 print(f"loss for epoch {i} : {agg_loss} \n")
            # ndarray.sum(loss).asscalar()
                 if (minLoss is None or agg_loss < minLoss): #stochastic volative, so look per batch which is best
                    minLoss = agg_loss
                    self.save_params(net)
        return net
    
    def bestModel(self): #load best model for given architecture 
        model =  Wavenet(self.res_channels, self.mu, self.skip_channels, 2, self.stack_size, self.layer_size)
        model.load_state_dict(torch.load('models/best/wavenet.pth'))
        self.net = model
        return model 
    
    def generate_slow(self, x, model, n, dilation_depth, n_repeat):
         dilations = [2*i for i in  range(dilation_depth)] * n_repeat
         reference_window = sum(dilations)
         x_generated = x.copy()
         for i in tqdm(range(n)):
              y = model(x_generated[-reference_window - 1:])
            #   y = model(x_generated[-reference_window -1: ]) ALR
              y_next = np.squeeze(y.argmax(1).numpy())[-1] #n, c, samples
            #   print(y.argmax(1).numpy(), np.squeeze(y.argmax(1).numpy())[-1])
              x_generated = np.append(x_generated, y_next)
              #similar to LSTM logic but now instead of reference window of 1, you have reference window of dilations 
              #still add the prev output! 
            #   print(x_generated.shape)
         return x_generated
    
    def generator(self, model, n, dilation_depth, n_repeat): 
         print('generating now...')
         fr, data = load_music('data_parametric-2')
         data_sample = data_generation_sample(data, fr, self.seq_length, self.mu, None)
         generated_song = self.generate_slow(data_sample, model, n, self.layer_size, self.stack_size)
         gen_wav = np.array(generated_song)
         decoded_wave = decode_mu_law(gen_wav, 256)
         np.save("wav_long.npy",decoded_wave)
        #  write('test.wav', fr, decoded_wave)
        #  sd.play(decoded_wave, fr)

        #  Audio(gen_wav, rate=fr)

    

    

    
    # LOGIC: Note that for the last output, elements beyond the reference window don't affect the subsequent output. 
    # as such we simply need to consider the reference window only. Reference window is the sum of the dilations for last point. 
    #i.e always consider left most point contributing towards it! Always + dilation for reference window. 
    #But what is x too small ie starts from 0? Not sure about - 1 also but still works This is especially because of the 
        # 1by1 conv. Therefore since weight predecided we only really need to know of the last node, if 2 by 1 would need to know of the other one also i.e one before.  
    
    # def generate_slower(self, x, models, dilation_depth, n_repeat, ctx, n):
    #     dilations = [2**i for i in range(dilation_depth)] * n_repeat 
    #     res = list(x.asnumpy())
    #     for _ in trange(n):
    #         x = nd.array(res[-sum(dilations)-1:],ctx=ctx) i.e losing (k - 1)*dilation every time. So we sum all --> here only looking at 1 skip conn output
    #         y = models(x)
    #         res.append(y.argmax(1).asnumpy()[-1])
    #     return res
         

        







        
        

