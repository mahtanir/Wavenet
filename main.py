from train import * 
import dotsi

class Config():
    def __init__(self) -> None:
        self.layer_size = 10
        self.stack_size = 2 
        self.res_channels = 24 
        self.skip_channels =  128 
        self.mu = 256
        self.batch_size = 64
        self.epochs = 10 
        self.seq_length = 20000
        self.generation = True
        

def main():
    config = Config()
    trainer = Train(config) #this is an object
    print('trainer object', trainer, '\n')
    wavenet = trainer.train()
    if (config.generation):
        trainer.generator(wavenet, 2000, config.layer_size, config.stack_size)

    
if (__name__ == '__main__'):
    y = input('Re-train (y/n)?')
    if y == 'y':
        main()
    else:
        config = Config()
        trainer = Train(config)
        model = trainer.bestModel()
        trainer.generator(model, 2000, config.layer_size, config.stack_size)

# print(__name__)


    