from train import * 

def main():
    config = {
        'layer_size': 10,
        'stack_size': 2, 
        'res_channels': 24, 
        'skip_channels': 128, 
        'mu': 256,
        'batch_size': 64,
        'epochs': 10, 
        'seq_length': 20000,
        'generation': True
    }

    trainer = Train(config) #this is an object
    wavenet = trainer.train()
    if (config.generation):
        trainer.generator(wavenet, 2000, config.layer_size, config.stack_size)

    
if (__name__ == 'main'):
    main()
    


    