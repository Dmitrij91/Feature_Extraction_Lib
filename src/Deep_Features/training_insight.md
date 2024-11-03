flat_dense
- relu seems superior to sigmoid
- small networks (size 6) underfit

resnet18
7x7x7: 
- patches are overfitted
- even with 0.8 dropout
5x5x5:
- training loss 0.6 but testing loss is seemingly more like 1.5 and fluctuating (dropout = 0.5)
- a little dropout seems to help, a different run without dropout did not overfit, but loss is 2.5

resnet 2D
- all models show massively improved training results if a little dropout is at play
- stepping down model sizes, we get the expected result: larger windows give better fitting
- the model size sweetspot seems to be around resnet8

fully convolutional
- inference is of course much quicker
- the network decisions between different patches are inconsistent. It looks like the network can decide where it sees layers, but not which ones. This leads to inconsistencies between neighboring patches which can not be smoothed out by the AF.