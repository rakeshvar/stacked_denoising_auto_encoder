# Stacked Denoising Auto-Encoder 
In Theano. Using the mnist dataset.

## Running

```sh
python3 train.py
```

Is the basic run. You can specify more options like this. 

```sh
python3 train.py n_nodes=[180,42,10] noise=[.1,.05,.025] learning_rate=.1 lambda1=[.5,.1,.1] n_epochs=50 output_folder=plots
```

In the end you will get a pickle file with three pairs of (w, b) matrices. And a ton of cool pictures.

## Dependencies
* Python 3.0 (Yes, not the old dead version! The brand new one in stead!!!)
* Numpy, Scipy, Nose etc. etc.
* Theano
* Pillow

## References
* Bengio, Yoshua, Aaron Courville, and Pierre Vincent. "Representation learning: A review and new perspectives." Pattern Analysis and Machine Intelligence, IEEE Transactions on 35.8 (2013): 1798-1828.
