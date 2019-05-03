# rbm-mnist
An implementation of a simple generative RBM and discriminative RBM, tested against the MNIST dataset. 

# How to run

**Copying the directory:**

`git clone https://github.com/abelanger5/rbm-mnist`

**Installing Libraries:**

Although we implemented the majority of the model from scratch, we used sklearn functions for data preprocessing, some numerical calculations, and to generate confusion matrices. We used matplotlib for generating all plots, and python-mnist for importing the dataset. 

`pip install matplotlib`
`pip install sklearn`
`pip install python-mnist`

**Generative RBM**
 `python grbm.py`

 Hyperparameters can be tuned in the **grbm.py** file when calling the model. Lines at the bottom of the file can be uncommented to generate animations and different types of plots. 

 **Discriminative RBM**
 `python drbm.py`

 Hyperparameters can be tuned in the **drbm.py** file. 