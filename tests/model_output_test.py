import pytest
import torch

"""
benefits over pdb 

no repetition of ellements

reusable elemtns - can test modularize specific pieces of fucntionality for attention


# tdd - write objectives to the test ---> this ensures that every step that I am doing is CORRECT - which is the thing that leads to the most struggle -- is this certainty unneceesary especially in debuggging?P o
# clearly in this case testing on every single fucntion is bad - i just want to do attention based and rapid development testing when actually necessary, often in the beginning you might not need certainty on every thign
# just basically on key aspects that you feel the most unconfident in.

"""


def test_output_CNN_model_generator(cnn_model,X_tensor_cuda):
    """
    Test for testing whether the model outputs the right datatype 
    """


    # TODO: is there a way that in controlled enviroments liek these I can do a pdb like thing just like i do in the other environment? 
#    pytest.set_trace() # useful method instead of pdb as here we can not only document key interactions but also perform saerching in the diffrent test. while 

    X_tensor = X_tensor_cuda
    y = cnn_model(X_tensor)


    assert y.shape == torch.Size([1])  # What is the meaning of this? 


    # abstraction is equivalent ot (1,) in numpy 
def test_output_CNN_model(cnn_model, X_tensor_cuda):
    """
    Test for testing whether the model outputs the right datatype 
    Test with errors.
    """


    # TODO: is there a way that in controlled enviroments liek these I can do a pdb like thing just like i do in the other environment? 
#    pytest.set_trace() # useful method instead of pdb as here we can not only document key interactions but also perform saerching in the diffrent test. while 

    y = cnn_model(X_tensor_cuda)


    assert y.shape == torch.Size([1])# (1,1) torch size useful for maintaining the abstration of objects, using numel method, acts like a tuple. i can pass shapes into other tensors
    # only useful for numel

def test_output_CNN_model_name(cnn_model): 
    #

    print("log output") # debugging
    model = cnn_model

    assert model.__class__.__name__ == 'CNNScorer'