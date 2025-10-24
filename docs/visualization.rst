Visualization
=============

learnMSA can plot a sequence logo based on the model it learned.
It can also generate a gif that shows how the logo changes over the course of training.

Arguments
---------

``--logo`` *PATH*
    Produces a pdf of the learned sequence logo. Non-existing directories are created.

``--logo_gif`` *PATH*
    Produces a gif that animates the learned sequence logo over training time.
    Slows down training significantly and performs a training with simplified setting.
    The resulting MSA can be less accurate. Non-existing directories are created.
