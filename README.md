# Book Examples from Deep Learning with Python by François Chollet
Working through [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by [François Chollet](https://github.com/fchollet).

François Chollet can be found at:

* [@fchollet on twitter](https://twitter.com/fchollet)
* [@fchollet on github](https://github.com/fchollet)
* [google research](https://research.google.com/pubs/105096.html)

# Running on EC2
```
cd ops
terraform plan
terraform apply
./post-terraform.sh
```

If all goes well, to work with the server, do the following:

* ssh aws-gpu (with 6006 and 8888 forwarded)
* tmux
* create a window for jupyter server
    * source activate dlb
    * jupyter lab
* create a window for tensorboard
    * source activate dlb
    * tensorboard --logdir /tmp/tensorboard


### Questions/Contact
[Tim Hoolihan](https://github.com/thoolihan)

[@thoolihan](https://twitter.com/thoolihan)
