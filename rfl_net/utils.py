import tensorflow as tf

def activation_summary(x, name=None):
    """Helper to create summaries for activations."""
    if name == None:
        name = x.op.name
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))

def print_and_log(*args,**kwargs):
    print(*args, **kwargs)
    with open('output/log.txt', "a") as f:
        print(file=f, *args, **kwargs)