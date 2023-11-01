
DATA = {
    'type': 'MNIST',
    'resolution': (32, 32),
    'test_vali_split': 0.2,
    'batch_size': 32,
    'iter': 50,
    'epochs': 30
    }

MODEL = {
    'backbone': {
        'type': 'ResNet50',
        'first_layer':{'input_channels':0, 'output_channels':0, 'stride':0}
        # ...
        },
    'head': {
        'type': 'MlpClassifier',
        # ...
        },
    }

LOSS = {
    'type': 'CrossEntropy',
    # ...
    }

OPTIMIZER = {
    'type': 'Adam',
    # ...
    }

BATCH_SIZE = 4

TRAIN = {
    'epochs': 10
}
# ...
