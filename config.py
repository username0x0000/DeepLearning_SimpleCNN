
DATA = {
    'type': 'MNIST',
    'resolution': (32, 32),
    'test_vali_split': 0.2,
    'batch_size': 32,
    'shuffle': True,
    'iter': 50,
    'epochs': 30
    }

MODEL = {
    'ResNet50':{
        'backbone': {
            'input_channels':1,
            'first_layer':{'input_channels':64, 'output_channels':64, 'stride':1, 'layer_num':3},
            'second_layer':{'input_channels':64, 'output_channels':128, 'stride':2, 'layer_num':4},
            'third_layer':{'input_channels':128, 'output_channels':256, 'stride':2, 'layer_num':6},
            'forth_layer':{'input_channels':256, 'output_channels':512, 'stride':2, 'layer_num':3},
        },
        'head':{
            'input_feature':512,
            'class_num':10
        }
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
