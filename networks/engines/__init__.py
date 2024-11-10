
from networks.engines.aot_engine_convmae import OneVOS_InferEngine,OneVOS_Engine

def build_engine(name, phase='train', **kwargs):
    if name=='onevos_engine_convmae':
        if phase == 'train':
            return OneVOS_Engine(**kwargs)
        elif phase == 'eval':
            return OneVOS_InferEngine(**kwargs)
        else:
            print("enter")
            raise NotImplementedError

    else:
        raise NotImplementedError