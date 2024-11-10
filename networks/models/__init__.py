
# from networks.models.aot_convmae_andmem_decoup_eval import OneVOS_CONVMAE_eval
from networks.models.aot_convmae_andmem_decoup import OneVOS_CONVMAE


def build_vos_model(name, cfg, **kwargs):
    if name == 'onevos_convmae':
        return OneVOS_CONVMAE(cfg, **kwargs)
    if name == 'onevos_convmae_eval':
        return OneVOS_CONVMAE(cfg, **kwargs)
    else:
        raise NotImplementedError