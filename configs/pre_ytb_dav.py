import os
from .default_onevos import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='onevos_convmae'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PRE_YTB_DAV'

        self.init_dir()

        self.DATASETS = ['youtubevos', 'davis2017']
        self.DATA_SEQ_LEN = 5

        '''self.TRAIN_LR = 5e-5
        self.TRAIN_LR_MIN = 5e-6'''
        self.TRAIN_LR = 1e-4
        self.TRAIN_LR_MIN = 1e-5
        self.TRAIN_TOTAL_STEPS = 200000

        # for gradient accumulation
        self.ACCUMULATION_STEPS = 2
        self.TRAIN_MEM_EVERY=1
        self.training =True

        self.DIR_ROOT= "./"
        self.DIR_RESULT =os.path.join(self.DIR_ROOT, 'checkpoints_saves', self.EXP_NAME,
                                       self.STAGE_NAME)

        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, 'ema_ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, 'log', 'tensorboard')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')
        #
        for path in [
            self.DIR_RESULT, self.DIR_CKPT, self.DIR_EMA_CKPT,
            self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG,
            self.DIR_TB_LOG
        ]:
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except Exception as inst:
                    print(inst)
                    print('Failed to make dir: {}.'.format(path))

        pretrain_stage = 'PRE'
        pretrain_ckpt = 'save_step_200000.pth'
        self.PRETRAIN = True
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'checkpoints_saves',
                                           self.EXP_NAME, pretrain_stage,
                                           'ema_ckpt', pretrain_ckpt)
