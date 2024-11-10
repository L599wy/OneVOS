from .default_onevos import DefaultEngineConfig
import os


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='onevos_convmae'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PRE'

        self.init_dir()

        self.DATASETS = ['static']
        self.DATA_SEQ_LEN = 4

        self.DATA_DYNAMIC_MERGE_PROB = 1.0
        self.TRAIN_LR = 2e-4
        self.TRAIN_LR_MIN = 2e-5
        self.TRAIN_TOTAL_STEPS = 200000
        self.TRAIN_WEIGHT_DECAY = 0.03
        self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 0.1
        self.DATA_WORKERS = 5

        self.ACCUMULATION_STEPS = 2
        self.TRAIN_MEM_EVERY = 1
        # self.PRETRAIN = False
        
        # self.DIR_RESULT ="/mnt/liwy/onevos/test/PRE_CONVMAE/"
        self.DIR_ROOT= "./"
        self.DIR_RESULT =os.path.join(self.DIR_ROOT, 'checkpoints_saves', self.EXP_NAME,
                                self.STAGE_NAME)
        # self.DIR_RESULT = self.DIR_RESULT
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, 'ema_ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, 'log', 'tensorboard')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')

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

