class DefaultModelConfig():
    def __init__(self):
        self.MODEL_NAME = 'OneVOSDefault'

        self.MODEL_VOS = 'onevos_convmae'
        self.MODEL_ENGINE = 'onevos_engine_convmae'
        print("self.MODEL_ENGIN: "+self.MODEL_ENGINE)
        self.expand_ratio_small=1.3


        self.MODEL_ALIGN_CORNERS = False
        self.MODEL_ENCODER = 'convmae'
        self.MODEL_ENCODER_PRETRAIN = "./pretrain_weights/convmae_base.pth"

        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]  # 4x, 8x, 16x, 16x
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 10
        self.MODEL_SELF_HEADS = 8
        self.MODEL_ATT_HEADS = 8
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_USE_PREV_PROB = False

        self.TRAIN_LONG_TERM_MEM_GAP = 9999

        self.TEST_LONG_TERM_MEM_GAP = 9999
