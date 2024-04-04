import os
# torch
import torch
# xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

use_cuda = os.environ.get('TTS_WORKER_USE_CUDA', 'True').lower() == 'true'

class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def setup(self):
        self.config = XttsConfig()
        self.config.load_json(
            os.path.join(self.model_dir, "config.json")
        )
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=self.model_dir,
            use_deepspeed=True,
            eval=True
        )
        if use_cuda:
            self.model.cuda()

    @torch.inference_mode()
    def predict(
            self,
            text: str,
            speaker_wav: str,
            gpt_cond_len: int,
            max_ref_len: int,
            language: str
    ):
        outputs = self.model.synthesize(
            text,
            self.config,
            speaker_wav=speaker_wav,
            gpt_cond_len=gpt_cond_len,
            language=language,
            enable_text_splitting=True
        )
        return outputs['wav']
