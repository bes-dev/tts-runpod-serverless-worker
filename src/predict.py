import os
# torch
import torch
# xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_enhancer import AudioEnhancer

use_cuda = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'

class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def setup(self):
        # Load XTTSv2 model
        self.config = XttsConfig()
        self.config.load_json(
            os.path.join(self.model_dir, "xttsv2", "config.json")
        )
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=os.path.join(self.model_dir, "xttsv2"),
            use_deepspeed=True,
            eval=True
        )
        if use_cuda:
            self.model.cuda()
        # Load Audio Enhancer model
        self.audio_enhancer = AudioEnhancer.from_pretrained(
            os.path.join(self.model_dir, "audio_enhancer", "enhancer_stage2"),
            "cuda" if use_cuda else "cpu"
        )

    @torch.inference_mode()
    def predict(
            self,
            text: str,
            speaker_wav: str,
            gpt_cond_len: int,
            max_ref_len: int,
            language: str,
            speed: float,
            enhance_audio: bool
    ):
        outputs = self.model.synthesize(
            text,
            self.config,
            speaker_wav=speaker_wav,
            gpt_cond_len=gpt_cond_len,
            language=language,
            enable_text_splitting=True,
            max_ref_len=max_ref_len,
            speed=speed
        )
        wave, sr = outputs['wav'], 24000
        if enhance_audio:
            wave, sr = self.audio_enhancer(
                torch.from_numpy(wave),
                sr
            )
            wave = wave.detach().cpu().numpy()
        return wave, sr
        # return outputs['wav']
