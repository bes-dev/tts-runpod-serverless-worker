import os
import torch
import numpy as np
from numpy.typing import NDArray
from resemble_enhance.enhancer.enhancer import Enhancer
from resemble_enhance.enhancer.hparams import HParams
from resemble_enhance.inference import inference


class AudioEnhancer:
    def __init__(self, model: Enhancer, device: str = "cpu"):
        self.model = model
        self.device = device

    def to(self, device: str):
        self.model.to(device)
        self.device = device

    @staticmethod
    def setup(install_dir: str = "data/models/resemble-enhance"):
        """ Setup Audio Enhancer model """
        relpaths = [
            "hparams.yaml",
            os.path.join("ds", "G", "latest"),
            os.path.join("ds", "G", "default", "mp_rank_00_model_states.pt")
        ]
        for relpath in relpaths:
            path = os.path.join(install_dir, relpath)
            if os.path.exists(path):
                continue
            url = f"https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/{relpath}?download=true"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.hub.download_url_to_file(url, path)
        return os.path.join(install_dir, "")

    @classmethod
    def from_pretrained(
            cls,
            install_dir: str | os.PathLike | None,
            device: str | torch.device = "cpu"
    ):
        params = HParams.load(install_dir)
        model = Enhancer(params)
        state_dict = torch.load(
            os.path.join(install_dir, "ds", "G", "default", "mp_rank_00_model_states.pt"),
            map_location="cpu"
        )["module"]
        model.load_state_dict(state_dict)
        model.eval()
        output = cls(model, device)
        output.to(device)
        return output

    @torch.inference_mode()
    def __call__(
            self,
            wave: torch.Tensor,
            sample_rate: int,
            nfe: int = 64,
            solver: str = "midpoint", # "midpoint", "rk4", "euler"
            lambd: float = 1.0,
            tau: float = 0.5
    ):
        assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
        assert solver in ("midpoint", "rk4", "euler"), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
        assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
        assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"
        self.model.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        return inference(model=self.model, dwav=wave, sr=sample_rate, device=self.device)


if __name__ == "__main__":
    import torchaudio
    # enhancer = AudioEnhancer.from_pretrained(AudioEnhancer.setup())
    enhancer = AudioEnhancer.from_pretrained("/home/sergei/work/generative_media/ckpt/resemble-enhance/enhancer_stage2")
    # dwav, sr = torchaudio.load("/home/sergei/Desktop/female_ru_0001.wav")
    # dwav = dwav.mean(0)
    # hwav, sr = enhancer(dwav, sr)
    # torchaudio.save("out.wav", hwav[None], sr)
