INPUT_SCHEMA = {
    "language": {
        "type": str,
        "required": True,
        "constraints": lambda language: language in ["en", "ru"]
    },
    "voice": {
        "type": str,
        "required": True
    },
    "text": {
        "type": str,
        "required": True
    },
    "gpt_cond_len": {
        "type": int,
        "required": True
    },
    "max_ref_len": {
        "type": int,
        "required": True
    },
    "speed": {
        "type": float,
        "required": False,
    }
}
