"""Explicit model precision shared by base generation and SFT."""


def model_kwargs(config, *, training=False):
    import torch
    from transformers import BitsAndBytesConfig
    precision = config.get("precision", "bf16")
    if precision not in {"bf16", "fp16"}:
        raise ValueError("Precision must be bf16 or fp16")
    kwargs = {"dtype":torch.bfloat16 if precision == "bf16" else torch.float16}
    quantization = config.get("quantization")
    if quantization not in {None, "nf4"}:
        raise ValueError("Only the declared NF4 quantization profile is supported")
    if quantization == "nf4":
        kwargs.update(device_map={"":0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=kwargs["dtype"]))
    elif not training:
        kwargs["device_map"] = "auto"
    return kwargs
