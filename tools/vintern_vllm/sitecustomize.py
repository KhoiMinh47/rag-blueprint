"""Apply the local vLLM/Transformers compatibility shims in child processes."""

from vllm_entrypoint import (
    _add_mistral_processor_init_kwargs,
    _allow_mistral_common_processor_tokenizer,
    _ignore_vllm_processor_tokenizer_override,
    _patch_mistral_common_special_token_ids,
    _register_ministral3_text_config_alias,
)


_register_ministral3_text_config_alias()
_patch_mistral_common_special_token_ids()
_allow_mistral_common_processor_tokenizer()
_ignore_vllm_processor_tokenizer_override()
_add_mistral_processor_init_kwargs()
