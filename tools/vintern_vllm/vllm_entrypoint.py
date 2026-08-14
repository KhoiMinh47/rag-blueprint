"""Start vLLM with the Prometheus/Starlette compatibility shim for this image.

The shared Retriever GPU image contains a prometheus-fastapi-instrumentator
version that assumes every Starlette route has ``path``. vLLM mounts an
``_IncludedRouter`` for some endpoints, so the metrics middleware otherwise
turns every HTTP request into a 500. Keep the workaround local to Vintern's
vLLM process; the Retriever service and NIM containers are untouched.
"""

from __future__ import annotations

from typing import Any

from prometheus_fastapi_instrumentator import routing


_original_get_route_name = routing.get_route_name


def _safe_get_route_name(request: Any) -> str | None:
    try:
        return _original_get_route_name(request)
    except AttributeError as exc:
        if "path" not in str(exc):
            raise
        return None


routing.get_route_name = _safe_get_route_name


def _register_ministral3_text_config_alias() -> None:
    """Bridge Transformers 4.57's missing ``ministral3`` text alias.

    Mistral 3 HF checkpoints identify the nested language config as
    ``ministral3`` while Transformers 4.57.6 only registers the outer
    ``mistral3`` config.  vLLM's Mistral 3 implementation only needs the
    standard Mistral text-config fields, so register a narrow alias before
    vLLM asks AutoConfig to parse the checkpoint.  Newer Transformers already
    register the alias; ``exist_ok=True`` keeps this shim harmless there.
    """

    from transformers import AutoConfig, MistralConfig

    class Ministral3TextConfig(MistralConfig):
        model_type = "ministral3"

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("architectures", ["MistralForCausalLM"])
            super().__init__(**kwargs)

    AutoConfig.register(
        "ministral3",
        Ministral3TextConfig,
        exist_ok=True,
    )


def _patch_mistral_common_special_token_ids() -> None:
    """Make Transformers 4.57 resolve Mistral multimodal control tokens.

    ``MistralCommonTokenizer`` exposes control tokens such as ``[IMG]``
    through its special-token table, but its Tekken-backed ``_piece_to_id``
    path sends them through the ordinary text encoder.  That splits the
    control token into three byte tokens and makes ``PixtralProcessor`` fail
    during vLLM's multimodal budget initialization.  Resolve known control
    tokens directly from their ranks while retaining the upstream path for
    ordinary text.
    """

    try:
        from transformers.tokenization_mistral_common import MistralCommonTokenizer
    except ImportError:
        return

    original_piece_to_id = MistralCommonTokenizer._piece_to_id
    if getattr(original_piece_to_id, "_nvfp4_special_token_patch", False):
        return

    def _piece_to_id(self: Any, piece: str) -> int:
        tokenizer = getattr(getattr(self, "tokenizer", None), "instruct_tokenizer", None)
        tokenizer = getattr(tokenizer, "tokenizer", None)
        for special_token in getattr(tokenizer, "_all_special_tokens", ()):
            if special_token.get("token_str") == piece:
                return int(special_token["rank"])
        return original_piece_to_id(self, piece)

    _piece_to_id._nvfp4_special_token_patch = True
    MistralCommonTokenizer._piece_to_id = _piece_to_id


def _allow_mistral_common_processor_tokenizer() -> None:
    """Allow PixtralProcessor to accept Transformers' Mistral backend.

    In Transformers 4.57 the Mistral-common backend is intentionally not a
    ``PreTrainedTokenizerBase`` subclass, while ``PixtralProcessor`` still
    enforces that older base-class check. vLLM already unwraps its Mistral
    tokenizer to this backend before constructing the processor, so accepting
    this one known pair is safe and keeps the multimodal path usable.
    """

    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_mistral_common import MistralCommonTokenizer

    original_check = ProcessorMixin.check_argument_for_proper_class
    if getattr(original_check, "_nvfp4_mistral_patch", False):
        return

    def _check_argument_for_proper_class(self: Any, argument_name: str, argument: Any):
        if argument_name == "tokenizer" and isinstance(argument, MistralCommonTokenizer):
            return type(argument)
        return original_check(self, argument_name, argument)

    _check_argument_for_proper_class._nvfp4_mistral_patch = True
    ProcessorMixin.check_argument_for_proper_class = _check_argument_for_proper_class


def _ignore_vllm_processor_tokenizer_override() -> None:
    """Drop vLLM's processor-only ``tokenizer`` kwarg for Mistral-common.

    vLLM passes its already-selected tokenizer into ``ProcessorMixin``. In
    Transformers 4.57, that kwarg is forwarded to
    ``MistralCommonTokenizer.from_pretrained`` even though the method rejects
    it. Let the processor load the same local ``tekken.json`` tokenizer; the
    class/type and special-token shims above handle the remaining mismatch.
    """

    from transformers.tokenization_mistral_common import MistralCommonTokenizer

    original_from_pretrained = MistralCommonTokenizer.from_pretrained
    if getattr(original_from_pretrained, "_nvfp4_processor_patch", False):
        return

    original_function = original_from_pretrained.__func__

    def _from_pretrained(cls: Any, *args: Any, **kwargs: Any):
        kwargs.pop("tokenizer", None)
        return original_function(cls, *args, **kwargs)

    _from_pretrained._nvfp4_processor_patch = True
    MistralCommonTokenizer.from_pretrained = classmethod(_from_pretrained)


def _add_mistral_processor_init_kwargs() -> None:
    """Provide tokenizer compatibility required by ``PixtralProcessor``."""

    import re

    from transformers.tokenization_mistral_common import MistralCommonTokenizer

    if not hasattr(MistralCommonTokenizer, "init_kwargs"):
        MistralCommonTokenizer.init_kwargs = {}

    original_text_to_ids = MistralCommonTokenizer._text_to_ids
    if getattr(original_text_to_ids, "_nvfp4_multimodal_patch", False):
        return

    def _text_to_ids(self: Any, text: Any, add_special_tokens: bool) -> list[int]:
        if not isinstance(text, str):
            return original_text_to_ids(self, text, add_special_tokens)

        tokenizer = getattr(getattr(self, "tokenizer", None), "instruct_tokenizer", None)
        tokenizer = getattr(tokenizer, "tokenizer", None)
        special_tokens = {
            item["token_str"]: int(item["rank"])
            for item in getattr(tokenizer, "_all_special_tokens", ())
        }
        matched_tokens = [token for token in special_tokens if token in text]
        if not matched_tokens:
            return original_text_to_ids(self, text, add_special_tokens)

        pattern = "|".join(re.escape(token) for token in sorted(matched_tokens, key=len, reverse=True))
        pieces = re.split(f"({pattern})", text)
        token_ids: list[int] = []
        for piece in pieces:
            if not piece:
                continue
            if piece in special_tokens:
                token_ids.append(special_tokens[piece])
            else:
                token_ids.extend(tokenizer.encode(piece, bos=False, eos=False))

        if add_special_tokens:
            token_ids.insert(0, self.bos_token_id)
            token_ids.append(self.eos_token_id)
        return token_ids

    _text_to_ids._nvfp4_multimodal_patch = True
    MistralCommonTokenizer._text_to_ids = _text_to_ids


if __name__ == "__main__":
    _register_ministral3_text_config_alias()
    _patch_mistral_common_special_token_ids()
    _allow_mistral_common_processor_tokenizer()
    _ignore_vllm_processor_tokenizer_override()
    _add_mistral_processor_init_kwargs()
    from vllm.entrypoints.cli.main import main

    main()
