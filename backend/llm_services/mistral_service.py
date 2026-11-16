"""Groq chat and Mistral embedding service helpers."""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from pydantic import SecretStr

try:
    from backend.config import (
        GROQ_API_KEY,
        GROQ_CHAT_MODEL,
        MISTRAL_API_KEY,
        MISTRAL_EMBEDDING_MODEL,
        RAG_MAX_OUTPUT_TOKENS,
        GUARDRAILS_ENABLED,
        GUARDRAILS_CONFIG_DIR,
    )
except ModuleNotFoundError:
    from config import (
        GROQ_API_KEY,
        GROQ_CHAT_MODEL,
        MISTRAL_API_KEY,
        MISTRAL_EMBEDDING_MODEL,
        RAG_MAX_OUTPUT_TOKENS,
        GUARDRAILS_ENABLED,
        GUARDRAILS_CONFIG_DIR,
    )


FALLBACK_RESPONSE = "I don't know based on the uploaded PDF."


def _require_mistral_key() -> SecretStr:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not configured")
    return SecretStr(MISTRAL_API_KEY)


def _require_groq_key() -> SecretStr:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")
    return SecretStr(GROQ_API_KEY)


@lru_cache(maxsize=1)
def get_embeddings_model() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(
        api_key=_require_mistral_key(),
        model=MISTRAL_EMBEDDING_MODEL,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    return ChatGroq(
        groq_api_key=_require_groq_key(),
        model=GROQ_CHAT_MODEL,
        temperature=0.0,
        max_tokens=RAG_MAX_OUTPUT_TOKENS,
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_embeddings_model().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embeddings_model().embed_query(text)


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        return "".join(text_parts)

    return ""


def generate_grounded_answer(question: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return FALLBACK_RESPONSE

    context_lines: list[str] = []
    for item in context_chunks:
        context_lines.append(
            f"[Chunk {item.get('chunk_index', 0) + 1} | Score {item.get('score', 0.0):.3f}] {item.get('text', '')}"
        )

    system_prompt = (
        "You are a policy assistant. Answer strictly from the provided PDF context. "
        "Never use outside knowledge. Never infer missing facts. "
        "Format your response using markdown with:\n"
        "- Use **bold** for key terms and headers\n"
        "- Use bullet points (- or *) for lists\n"
        "- Use proper sections and structure\n"
        "- Use line breaks between sections for readability\n"
        "If the answer is not explicitly present in the context, reply exactly: "
        f'"{FALLBACK_RESPONSE}"'
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{chr(10).join(context_lines)}\n\n"
        "Instruction: Return only a concise, grounded answer from context."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        response = get_chat_model().invoke(messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(exc)}")

    answer = _extract_response_text(getattr(response, "content", "")).strip()
    if not answer:
        return FALLBACK_RESPONSE
    return answer


# ═══════════════════════════════════════════════════════════════
# NeMo Guardrails Integration
# ═══════════════════════════════════════════════════════════════
#
# Adds programmatic guardrails around the LLM call:
#
#   INPUT RAILS (before generation):
#     1. Topic control    — blocks off-topic questions
#     2. Prompt injection  — blocks jailbreak attempts
#
#   OUTPUT RAILS (after generation):
#     3. Groundedness      — verifies answer is in retrieved context
#     4. Content safety    — blocks harmful output
#
# If NeMo Guardrails is disabled (GUARDRAILS_ENABLED=false in .env)
# or if it fails to initialize, the system gracefully falls back
# to the existing direct LLM call (generate_grounded_answer above).
# This ensures the RAG pipeline never breaks due to guardrail issues.
# ═══════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

_rails_instance = None
_rails_init_attempted = False


def _init_guardrails():
    """Initialize NeMo Guardrails LLMRails instance. Cached singleton."""
    global _rails_instance, _rails_init_attempted

    if _rails_init_attempted:
        return _rails_instance

    _rails_init_attempted = True

    if not GUARDRAILS_ENABLED:
        logger.info("NeMo Guardrails disabled via GUARDRAILS_ENABLED=false")
        return None

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — guardrails cannot initialize")
        return None

    try:
        from nemoguardrails import LLMRails, RailsConfig

        config_path = GUARDRAILS_CONFIG_DIR
        if not os.path.isdir(config_path):
            logger.warning(f"Guardrails config dir not found: {config_path}")
            return None

        rails_config = RailsConfig.from_path(config_path)
        _rails_instance = LLMRails(rails_config)
        logger.info(f"NeMo Guardrails initialized from {config_path}")
        return _rails_instance

    except Exception as exc:
        logger.warning(f"NeMo Guardrails initialization failed: {exc}")
        return None


def generate_grounded_answer_guarded(
    question: str,
    context_chunks: list[dict],
) -> str:
    """
    Generate a grounded answer with NeMo Guardrails protection.

    Input rails block off-topic questions and prompt injection attempts
    before the LLM is called. Output rails verify the answer is
    grounded in the retrieved context and free of harmful content.

    Falls back to the unguarded generate_grounded_answer() if:
      - GUARDRAILS_ENABLED is false
      - NeMo Guardrails fails to initialize
      - The guarded call raises an exception
    """
    if not context_chunks:
        return FALLBACK_RESPONSE

    rails = _init_guardrails()

    # ── Fallback: guardrails not available → use existing direct call ──
    if rails is None:
        return generate_grounded_answer(question, context_chunks)

    # ── Build context string for the guarded call ──
    context_lines: list[str] = []
    for item in context_chunks:
        context_lines.append(
            f"[Chunk {item.get('chunk_index', 0) + 1} | Score {item.get('score', 0.0):.3f}] {item.get('text', '')}"
        )
    context_str = "\n".join(context_lines)

    system_prompt = (
        "You are a policy assistant. Answer strictly from the provided PDF context. "
        "Never use outside knowledge. Never infer missing facts. "
        "Format your response using markdown with:\n"
        "- Use **bold** for key terms and headers\n"
        "- Use bullet points (- or *) for lists\n"
        "- Use proper sections and structure\n"
        "- Use line breaks between sections for readability\n"
        "If the answer is not explicitly present in the context, reply exactly: "
        f'"{FALLBACK_RESPONSE}"'
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Instruction: Return only a concise, grounded answer from context."
    )

    try:
        # NeMo Guardrails intercepts input → LLM → output automatically
        response = rails.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Extract text from NeMo response
        if isinstance(response, dict):
            answer = response.get("content", "")
        elif isinstance(response, list):
            # NeMo may return a list of message dicts
            answer = ""
            for msg in response:
                if isinstance(msg, dict):
                    answer += msg.get("content", "")
                elif isinstance(msg, str):
                    answer += msg
        elif isinstance(response, str):
            answer = response
        else:
            answer = _extract_response_text(getattr(response, "content", ""))

        answer = answer.strip()
        if not answer:
            return FALLBACK_RESPONSE
        return answer

    except Exception as exc:
        logger.warning(
            f"NeMo Guardrails generation failed, falling back to direct call: {exc}"
        )
        return generate_grounded_answer(question, context_chunks)
