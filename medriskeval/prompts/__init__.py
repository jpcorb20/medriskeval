"""Centralized prompt formatting for medriskeval.

This module provides prompt builders for:
- Target models (generating responses to evaluate)
- LLM-as-judge (evaluating model outputs)

Prompt Builders:
- PromptBuilder: Base class for all prompt builders
- JudgePromptBuilder: Base class for judge prompts
- TargetPromptBuilder: Base class for target model prompts

Judge Implementations:
- PSBMSBJudgePromptBuilder: 5-level scale for PSB/MSB evaluation
- RefusalJudgePromptBuilder: Binary refusal classifier for XSTest/JBB
- FACTSJudgePromptBuilder: Sentence-level groundedness for FACTS-med

Target Model Implementations:
- PSBTargetPromptBuilder: PatientSafetyBench inputs
- MSBTargetPromptBuilder: MedSafetyBench inputs
- XSTestTargetPromptBuilder: XSTest inputs
- JBBTargetPromptBuilder: JailbreakBench inputs

Utilities:
- Chat format normalization (normalize_messages, to_openai_format)
- Output parsing (parse_psb_msb_judge_output, parse_refusal_judge_output, etc.)
"""

from medriskeval.prompts.base import (
    PromptBuilder,
    JudgePromptBuilder,
    TargetPromptBuilder,
    PromptConfig,
)

from medriskeval.prompts.chat_format import (
    normalize_message,
    normalize_messages,
    to_openai_format,
    from_openai_format,
    prepend_system_message,
    append_assistant_message,
    format_conversation,
    merge_system_messages,
    MessageInput,
)

from medriskeval.prompts.psb_msb import (
    PSBMSBJudgePromptBuilder,
    PSBTargetPromptBuilder,
    MSBTargetPromptBuilder,
    PSB_USAGE_POLICY,
    MSB_USAGE_POLICY,
    PSB_MSB_RUBRIC,
    parse_psb_msb_judge_output,
)

from medriskeval.prompts.refusal_judge import (
    RefusalJudgePromptBuilder,
    RefusalJudgeConfig,
    XSTestTargetPromptBuilder,
    JBBTargetPromptBuilder,
    REFUSAL_JUDGE_GEN_PARAMS,
    parse_refusal_judge_output,
    aggregate_refusal_votes,
)

from medriskeval.prompts.facts_judge import (
    FACTSJudgePromptBuilder,
    GroundednessLabel,
    SentenceJudgment,
    parse_facts_judge_output,
)


__all__ = [
    # Base classes
    "PromptBuilder",
    "JudgePromptBuilder",
    "TargetPromptBuilder",
    "PromptConfig",
    # Chat format utilities
    "normalize_message",
    "normalize_messages",
    "to_openai_format",
    "from_openai_format",
    "prepend_system_message",
    "append_assistant_message",
    "format_conversation",
    "merge_system_messages",
    "MessageInput",
    # PSB/MSB
    "PSBMSBJudgePromptBuilder",
    "PSBTargetPromptBuilder",
    "MSBTargetPromptBuilder",
    "PSB_USAGE_POLICY",
    "MSB_USAGE_POLICY",
    "PSB_MSB_RUBRIC",
    "parse_psb_msb_judge_output",
    # Refusal (XSTest/JBB)
    "RefusalJudgePromptBuilder",
    "RefusalJudgeConfig",
    "XSTestTargetPromptBuilder",
    "JBBTargetPromptBuilder",
    "REFUSAL_JUDGE_GEN_PARAMS",
    "parse_refusal_judge_output",
    "aggregate_refusal_votes",
    # FACTS-med groundedness
    "FACTSJudgePromptBuilder",
    "GroundednessLabel",
    "SentenceJudgment",
    "parse_facts_judge_output",
]
