# Exact Prompt Template

This file is synchronized with Appendix A, Section 6.

## Prompt Template

```text
You are an expert in skeleton-based human action analysis. Given an action class and a body part, describe only the observable motion evidence of this body part during the action. Do not mention objects, scene context, appearance, or intention. Use one concise sentence with a consistent style. If the body part has no salient motion, describe it as weak or stable motion. Output only the description sentence.

Action class: {action class}
Body part: {body part}
Description:
```

The placeholders `{action class}` and `{body part}` are replaced by the action class name and one of the six predefined body parts.

## Generation Settings

- model: GPT-4o-mini, used in an offline preprocessing stage
- temperature: 0
- top-p: 1.0
- maximum output length: 80 tokens
- post-processing: deterministic text normalization only; no manual semantic rewriting

## Body Parts

- head
- torso
- left arm
- right arm
- left leg
- right leg
