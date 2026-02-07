"""Customer Support Chatbot with PEFT/LoRA fine-tuning and advanced prompt engineering.

This package provides a production-grade customer support chatbot that leverages:
- Advanced prompt engineering with template versioning and few-shot examples
- PEFT/LoRA fine-tuning pipelines for domain adaptation
- Intent classification and routing
- Conversation state management with Redis-backed sessions
- Streaming responses via SSE
"""

from __future__ import annotations

__version__ = "0.1.0"
