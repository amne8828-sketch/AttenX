"""Inference package initialization"""

from .style_prompts import (
    get_system_prompt,
    ADMIN_TEMPLATE,
    FRIEND_TEMPLATE,
    JSON_OUTPUT_FORMAT,
    CAMERA_ACTIVITY_PROMPT,
    PRIVACY_CHECK_PROMPT,
    CONFIDENCE_GUIDELINES
)

__all__ = [
    'get_system_prompt',
    'ADMIN_TEMPLATE',
    'FRIEND_TEMPLATE',
    'JSON_OUTPUT_FORMAT',
    'CAMERA_ACTIVITY_PROMPT',
    'PRIVACY_CHECK_PROMPT',
    'CONFIDENCE_GUIDELINES'
]
