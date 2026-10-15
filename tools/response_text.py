"""Separate a verified SDK response scaffold from model-generated text.

This is transport decoding, not answer extraction. Never remove Markdown fences,
model reasoning, repeated scaffolds, whitespace, or malformed JSON from the body.
"""
from __future__ import annotations

VERSION = 'swift-empty-think-scaffold-v1'
SWIFT_NON_THINKING_PREFIX = '<think>\n\n</think>\n\n'
POLICY = {'version': VERSION, 'sdk_response_prefix': SWIFT_NON_THINKING_PREFIX,
          'operation': 'remove exactly one verified SDK prefix; leave generated body unchanged'}
RAW_POLICY = {'version': 'identity-generated-text-v1', 'sdk_response_prefix': '',
              'operation': 'preserve the entire SDK response; template adds no response scaffold'}


def generated_text(response, sdk_response_prefix):
    if sdk_response_prefix == '':
        if not isinstance(response, str): raise ValueError('SDK response must be text')
        return response
    if sdk_response_prefix != SWIFT_NON_THINKING_PREFIX:
        raise ValueError('Unsupported SDK response scaffold; verify the actual template first')
    if not isinstance(response, str) or not response.startswith(sdk_response_prefix):
        raise ValueError('SDK response does not contain its declared fixed scaffold')
    return response[len(sdk_response_prefix):]
