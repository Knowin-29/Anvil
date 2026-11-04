"""Versioned model I/O for a geometry-constrained DoF candidate.

This is a disclosed model representation change, not a new scoring rule.
Only training targets enter encode_target. Inference receives the original user
prompt and generated latent JSON; no reference answers are used by the decoder.
"""
from copy import deepcopy
import json
from dof_codec_prototype import encode, decode

VERSION = "anvil-dof-rotation6d-contact-v1"


def context(prompt):
    return json.loads(prompt.removeprefix("<image>").strip())["input"]


def model_prompt(prompt):
    request = dict(contract=VERSION, input=context(prompt),
        task="Predict the camera-frame gripper pose and contact for the specified entity.",
        output="Return only a JSON object with key latent and exactly 12 finite numbers.",
        latent_order=["rotation_column0_x", "rotation_column0_y", "rotation_column0_z",
                      "rotation_column1_x", "rotation_column1_y", "rotation_column1_z",
                      "gripper_translation_x", "gripper_translation_y", "gripper_translation_z",
                      "logit_contact_u_over_image_width", "logit_contact_v_over_image_height",
                      "log_contact_camera_depth"],
        decoder="Orthonormalize the two orientation columns with Gram-Schmidt; cross product gives approach. Sigmoid the two contact coordinates and exp the depth; use the supplied intrinsics to back-project.")
    return "<image>" + json.dumps(request, sort_keys=True, separators=(",", ":"))


def encode_target(target, prompt):
    return json.dumps(dict(latent=[round(float(x), 6) for x in encode(target, context(prompt))]),
                      separators=(",", ":"), allow_nan=False)


def decode_response(response, prompt):
    def reject_constant(value):
        raise ValueError("Nonfinite generated JSON: " + value)
    value = json.loads(response, parse_constant=reject_constant)
    if not isinstance(value, dict) or set(value) != {"latent"}:
        raise ValueError("Expected exactly the latent field")
    latent = value["latent"]
    if not isinstance(latent, list) or len(latent) != 12 or any(type(x) not in (int, float) for x in latent):
        raise ValueError("Expected 12 generated numbers")
    return json.dumps(decode(latent, context(prompt)), separators=(",", ":"), allow_nan=False)


def training_row(row):
    assert row["gsa_meta"]["split"] == "train" and row["gsa_form"] == "dof"
    result = deepcopy(row)
    prompt = row["messages"][0]["content"]
    result["messages"][0]["content"] = model_prompt(prompt)
    result["messages"][1]["content"] = encode_target(json.loads(row["messages"][1]["content"]), prompt)
    return result
