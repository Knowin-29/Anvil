"""Train-only pose codebook interface, explicitly separate from native scoring."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from dof_codec_interface import context
from dof_codec_prototype import encode, decode

VERSION = 'anvil-dof-train-rotation128-v1'
PATH = Path(__file__).resolve().parent / 'dof_rotation_codebook.json'
BOOK = json.loads(PATH.read_text())
assert BOOK['heldout_used_to_fit'] is False and BOOK['codebook_size'] == 128
ROTATIONS = np.array(BOOK['rotations'])


def model_prompt(prompt):
    c = deepcopy(context(prompt))
    # Random source instance IDs are lookup keys, not pose observations. Keep
    # label, task, image, explicit box and calibration; decoder restores ID.
    del c['entity']['entity_id']
    c['entity']['bbox_coordinate_units'] = 'thousandths of image width and height, [0,1000]'
    request = dict(contract=VERSION, input=c,
        task='Predict the calibrated camera-frame gripper pose and contact for the specified image entity.',
        output='Return exactly a JSON object with rotation_bin and position. rotation_bin is an integer from 0 to 127 in the fixed training pose codebook; position is six finite numbers.',
        position_order=['gripper_translation_x','gripper_translation_y','gripper_translation_z',
                        'logit_contact_u_over_image_width','logit_contact_v_over_image_height','log_contact_camera_depth'])
    return '<image>' + json.dumps(request, sort_keys=True, separators=(',',':'))




def decode_response(response, prompt):
    def reject(value):
        raise ValueError('Nonfinite JSON ' + value)
    value = json.loads(response, parse_constant=reject)
    if not isinstance(value,dict) or set(value) != {'rotation_bin','position'}:
        raise ValueError('Expected rotation_bin and position')
    code, position = value['rotation_bin'], value['position']
    if type(code) is not int or not 0 <= code < 128:
        raise ValueError('Rotation code outside the frozen codebook')
    if not isinstance(position,list) or len(position) != 6 or any(type(v) not in (int,float) for v in position):
        raise ValueError('Expected six position numbers')
    rotation = ROTATIONS[code]
    latent = [*rotation[:,0], *rotation[:,1], *position]
    return json.dumps(decode(latent, context(prompt)), separators=(',',':'), allow_nan=False)


