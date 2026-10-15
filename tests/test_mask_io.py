import base64
import io
import os

import numpy as np
import pytest
from PIL import Image

from apps.data_flywheel.perception.protocol import MaskPersistenceError
from apps.data_flywheel.perception.mask_io import load_mask_png, save_mask_png


def _fake_png_base64(mode: str = 'L') -> str:
    image = Image.new(mode, (4, 3), color=255)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_save_and_load_mask_png(tmp_path):
    path = tmp_path / 'mask.png'

    save_mask_png(_fake_png_base64(), str(path))

    assert path.exists()
    array = load_mask_png(str(path))
    assert array.dtype == np.uint8
    assert array.shape == (3, 4)


def test_save_mask_creates_parent_dir(tmp_path):
    nested = tmp_path / 'deep' / 'nested' / 'dir' / 'mask.png'

    save_mask_png(_fake_png_base64(mode='RGBA'), str(nested))

    assert nested.exists()


def test_save_mask_failure_raises_persistence_error(monkeypatch, tmp_path):
    path = tmp_path / 'mask.png'
    mask_b64 = _fake_png_base64()

    def fail_save(self, fp, format=None, **params):
        raise OSError('disk full')

    monkeypatch.setattr(Image.Image, 'save', fail_save)

    with pytest.raises(MaskPersistenceError) as exc_info:
        save_mask_png(mask_b64, str(path))

    assert exc_info.value.path == str(path)
    assert isinstance(exc_info.value.cause, OSError)
