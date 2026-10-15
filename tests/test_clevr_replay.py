"""Official-question identity regression; tiny fixtures are not dataset evidence."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from tools import map_clevr_replay as replay


def question(image=7, identifier=70, text="How many red things are there?", answer="2"):
    return {"image_index": image, "question_index": identifier,
            "image_filename": f"CLEVR_train_{image:06d}.png", "split": "train",
            "question": text, "answer": answer}


def source(questions):
    messages = []
    for i, q in enumerate(questions):
        messages += [{"role": "user", "content": ("<image>\n" if not i else "") + q["question"]
                     + "\nOffer a very short reply."},
                    {"role": "assistant", "content": q["answer"].capitalize() + "."}]
    return {"id": "clevr_000000", "images": ["/original/images/clevr_000000.jpg"], "messages": messages,
            "width": [8], "height": [6]}


def test_official_identity_not_local_row_number_and_order_independent():
    qs = [question(), question(identifier=71, text="What color is the cube?", answer="red")]
    index, images, count = replay.official_index({"questions": qs})
    assert (images, count) == (1, 2)
    row = source(list(reversed(qs)))
    match = replay.match_official(row, index)
    assert [q["question_index"] for q in match] == [70, 71]
    assert all(q["image_index"] == 7 for q in match)


@pytest.mark.parametrize("mutation", [
    lambda r: r["messages"][1].update(content="3."),
    lambda r: r["messages"][0].update(content="<image>\nHow many blue things are there?\nOffer a very short reply."),
    lambda r: r["messages"][0].update(content="<image>\nHow many red things are there?\nUnrecognized instruction."),
    lambda r: r["messages"][0].update(role="system"),
    lambda r: r["messages"].pop(),
    lambda r: r["messages"][1].update(content="TWO"),
])
def test_changed_or_unknown_source_wrapper_rejected(mutation):
    qs = [question()]
    index, _, _ = replay.official_index({"questions": qs})
    row = source(qs); mutation(row)
    with pytest.raises(ValueError):
        replay.match_official(row, index)


def test_partial_question_group_not_enough_to_assign_image():
    qs = [question(), question(identifier=71, text="What color is the cube?", answer="red")]
    index, _, _ = replay.official_index({"questions": qs})
    with pytest.raises(ValueError, match="unmatched"):
        replay.match_official(source(qs[:1]), index)


def test_identical_qa_groups_are_ambiguous_not_arbitrarily_selected():
    qs = [question(), question(image=8, identifier=80)]
    index, _, _ = replay.official_index({"questions": qs})
    with pytest.raises(ValueError, match="ambiguous"):
        replay.match_official(source(qs[:1]), index)


@pytest.mark.parametrize("field,value", [("split", "val"), ("image_index", True),
    ("image_filename", "CLEVR_train_000008.png"), ("answer", None), ("question_index", -1)])
def test_nontraining_or_inconsistent_official_annotation_rejected(field, value):
    q = question(); q[field] = value
    with pytest.raises(ValueError):
        replay.official_index({"questions": [q]})


def test_duplicate_official_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        replay.official_index({"questions": [question(), question()]})


def test_source_image_identity_and_root_are_checked(tmp_path):
    image = tmp_path / "clevr_000000.jpg"
    Image.new("RGB", (8, 6), "red").save(image)
    row = source([question()])
    assert replay.source_image(row, tmp_path) == image.resolve()
    row["images"] = ["/original/images/clevr_000001.jpg"]
    with pytest.raises(ValueError, match="mismatch"):
        replay.source_image(row, tmp_path)


def test_archive_pin_and_member_are_verified(tmp_path):
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(replay.QUESTION_MEMBER, json.dumps({"questions": [question()]}))
    index, evidence = replay.read_official(archive, replay.sha(archive))
    assert evidence["images"] == evidence["questions"] == 1
    assert len(evidence["member_sha256"]) == 64
    assert replay.match_official(source([question()]), index)[0]["image_index"] == 7
    with pytest.raises(ValueError, match="SHA256"):
        replay.read_official(archive, "0" * 64)


def test_missing_protected_image_requires_explicit_contained_remap(tmp_path):
    root = tmp_path / "images"; root.mkdir()
    image = root / "image.jpg"; image.write_bytes(b"fixture")
    assert replay.protected_image_path("/legacy/image.jpg", [(Path("/legacy"), root)]) == image
    with pytest.raises(ValueError, match="missing"):
        replay.protected_image_path("/legacy/image.jpg", [])
    outside = tmp_path / "outside.jpg"; outside.write_bytes(b"fixture")
    (root / "escape.jpg").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        replay.protected_image_path("/legacy/escape.jpg", [(Path("/legacy"), root)])


def test_exported_qa_is_independent_official_target_not_native_or_review_claim():
    q = question(); row = source([q]); before = deepcopy(row)
    fp = {"sha256": "1" * 64, "decoded_rgb_sha256": "2" * 64}
    official = {"member_sha256": "3" * 64, "archive_sha256": "4" * 64}
    result = replay.replay_record(row, q, fp, Path("/fixture.jpg"), 1, "5" * 64, "6" * 64, official)
    assert result["messages"] == [{"role": "user", "content": "<image>\n" + q["question"]},
                                  {"role": "assistant", "content": "2"}]
    assert row == before
    meta = result["anvil_meta"]
    assert meta["record_type"] == "qa_replay" and "native_form" not in meta
    assert meta["split"] == "train" and meta["independent_question"]
    for field in ["human_reviewed", "strong_model_verified", "public_release_eligible",
                  "official_original_image_bytes_verified"]:
        assert meta[field] is False
