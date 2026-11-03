"""Same sealed sampler with exact-length reads instead of large VEPFS read-ahead."""
import json
import os
from pool import Pool as OriginalPool, SCHEDULES, FORMS, permute

class Pool(OriginalPool):
    def native_row(self, stage, form, ordinal, source):
        offsets = self.offsets[form]
        selected = permute(ordinal + self.offset(stage, source), len(offsets), self.seed)
        offset, length = map(int, offsets[selected])
        payload = os.pread(self.handles[form].fileno(), length, offset)
        assert len(payload) == length, 'Incomplete indexed native row read'
        row = json.loads(payload)
        assert row['gsa_form'] == form
        return row
