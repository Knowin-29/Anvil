"""CPU regression checks for preserving a real committed resume prefix."""
import json
from pathlib import Path
import tempfile
from resume_state import restore_exposures, restore_logs


def main():
    with tempfile.TemporaryDirectory(prefix='anvil-resume-state-') as tmp:
        root = Path(tmp)
        committed = [dict(step=s, micro=m, rank=3) for s in (1, 2) for m in (0, 1)]
        original = ''.join(json.dumps(r)+'\n' for r in committed + [dict(step=3, micro=0, rank=3)])
        original += '{"step":3'
        path = root/'exposure_rank3.jsonl'
        path.write_text(original)
        backup = restore_exposures(path, 2, 3, 2)
        assert Path(backup).read_text() == original
        assert [json.loads(x) for x in path.read_text().splitlines()] == committed
        assert restore_exposures(path, 2, 3, 2) is None
        for bad in [committed[:-1], committed + [committed[-1]], list(reversed(committed)),
                    [{**r, 'rank':4} for r in committed]]:
            content = ''.join(json.dumps(r)+'\n' for r in bad)
            path.write_text(content)
            try:
                restore_exposures(path, 2, 3, 2)
            except ValueError:
                assert path.read_text() == content
            else:
                raise AssertionError('Invalid committed prefix was accepted')
        path.write_text('{"broken":\n' + json.dumps(committed[0])+'\n')
        try:
            restore_exposures(path, 2, 3, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed committed prefix was accepted')
        log = root/'logging.jsonl'
        log.write_text('{"step":1,"loss":1.0}\n{"step":2,"loss":0.5}\n{"step":3,"loss":0.7}\n')
        history = [dict(step=1,loss=1.0),dict(step=2,loss=.5)]
        old = log.read_bytes()
        backup = restore_logs(log, history, 2)
        assert Path(backup).read_bytes() == old
        assert [json.loads(x) for x in log.read_text().splitlines()] == history
        assert restore_logs(log, history, 2) is None
    print('Resume prefix preservation, malformed/duplicate/rank rejection and idempotence passed')


if __name__ == '__main__':
    main()
