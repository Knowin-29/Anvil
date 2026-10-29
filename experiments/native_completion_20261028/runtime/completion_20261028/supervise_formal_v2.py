"""Resume existing formal orchestration with conservative cloud-observation retries.

The frozen trainer, formal plan, checkpoints and original controller code remain
unchanged. A missing observation never triggers a replacement training job.
"""
import json
from pathlib import Path
import sys
import threading
import time
C = Path(__file__).resolve().parent
sys.path.insert(0, str(C / 'formal_mopd'))
import controller
from formal_common import save


def main():
    plan_path = Path(sys.argv[sys.argv.index('--plan') + 1]).resolve()
    root = plan_path.parent
    original_check = controller.check
    error_lock = threading.Lock()

    def observed_check(job):
        for attempt in range(3):
            try:
                return original_check(job)
            except Exception as exc:
                with error_lock:
                    with (root / 'cloud_observation_errors.jsonl').open('a') as log:
                        log.write(json.dumps(dict(id=job['id'], attempt=attempt + 1,
                            error=repr(exc), time=time.time(), task_restarted=False)) + '\n')
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        # Keep all unresolved resources charged against the cap. This means
        # observation failure can delay admission, but cannot create capacity.
        return {**job, 'status': 'UNKNOWN', 'observation_unresolved': True}

    controller.check = observed_check
    try:
        controller.main()
    except BaseException as exc:
        save(root / 'controller_failure.json', dict(error=repr(exc), time=time.time(),
                                                   supervisor_version=2, task_restarted=False))
        raise


if __name__ == '__main__':
    main()
