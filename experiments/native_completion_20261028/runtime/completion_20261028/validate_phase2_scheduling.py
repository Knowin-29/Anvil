"""Exercise capacity and scheduled-public ordering without any cloud mutation."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
C=Path(__file__).resolve().parent;O=C/'objective_controls_phase2_20261029'
sys.path.insert(0,str(O))
from objective_control_common import bound,save


def main():
    spec=importlib.util.spec_from_file_location('phase2_submit_test',O/'submit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    formal=dict(id='formal',gpus=16,status='Running',scope='formal')
    pure=dict(id='pure',gpus=8,status='Running',scope='objective_phase2_no_native_sft')
    public=dict(id='public512',gpus=8,status='Success',scope='formal_MOPD_public_step512')
    assert module.ready('no_native_sft',[formal])[0]
    assert not module.ready('no_native_sft',[formal,pure])[0]
    with tempfile.TemporaryDirectory() as temp:
        module.FORMAL=Path(temp)
        assert not module.ready('low_native_sft_no_mtd',[formal,pure])[0]
        folder=module.FORMAL/'evaluation/512'
        folder.mkdir(parents=True)
        (module.FORMAL/'training').mkdir()
        save(folder/'submission.json',dict(response=dict(Id='public512')))
        save(folder/'report.json',dict(state='complete_requires_quality_review'))
        save(module.FORMAL/'training/status.json',dict(step=600))
        assert module.ready('low_native_sft_no_mtd',[formal,pure,public])[0]
        assert not module.ready('low_native_sft_no_mtd',[formal,pure,{**public,'status':'UNKNOWN'}])[0]
        save(module.FORMAL/'training/status.json',dict(step=800))
        assert not module.ready('low_native_sft_no_mtd',[formal,pure,public])[0]
        assert module.ready('low_native_sft_no_mtd',[formal,{**pure,'status':'Success'},public])[0]
    receipt=dict(state='passed',first_control_reserves_public8=True,second_requires512_report_and_cloud_success=True,
        unknown_status_does_not_release_public=True,near1024_public_reservation_restored=True,
        simultaneous_cap=32,controller=bound(O/'submit.py'),validation_code=bound(Path(__file__)),
        cloud_mutated=False,completed=time.time())
    save(O/'scheduling_validation.json',receipt);print(receipt)


if __name__=='__main__':main()
