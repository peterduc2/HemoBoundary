"""Fast deterministic checks for the additional HemoBoundary workflows."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src import nested_classical as nc
from src import reproduction_utils as ru
from src import transfer_diagnostics as td


def check_snirf_metadata():
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / 'case.snirf'
        with h5py.File(path, 'w') as f:
            nirs = f.create_group('nirs').create_group('data1')
            for number, (source, detector, label) in enumerate(
                ((1, 1, 0), (2, 2, 1), (3, 3, 0), (3, 3, 1)), 1
            ):
                g = nirs.create_group(f'measurementList{number}')
                g.create_dataset('sourceIndex', data=np.array([source]))
                g.create_dataset('detectorIndex', data=np.array([[detector]]))
                g.create_dataset('dataTypeLabel', data=np.array(label))
        assert ru.unaugmented_pairs(path) == {'S1_D1'}


def check_known_response_bank():
    class Stub:
        FULL_T = np.arange(-5, 20.5, .5)
        POST_MASK = FULL_T >= 0
        @staticmethod
        def original_half_cosine_shape(rng):
            onset = float(rng.uniform(.5, 4.5))
            offset = onset + 8.
            peak = onset + 2.
            shape = np.sin(np.linspace(0., np.pi, 41))
            return shape, onset, peak, offset, {}
        @staticmethod
        def trial_to_predictors(h, r, present, onset, peak, offset):
            return {'present': int(present), 'onset': onset, 'peak': peak, 'offset': offset,
                    'censored': int(present and offset > 19.75),
                    'obs': np.zeros((41, 1)), 'state_feat': np.zeros((41, 9))}
    m = Stub()
    rng = np.random.default_rng(25)
    windows = [(rng.normal(size=(5, 51)), rng.normal(size=(5, 51))) for _ in range(4)]
    first, metadata = ru.make_trials(m, windows, participant_index=0)
    second, _ = ru.make_trials(m, windows, participant_index=0)
    assert len(metadata) == 16
    assert len(first['present']) == 16 and int(first['present'].sum()) == 12
    assert [x['trial_index'] for x in metadata] == list(range(16))
    assert np.allclose(first['onset'], second['onset'], equal_nan=True)
    assert first['state_feat'].shape == (16, 41, 9)
    assert nc.holm([.01, .03, .04]).tolist() == [0.03, 0.06, 0.06] or np.allclose(
        nc.holm([.01, .03, .04]), [.03, .06, .06]
    )
    h, r = windows[0]
    sh, sr = td.make_ar_surrogate(h, r, m.FULL_T, np.random.default_rng(3))
    sh2, sr2 = td.make_ar_surrogate(h, r, m.FULL_T, np.random.default_rng(3))
    assert sh.shape == h.shape and sr.shape == r.shape
    assert np.array_equal(sh, sh2) and np.array_equal(sr, sr2)
    assert np.allclose(sh[:, m.FULL_T < 0].std(axis=1), h[:, m.FULL_T < 0].std(axis=1))


def main():
    assert len(ru.get_manifest()) == 14
    assert ru.self_test()
    check_snirf_metadata()
    check_known_response_bank()
    print('PASS: fixed cohort manifest, SNIRF metadata, injections, source-only tests, and AR control')


if __name__ == '__main__':
    main()
