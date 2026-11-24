"""Read-only consistency checks plus a deterministic real-data prequential rerun."""
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from benchmarks import ridge, ridge_v2
from scripts.run_ridge_experiments import write
from scripts.run_ridge_experiments_v2 import protected_hashes


def normalized(frame):
    buffer = io.StringIO()
    write(frame, buffer)
    return pd.read_csv(io.StringIO(buffer.getvalue()), float_precision='round_trip')


def main():
    output = ROOT/'results/ridge_experiments_v2'
    cfg = ridge_v2.config()
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['protected_sha256'] == protected_hashes(cfg)
    for name, digest in manifest['implementation_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
    scores = pd.read_csv(output/'scored_common_predictions.csv')
    assert len(scores) == 35560
    assert scores.groupby(['model_id', 'horizon', 'scope', 'transform']).size().eq(254).all()
    assert (scores.train_end_date <= scores.origin_date).all()
    assert scores.target_date.between('2025-01-01','2025-12-31').all()
    assert (scores[scores.scope.eq('frozen')].train_end_date <= '2024-12-31').all()
    assert scores[scores.scope.eq('prequential')].train_end_date.gt('2024-12-31').any()
    assert scores[~scores.model_id.eq('price_only')].pit_status.eq('ASSUMED_AVAILABLE').all()
    comparisons = pd.read_csv(output/'comparisons_hac_holm.csv')
    assert len(comparisons) == 532 and comparisons.primary.sum() == 98
    assert comparisons.n.eq(254).all()
    frame, _, columns = ridge_v2.load_inputs(cfg)
    pred, coef, _, origins, fits = ridge_v2.walk_forward(frame, columns['core_v2'], 'core_v2', 7, 'delta', cfg, 'prequential')
    dest = output/'prequential_delta'
    saved = pd.read_csv(dest/'predictions_oos.csv', float_precision='round_trip')
    saved = saved[saved.model_id.eq('core_v2') & saved.horizon.eq(7)].reset_index(drop=True)
    pd.testing.assert_frame_equal(normalized(pred), saved, check_exact=True)
    saved = pd.read_csv(dest/'coefficients_windows.csv', float_precision='round_trip')
    saved = saved[saved.model_id.eq('core_v2') & saved.horizon.eq(7)].reset_index(drop=True)
    pd.testing.assert_frame_equal(normalized(coef), saved, check_exact=True)
    importance = pd.DataFrame([dict(model_id='core_v2', horizon=7, transform='delta', regime='prequential',
                                   pit_status='ASSUMED_AVAILABLE', **r)
                              for r in ridge.permutation(frame, columns['core_v2'], pred, origins, fits, cfg)])
    saved = pd.read_csv(dest/'permutation_importance.csv', float_precision='round_trip')
    saved = saved[saved.model_id.eq('core_v2') & saved.horizon.eq(7)].reset_index(drop=True)
    pd.testing.assert_frame_equal(normalized(importance), saved, check_exact=True)
    report = dict(prediction_count=len(scores), common_pairs_per_model_horizon=254,
                  matured_labels_verified=True, frozen_train_cap_verified=True,
                  prequential_2025_labels_present=True, validation_targets_only=True,
                  protected_v1_data_baselines_unchanged=True, source_hashes_verified=True,
                  real_data_reproduction={'model':'core_v2','horizon':7,'regime':'prequential',
                      'transform':'delta','predictions_exact':True,'coefficients_exact':True,'importance_exact':True},
                  hac_comparison_count=532, primary_holm_family_size=98, secondary_holm_family_size=434,
                  separate_test_results='See reports/ridge_experiments_v2.md; this verifier checks artifacts, not the unittest suite')
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
