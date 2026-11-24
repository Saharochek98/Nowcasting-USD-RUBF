"""Check committed source identity and rerun RW drift in a temporary output root."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'results/rw_drift_experiment'
    manifest = json.loads((output/'manifest.json').read_text())
    for name, digest in manifest['protected_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest['implementation_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
        committed = subprocess.check_output(['git','show',manifest['git_sha']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n', b'\n')).hexdigest() == digest, name
    matched = {}
    with tempfile.TemporaryDirectory(prefix='usdrubf-rw-drift-') as temporary:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_rw_drift_experiment.py'),
                        '--output-root',temporary],cwd=ROOT,check=True,capture_output=True,text=True)
        for original in sorted(output.glob('*.csv')):
            assert original.read_bytes() == (Path(temporary)/original.name).read_bytes(), original.name
            matched[original.name] = hashlib.sha256(original.read_bytes()).hexdigest()
    record = dict(committed_implementation_verified=True,protected_files_unchanged=True,
                  all_numeric_csv_bytes_identical=True,csv_sha256=matched,
                  source_commit=manifest['git_sha'],
                  manifests_not_compared='Run-specific Git working-tree status and output paths are not numeric results')
    (output/'reproducibility.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
