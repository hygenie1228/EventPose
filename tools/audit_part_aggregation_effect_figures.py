"""Verify exported paired-effect figures against their exact archived inputs."""
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation'
STEMS = ['completed_paired_effects', 'v6_part_control_effects',
         'completed_paired_effects_with_long', 'v6_long_part_control_effects']


def main():
    rows = []
    for stem in STEMS:
        folder = OUT/'figures'; manifest = json.loads((folder/(stem+'.json')).read_text())
        assert set(manifest['source_sha256']) == set(manifest['source_snapshots'])
        inputs = {}
        for original, snapshot in manifest['source_snapshots'].items():
            source = (ROOT/snapshot).read_bytes()
            assert hashlib.sha256(source).hexdigest() == manifest['source_sha256'][original]
            inputs[original] = json.loads(source)
        for row in manifest['actual_recorded_full_Test_statistics']:
            family = manifest.get('control_family')
            if family:
                suffix = {'Fixed':'fixed', 'Global dynamic':'global_dynamic', 'Part static':'part_static'}[row['version']]
                data = next(iter(inputs.values()))
                expected = data['tests'][family+'_'+suffix+'_'+row['metric']]
                assert row['holm_p_value'] == expected['holm_p_value_within_three_controls']
            else:
                key = str((OUT/(row['version']+'_comparison.json')).relative_to(ROOT))
                data = inputs[key]; assert data['all_complete'] and not data['pending']
                expected = data['tests'][row['version']+'_part_dynamic_'+row['metric']]
                assert row['holm_p_value'] == expected['holm_p_value']
            for key in ['effect', 'ci95', 'samples', 'p_value']:
                assert row[key] == expected[key], (stem, key)
        with Image.open(folder/(stem+'.png')) as image:
            size = image.size; image.verify()
        assert min(size) > 0
        svg = ET.parse(folder/(stem+'.svg')).getroot()
        assert svg.tag.endswith('svg') and 'viewBox' in svg.attrib
        pdf = subprocess.run(['pdfinfo', str(folder/(stem+'.pdf'))], capture_output=True, text=True, check=True)
        assert re.search(r'^Pages:\s+1\s*$', pdf.stdout, flags=re.MULTILINE)
        rows.append(dict(stem=stem,exact_archived_inputs_verified=len(inputs),statistical_rows_exact=len(manifest['actual_recorded_full_Test_statistics']),
                         PNG_decoded_size=size,SVG_XML_parsed=True,PDF_parsed_pages=1,
                         export_sha256={ext:hashlib.sha256((folder/(stem+'.'+ext)).read_bytes()).hexdigest() for ext in ['png','pdf','svg','json']}))
    result = dict(status='passed',checked_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),figures=rows,
                  scope='Exact archived JSON bytes and plotted effect/CI/sample/p-value records, PNG decoding, SVG XML parsing, PDF structural parsing; no model inference or statistical re-estimation',
                  limitation='Does not independently raster-compare PDF/SVG text layout; PNG figures were visually inspected separately')
    (OUT/'artifact_audits/paired_effect_figure_exports.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
