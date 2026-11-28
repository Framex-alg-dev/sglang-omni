"""FrameX v1.0 CLS deployment: destinations, timezone, and content guardrails."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_omni_cls', ROOT / 'scripts/prepare_omni_cls.py')
cls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cls)


class ClsDeploymentTest(unittest.TestCase):
    def targets(self):
        return dict(region='ap-guangzhou', host_log_root='/host/framex',
                    topics={service: str(uuid4()) for service in cls.SERVICES})

    def test_destinations_are_separate_and_keep_utc_milliseconds_and_parse_failures(self):
        target = self.targets()
        requests = cls.collection_requests(target)
        self.assertEqual(len({r['Output'] for r in requests}), 3)
        reply_only = cls.collection_requests(target, services=['omni-reply-worker'])
        self.assertEqual(len(reply_only), 1)
        self.assertEqual(reply_only[0]['Output'], target['topics']['omni-reply-worker'])
        self.assertEqual({r['Path'] for r in requests}, {f'/host/framex/{s}/app.log' for s in cls.SERVICES})
        for r in requests:
            self.assertEqual(r['LogType'], 'json_log')
            self.assertTrue(r['ExtractRule']['UnMatchUpLoadSwitch'])
            self.assertEqual(r['ExtractRule']['TimeKey'], 'timestamp')
            self.assertEqual(r['ExtractRule']['TimeFormat'], '%Y-%m-%dT%H:%M:%S.%f%z')

    def test_incomplete_shared_and_secret_destinations_are_rejected(self):
        for case in ['missing', 'shared', 'secret', 'wildcard']:
            t = self.targets()
            if case == 'missing': t['topics'].pop(cls.SERVICES[0])
            if case == 'shared': t['topics'] = {s: next(iter(t['topics'].values())) for s in cls.SERVICES}
            if case == 'secret': t['secret_key'] = 'must-not-be-an-input'
            if case == 'wildcard': t['host_log_root'] = '/logs/*'
            with self.subTest(case=case), self.assertRaises(ValueError): cls.collection_requests(t)

    def test_audit_allows_text_but_rejects_credentials_and_wrong_role(self):
        with tempfile.TemporaryDirectory() as d:
            for service in cls.SERVICES:
                p = Path(d) / service / 'app.log'
                p.parent.mkdir()
                row = dict(timestamp='2026-11-25T18:00:00.123Z', level='INFO', service=service,
                           env='shared', version='revision', host='host', logger='module',
                           event='inference.completed', message='完成', request_id='req_owner',
                           root_session_id='root', turn_id='turn')
                if service == cls.SERVICES[0]:
                    row['model_input'] = {'system_prompt': 'private user text is allowed'}
                if service == cls.SERVICES[1]: row['service'] = cls.SERVICES[0]
                if service == cls.SERVICES[2]:
                    row['model_input'] = {'system_prompt': 'AKID' + 'A' * 20}
                p.write_text(json.dumps(row, ensure_ascii=False) + '\n')
            report = cls.audit(Path(d), limit=100, root_session='root')
            self.assertFalse(report['successful'])
            self.assertNotIn('forbidden_content', report['services'][cls.SERVICES[0]]['violations'])
            self.assertEqual(report['services'][cls.SERVICES[1]]['violations']['wrong_service'], 1)
            self.assertEqual(report['services'][cls.SERVICES[2]]['violations']['forbidden_content'], 1)
            self.assertNotIn('private user text', json.dumps(report))
            self.assertNotIn('prompt', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
