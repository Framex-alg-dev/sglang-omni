"""Render CLS file collection requests or audit bounded service log samples.

No credentials, network calls, service restarts, or cloud mutations occur here.
The deployment uses LogListener rather than adding uploads to inference paths.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import re
from uuid import UUID

SERVICES = ('omni-event-worker', 'omni-reply-worker', 'omni-action-worker')
REQUIRED = {'timestamp', 'level', 'service', 'env', 'version', 'host', 'logger', 'event'}
ID = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')
STAMP = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$')
FORBIDDEN = {'session_id', 'authorization', 'cookie', 'secret_id', 'secret_key',
             'secretid', 'secretkey', 'token', 'messages', 'prompt', 'negative_prompt',
             'user_text', 'assistant_text', 'audio', 'video', 'image', 'payload'}
SENSITIVE = re.compile(r'AKID[A-Za-z0-9]{16,}|Bearer\s+(?!\[REDACTED\])[^\s,;]+', re.I)


def collection_requests(targets: dict, *, services=SERVICES) -> list[dict]:
    """Produce API request bodies only after all destinations are explicit."""
    if set(targets) != {'region', 'host_log_root', 'topics'}:
        raise ValueError('targets must contain exactly region, host_log_root, topics')
    region = targets['region']
    if not isinstance(region, str) or not re.fullmatch(r'[a-z]{2}-[a-z]+(?:-[0-9]+)?', region):
        raise ValueError('invalid CLS region')
    root = targets['host_log_root']
    if not isinstance(root, str) or not root.startswith('/') or any(c in root for c in ',*?\n') or '..' in Path(root).parts:
        raise ValueError('host_log_root must be one absolute directory')
    topics = targets['topics']
    if not isinstance(topics, dict) or set(topics) != set(SERVICES):
        raise ValueError('one TopicId per Omni service is required')
    topic_ids = [str(UUID(topics[service])) for service in SERVICES]
    if len(set(topic_ids)) != len(SERVICES):
        raise ValueError('Omni roles must have separate topics')
    return [dict(Action='CreateConfig', Version='2020-10-16', Region=region,
                 Name=f'framex-{service}', Output=topic_id,
                 InputType='file', Path=str(Path(root) / service / 'app.log'),
                 LogType='json_log', ExtractRule=dict(
                     TimeKey='timestamp', TimeFormat='%Y-%m-%dT%H:%M:%S.%f%z',
                     IsGBK=0, JsonStandard=1, UnMatchUpLoadSwitch=True,
                     UnMatchLogKey='LogParseFailure', Backtracking=0))
            for service, topic_id in zip(SERVICES, topic_ids) if service in services]


def _forbidden(value) -> bool:
    if isinstance(value, dict):
        return any(str(k).lower() in FORBIDDEN or _forbidden(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_forbidden(v) for v in value)
    return isinstance(value, str) and bool(SENSITIVE.search(value))


def audit(root: Path, *, limit: int, root_session: str | None, services=SERVICES) -> dict:
    """Return counts and fixed failure codes; never print raw log content."""
    if limit < 1:
        raise ValueError('sample limit must be positive')
    result = {}
    for service in services:
        path = root / service / 'app.log'
        errors = Counter()
        records = []
        try:
            with path.open(encoding='utf-8') as source:
                for line in deque(source, maxlen=limit):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError()
                    except ValueError:
                        errors['invalid_json'] += 1
                        continue
                    if root_session is None or row.get('root_session_id') == root_session:
                        records.append(row)
        except (OSError, UnicodeError):
            errors['file_unreadable'] += 1
        for row in records:
            if REQUIRED - row.keys(): errors['missing_required_fields'] += 1
            if row.get('service') != service: errors['wrong_service'] += 1
            if row.get('env') != 'shared': errors['wrong_environment'] += 1
            try:
                if not STAMP.fullmatch(row.get('timestamp', '')): raise ValueError()
                datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            except (ValueError, TypeError): errors['invalid_timestamp'] += 1
            for key in ('request_id', 'root_session_id', 'turn_id'):
                if key in row and (not isinstance(row[key], str) or not ID.fullmatch(row[key])):
                    errors['invalid_correlation_id'] += 1
            if _forbidden(row): errors['forbidden_content'] += 1
        if not records: errors['no_matching_records'] += 1
        result[service] = {'sampled_records': len(records), 'violations': dict(errors)}
    return {'spec_version': '1.0', 'scope': 'local_files_only',
            'successful': all(not v['violations'] for v in result.values()), 'services': result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--targets', type=Path, help='public region, host_log_root, topics mapping; no secrets')
    mode.add_argument('--audit-root', type=Path, help='host directory bound to /var/log/framex')
    parser.add_argument('--root-session')
    parser.add_argument('--service', choices=SERVICES, action='append', help='only roles deployed on this host; defaults to all three')
    parser.add_argument('--limit', type=int, default=2000)
    args = parser.parse_args()
    try:
        if args.targets:
            output = {'collection_requests': collection_requests(json.loads(args.targets.read_text()), services=args.service or SERVICES)}
        else:
            output = audit(args.audit_root, limit=args.limit, root_session=args.root_session, services=args.service or SERVICES)
    except (ValueError, TypeError, AttributeError, OSError):
        parser.exit(2, 'Invalid or incomplete CLS deployment input; no changes applied.\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output.get('successful', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
