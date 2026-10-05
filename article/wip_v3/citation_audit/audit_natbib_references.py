"""Read-only Crossref audit for this manuscript's manual natbib bibliography."""
import concurrent.futures
import datetime
import difflib
import html
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

SOURCE = Path(__file__).resolve().parents[1] / 'manuscript.tex'
REPORT = SOURCE.with_suffix('.reference-audit.json')
CACHE = {}


def plain(text):
    text = text.replace(r'\&', '&').replace('~', ' ').replace('--', '-')
    text = text.replace(r'{\o}', 'o').replace('CO$_2$', 'CO2')
    text = re.sub(r'\\[\"\'`^=.]\{?([A-Za-z])\}?', r'\1', text)
    text = re.sub(r'\\[uvHc]\{([A-Za-z])\}', r'\1', text)
    text = re.sub(r'\\(?:textit|textbf|url)\{([^{}]*)\}', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def norm(text):
    text = re.sub(r'<[^>]+>', '', html.unescape(plain(text))).replace('&', ' and ').replace('ø', 'o').replace('Ø', 'O')
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)).lower()
    return re.sub('[^a-z0-9]+', ' ', text).strip()


def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'ManuscriptReferenceAudit/1.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)['message']
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def assess(entry, work):
    title = (work.get('title') or [''])[0]
    if work.get('subtitle') and work['subtitle'][0]:
        title += ': ' + work['subtitle'][0]
    author = (work.get('author') or work.get('editor') or [{}])[0].get('family', '')
    years = sorted({date['date-parts'][0][0] for name in ['issued', 'published', 'published-print', 'published-online']
                    if (date := work.get(name)) and date.get('date-parts')})
    similarity = difflib.SequenceMatcher(None, norm(entry['title']), norm(title)).ratio()
    author_match = norm(entry['first_author']).removesuffix(' jr').removeprefix('de ') == norm(author).removesuffix(' jr').removeprefix('de ')
    container = (work.get('container-title') or [''])[0]
    venue_match = bool(container and norm(container) in norm(entry['publication']))
    conference_mismatch = work.get('type') == 'proceedings-article' and not venue_match
    return {'reliable': similarity >= 0.9 and author_match and not conference_mismatch,
            'venue_match': venue_match,
            'title_similarity': round(similarity, 4), 'author_match': author_match,
            'year_match': entry['year'] in years, 'years': years,
            'doi': work.get('DOI'), 'title': title,
            'container': (work.get('container-title') or [''])[0],
            'volume': work.get('volume'), 'issue': work.get('issue'),
            'page': work.get('page'), 'article_number': work.get('article-number'),
            'type': work.get('type'), 'authors': work.get('author', []),
            'metadata': {key: work[key] for key in [
                'DOI', 'title', 'subtitle', 'type', 'author', 'editor', 'container-title',
                'volume', 'issue', 'page', 'article-number', 'issued', 'published',
                'published-print', 'published-online', 'publisher', 'ISBN', 'URL', 'resource'
            ] if key in work}}


def verify(entry):
    try:
        candidates = []
        if entry['key'] in CACHE:
            candidates = [m['metadata'] for m in CACHE[entry['key']].get('candidates', [])]
        elif entry['existing_doi']:
            try:
                candidates.append(fetch('https://api.crossref.org/works/' + urllib.parse.quote(entry['existing_doi'], safe='')))
            except Exception as error:
                entry['doi_lookup_error'] = str(error)
        if entry['key'] not in CACHE and (not candidates or not assess(entry, candidates[0])['reliable']):
            query = entry['title'] + ' ' + entry['first_author'] + ' ' + str(entry['year'])
            url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.bibliographic': query, 'rows': 8})
            candidates.extend(fetch(url)['items'])
        matches = sorted([assess(entry, work) for work in candidates],
                         key=lambda m: (m['reliable'], m['venue_match'], m['year_match'], m['title_similarity'], m['type'] != 'posted-content'), reverse=True)
        entry['crossref'] = matches[0] if matches else None
        entry['candidates'] = matches
    except Exception as error:
        entry['error'] = str(error)
        entry['crossref'] = None
    return entry


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if '--reuse-cache' in sys.argv and REPORT.exists():
        CACHE.update({entry['key']: entry for entry in json.loads(REPORT.read_text(encoding='utf-8'))['references']})
    text = SOURCE.read_text(encoding='utf-8-sig')
    body, bibliography = text.split(r'\begin{thebibliography}', 1)
    cited = set()
    for keys in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}', body):
        cited.update(key.strip() for key in keys.split(','))
    entries = []
    pattern = r'\\bibitem\[(.*?)\]\{([^}]+)\}\s*(.*?)(?=\s*\\bibitem|\s*\\end\{thebibliography\})'
    for label, key, raw in re.findall(pattern, bibliography, re.S):
        cleaned = plain(raw)
        match = re.match(r'(.*?)\s+(\d{4})\.\s+(.*)', cleaned)
        authors, year, rest = match.groups()
        title, _, publication = rest.partition('. ')
        title = re.sub(r', (?:second|\d+(?:st|nd|rd|th)) ed$', '', title)
        doi = re.search(r'https://doi\.org/([^}\s]+)', raw)
        entries.append({'key': key, 'label': label, 'original': raw.strip(), 'authors': authors,
                        'first_author': authors.split(',')[0], 'year': int(year), 'title': title,
                        'publication': publication, 'existing_doi': doi.group(1) if doi else None,
                        'cited': key in cited})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, entries))
    keys = [e['key'] for e in entries]
    report = {'source': str(SOURCE), 'audited_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'format': 'manual natbib bibitem; existing Elsevier author-year formatting preserved',
              'mode': 'report-only', 'cited_keys': sorted(cited), 'reference_count': len(entries),
              'missing_body_citations': sorted(cited-set(keys)),
              'uncited_references': sorted(set(keys)-cited),
              'duplicate_keys': sorted({k for k in keys if keys.count(k)>1}), 'references': checked}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Report:', REPORT)
    print('References:', len(entries), 'Missing:', report['missing_body_citations'], 'Uncited:', report['uncited_references'])
    for entry in checked:
        match = entry['crossref']
        if match:
            print(entry['key'], 'VERIFIED' if match['reliable'] else 'REVIEW', match['title_similarity'], match['years'], match['doi'], match['title'])
        else:
            print(entry['key'], 'UNVERIFIED', entry.get('error', 'No match'))


if __name__ == '__main__':
    main()
