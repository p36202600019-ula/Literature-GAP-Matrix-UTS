"""RAG artikel ilmiah lokal: PDF -> ChromaDB -> Ollama -> 2 sheet Excel.
Jalankan: python rag_ollama.py
Dependensi: pip install chromadb pymupdf requests
"""
from __future__ import annotations
import json, re, sys, time, zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import requests
import chromadb
import pymupdf

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / 'papers'
DB_DIR = ROOT / 'chroma_ollama_db'  # database terpisah: tidak menimpa ChromaDB lama
OUT_DIR = ROOT / 'output'
MODEL = 'qwen3:4b'
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'
COLLECTION = 'literature_ollama_v1'
CHUNK_SIZE = 1250
CHUNK_OVERLAP = 200
FIELDS = ['Paper', 'DOI', 'Problem', 'Metode', 'Dataset', 'Temuan', 'Keterbatasan', 'Future Work']
GAP_FIELDS = ['Aspek', 'Penelitian Terdahulu', 'Keterbatasan', 'Research Gap', 'Peluang Penelitian', 'Relevansi Tesis', 'Sumber Paper']


def pdf_text(path):
    with pymupdf.open(path) as doc:
        return '\n'.join(page.get_text(sort=True) for page in doc)


def chunks(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step) if len(text[i:i + CHUNK_SIZE].strip()) > 80]


def doi_from_text(text):
    match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text[:15000], re.I)
    return match.group(0).rstrip('.,;)') if match else 'Tidak ditemukan'


def call_ollama(prompt, max_tokens=1800):
    try:
        r = requests.post(OLLAMA_URL, json={
            'model': MODEL, 'stream': False, 'format': 'json', 'think': False,
            'messages': [
                {'role': 'system', 'content': 'Anda asisten penelitian ilmiah. Jawab hanya JSON valid berbahasa Indonesia. Jangan mengarang data. Jika bukti tidak tersedia tulis "Tidak disebutkan dalam sumber". Kutip informasi hanya dari konteks.'},
                {'role': 'user', 'content': prompt}
            ],
            'options': {'temperature': 0, 'num_predict': max_tokens, 'num_ctx': 8192}
        }, timeout=600)
        r.raise_for_status()
        raw = r.json()['message']['content'].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find('{'), raw.rfind('}')
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
            raise ValueError('Respons Ollama bukan JSON: ' + raw[:300])
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError('Tidak bisa terhubung ke Ollama. Jalankan Ollama terlebih dahulu.') from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError('Ollama gagal. Pastikan model qwen3:4b tersedia melalui: ollama list. ' + str(e)) from e


def search_paper(col, name, query, n=5):
    result = col.query(query_texts=[query], n_results=n, where={'paper': name}, include=['documents'])
    return result['documents'][0] if result.get('documents') else []


def extract_record(col, name, doi):
    queries = [
        'research problem objective motivation proposed method approach algorithm',
        'dataset data source sample experiments evaluation results quantitative findings',
        'limitations weaknesses future work conclusion research directions'
    ]
    passages = []
    for q in queries:
        passages.extend(search_paper(col, name, q, n=4))
    passages = list(dict.fromkeys(passages))[:10]
    context = '\n\n'.join(f'[Kutipan {i+1}] {p}' for i, p in enumerate(passages))[:12500]
    prompt = f'''Analisis SATU artikel ilmiah dengan nama file {name} dan DOI yang terdeteksi {doi}.
Berdasarkan KUTIPAN PDF berikut, isi JSON tepat dengan kunci:
"Paper", "DOI", "Problem", "Metode", "Dataset", "Temuan", "Keterbatasan", "Future Work".
Paper: judul artikel jika jelas, jika tidak nama file. DOI: gunakan DOI terdeteksi, jangan mengarang.
Tulis ringkas, faktual, dalam bahasa Indonesia. Jika data tidak tersedia, isi "Tidak disebutkan dalam sumber".
KUTIPAN:\n{context}'''
    data = call_ollama(prompt)
    if not isinstance(data, dict):
        raise ValueError('Output ekstraksi bukan objek JSON')
    return {k: str(data.get(k) or 'Tidak disebutkan dalam sumber') for k in FIELDS}


def make_gaps(records):
    summary = '\n'.join(f"Paper {i+1}: {r['Paper']} | DOI {r['DOI']} | Metode {r['Metode']} | Temuan {r['Temuan']} | Keterbatasan {r['Keterbatasan']} | Future Work {r['Future Work']}" for i, r in enumerate(records))
    prompt = f'''Anda membuat Research Gap Matrix untuk tesis: "Prediksi kualitas produk makanan ringan Pilus berdasarkan parameter proses produksi real-time".
Gunakan HANYA ringkasan artikel di bawah. Jangan mengklaim bahwa suatu metode belum pernah diteliti hanya berdasarkan 1 atau beberapa artikel.
Identifikasi 3-5 gap atau peluang yang DIDUKUNG oleh artikel. Jika belum cukup bukti, sebutkan "Perlu verifikasi literatur tambahan".
Kembalikan JSON objek dengan kunci "gaps" yang berisi array objek. Setiap objek WAJIB memiliki kunci:
"Aspek", "Penelitian Terdahulu", "Keterbatasan", "Research Gap", "Peluang Penelitian", "Relevansi Tesis", "Sumber Paper".
Sumber Paper harus menyebut judul atau DOI yang mendukung, bukan sumber fiktif.
RINGKASAN:\n{summary[:15000]}'''
    data = call_ollama(prompt, max_tokens=2400)
    items = data.get('gaps', []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        items = []
    return [{k: str(item.get(k) or 'Perlu verifikasi literatur tambahan') for k in GAP_FIELDS} for item in items if isinstance(item, dict)]


def excel_col(n):
    s = ''
    while n:
        n, rem = divmod(n-1, 26)
        s = chr(65+rem) + s
    return s


def worksheet_xml(headers, rows):
    data = [headers] + [[r.get(h, '') for h in headers] for r in rows]
    lines = []
    for rownum, row in enumerate(data, 1):
        cells = []
        for colnum, value in enumerate(row, 1):
            ref = f'{excel_col(colnum)}{rownum}'
            style = ' s="1"' if rownum == 1 else ' s="2"'
            cells.append(f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
        lines.append(f'<row r="{rownum}" ht="{32 if rownum == 1 else 75}" customHeight="1">' + ''.join(cells) + '</row>')
    cols = ''.join(f'<col min="{i}" max="{i}" width="{28 if i <= 2 else 46}" customWidth="1"/>' for i in range(1, len(headers)+1))
    ref = f'A1:{excel_col(len(headers))}{len(data)}'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{ref}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{cols}</cols><sheetData>{"".join(lines)}</sheetData><autoFilter ref="{ref}"/></worksheet>')


def save_excel(path, records, gaps):
    sheets = [('Literature Review', FIELDS, records), ('Research Gap', GAP_FIELDS, gaps)]
    types = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,3)) + '</Types>'
    wb = '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + ''.join(f'<sheet name="{escape(s[0])}" sheetId="{i}" r:id="rId{i}"/>' for i,s in enumerate(sheets,1)) + '</sheets></workbook>'
    wb_rels = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,3)) + '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    styles = '<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs></styleSheet>'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', types)
        z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', wb)
        z.writestr('xl/_rels/workbook.xml.rels', wb_rels)
        z.writestr('xl/styles.xml', styles)
        for i, (_, headers, rows) in enumerate(sheets, 1):
            z.writestr(f'xl/worksheets/sheet{i}.xml', worksheet_xml(headers, rows))


def main():
    PDF_DIR.mkdir(exist_ok=True)
    pdfs = sorted(PDF_DIR.glob('*.pdf'))
    if not pdfs:
        sys.exit(f'Tidak ada PDF. Masukkan artikel ke folder: {PDF_DIR}')
    try:
        r = requests.get('http://127.0.0.1:11434/api/tags', timeout=10)
        r.raise_for_status()
        if MODEL not in [m.get('name') for m in r.json().get('models', [])]:
            sys.exit(f'Model {MODEL} tidak ditemukan. Jalankan: ollama pull {MODEL}')
    except requests.RequestException:
        sys.exit('Ollama belum berjalan. Buka aplikasi Ollama atau jalankan: ollama serve')
    print(f'PDF ditemukan: {len(pdfs)}. Menyiapkan ChromaDB...', flush=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(COLLECTION)
    records = []
    for i, path in enumerate(pdfs, 1):
        print(f'[{i}/{len(pdfs)}] Memproses {path.name}', flush=True)
        text = pdf_text(path)
        pieces = chunks(text)
        if not pieces:
            print('  Lewati: teks tidak terbaca (mungkin PDF hasil scan).')
            continue
        doi = doi_from_text(text)
        # Hapus indeks lama untuk PDF ini agar pengulangan tidak menggandakan data.
        col.delete(where={'paper': path.name})
        for start in range(0, len(pieces), 50):
            batch = pieces[start:start+50]
            col.add(ids=[f'{path.name}::{j}' for j in range(start, start+len(batch))],
                    documents=batch, metadatas=[{'paper': path.name, 'doi': doi} for _ in batch])
        print(f'  Indexed {len(pieces)} chunks. DOI: {doi}', flush=True)
        try:
            rec = extract_record(col, path.name, doi)
            records.append(rec)
            print('  Ekstraksi berhasil.', flush=True)
        except Exception as e:
            print('  Ekstraksi gagal:', e)
            records.append({'Paper': path.name, 'DOI': doi, **{k:'Gagal diekstraksi; periksa log' for k in FIELDS[2:]}})
    if not records:
        sys.exit('Tidak ada PDF yang berhasil diproses.')
    print('Membuat Research Gap Matrix...', flush=True)
    try:
        gaps = make_gaps(records)
    except Exception as e:
        print('Pembuatan research gap gagal:', e)
        gaps = []
    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / 'Literature_Review_dan_Research_Gap.xlsx'
    save_excel(target, records, gaps)
    (OUT_DIR / 'hasil_analisis.json').write_text(json.dumps({'literature_review':records, 'research_gap':gaps}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSELESAI: {target}\nLiterature: {len(records)} artikel; Gap: {len(gaps)} baris.', flush=True)
    print('CATATAN: Validasi isi dengan PDF asli sebelum digunakan dalam tesis.')


if __name__ == '__main__':
    main()
