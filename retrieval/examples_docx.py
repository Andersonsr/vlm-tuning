"""
Word report of the text-to-text retrieval examples of retrieval_examples.py, for review by geologists.

Usage (needs python-docx: pip install python-docx):
    python retrieval/examples_docx.py <examples>.json <report>.docx
"""
import json
import os
from argparse import ArgumentParser

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

parser = ArgumentParser()
parser.add_argument('input', help='json written by retrieval_examples.py')
parser.add_argument('out', help='docx to write')
args = parser.parse_args()
with open(args.input, encoding='utf-8') as f:
    data = json.load(f)
out = args.out

CAMPO = data['campo']
MODEL = data.get('model', '')  # spec given to retrieval_examples.py (absent in older files: hf e5-large)
if MODEL.startswith('vlm:'):  # a trainLight.py run: its folder name
    ENCODER_NAME = os.path.basename(os.path.dirname(MODEL[4:].rstrip('/\\'))) or MODEL[4:]
else:
    ENCODER_NAME = data['encoder'].split()[0].split('/')[-1]
SPLIT = os.path.splitext(data.get('split', 'test'))[0].split('_')[-1]  # conv_test.json -> test
N_BEST = sum(s['group'] == 'best' for s in data['samples'])
N_WORST = sum(s['group'] == 'worst' for s in data['samples'])
TOP_K = max(len(s['retrieved']) for s in data['samples'])
TITLE = f'Text-to-text retrieval of {CAMPO} descriptions'
INK, MUTED, ACCENT = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x5B, 0x64, 0x70), RGBColor(0x1D, 0x4E, 0x89)
GOOD, PART, BAD = 'E3F1E6', 'FDF3D7', 'F9E0DE'

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(2.0)

styles = doc.styles
normal = styles['Normal']
normal.font.name, normal.font.size, normal.font.color.rgb = 'Calibri', Pt(10.5), INK
normal.paragraph_format.space_after = Pt(4)
for name, size in (('Title', 22), ('Heading 1', 15), ('Heading 2', 12)):
    st = styles[name]
    st.font.name, st.font.size, st.font.bold = 'Calibri', Pt(size), True
    st.font.color.rgb = INK if name == 'Title' else ACCENT
styles['Heading 1'].paragraph_format.space_before = Pt(14)
styles['Heading 2'].paragraph_format.space_before = Pt(12)
styles['Heading 2'].paragraph_format.keep_with_next = True


def shade(cell, fill):
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), fill)
    # schema order of tcPr children: shd before noWrap ... vAlign
    cell._tc.get_or_add_tcPr().insert_element_before(shd, 'w:noWrap', 'w:tcMar', 'w:textDirection', 'w:tcFitText',
                                                     'w:vAlign', 'w:hideMark', 'w:headers', 'w:cellIns', 'w:cellDel',
                                                     'w:cellMerge', 'w:tcPrChange')


def cell_margins(table, cm=0.12):
    mar = OxmlElement('w:tblCellMar')
    for side in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:w'), str(int(cm * 567))); el.set(qn('w:type'), 'dxa')
        mar.append(el)
    # schema order of tblPr children
    table._tbl.tblPr.insert_element_before(mar, 'w:tblLook', 'w:tblCaption', 'w:tblDescription')


def borders(table, color='C9CED6'):
    b = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4'); el.set(qn('w:space'), '0'); el.set(qn('w:color'), color)
        b.append(el)
    table._tbl.tblPr.insert_element_before(b, 'w:shd', 'w:tblLayout', 'w:tblCellMar', 'w:tblLook', 'w:tblCaption',
                                           'w:tblDescription')


def row_flag(row, tag):
    el = OxmlElement(f'w:{tag}')
    row._tr.get_or_add_trPr().append(el)


def write(cell, text, size=9, bold=False, color=INK, align=None):
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.size, r.bold, r.font.color.rgb = Pt(size), bold, color
    return p


def make_table(widths, header):
    t = doc.add_table(rows=1, cols=len(widths))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t); cell_margins(t)
    for i, (w, h) in enumerate(zip(widths, header)):
        c = t.rows[0].cells[i]
        c.width = Cm(w)
        shade(c, 'EEF1F5')
        write(c, h, size=8.5, bold=True, color=MUTED)
    row_flag(t.rows[0], 'tblHeader')
    return t


def add_row(t, widths, values, fill=None, aligns=None):
    row = t.add_row()
    row_flag(row, 'cantSplit')
    for i, (w, v) in enumerate(zip(widths, values)):
        c = row.cells[i]
        c.width = Cm(w)
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        if fill:
            shade(c, fill)
        if isinstance(v, list):  # one paragraph per line
            for j, line in enumerate(v):
                p = c.paragraphs[0] if j == 0 else c.add_paragraph()
                p.paragraph_format.space_after = Pt(1)
                r = p.add_run(line); r.font.size = Pt(8.5)
        else:
            write(c, v, size=9, align=(aligns or {}).get(i))
    return row


def labels_lines(labels):
    return [f'{cat}: {", ".join(vals)}' for cat, vals in labels.items()]


def para(text, size=10.5, color=INK, italic=False, after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.font.size, r.font.color.rgb, r.italic = Pt(size), color, italic
    return p


def bullet(parts):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    for text, bold in parts:
        p.add_run(text).bold = bold
    return p


agg = data['aggregate']
doc.add_paragraph(TITLE, style='Title')
para(f'{len(data["samples"])} examples from the {ENCODER_NAME} text encoder, for review by geologists', size=12,
     color=MUTED, after=10)

doc.add_heading('What this document shows', level=1)
para(f'We are testing whether a text encoder ({ENCODER_NAME}) places thin-section descriptions '
     f'that say the same thing geologically close together. Each {CAMPO} description of the {SPLIT} set was used as a '
     f'search query against the other {data["n_items"] - 1} {CAMPO} descriptions, and the encoder ranked them by '
     'similarity of meaning. A good encoder should rank first the descriptions annotated with the same labels.')
para(f'The examples below are the {N_BEST} queries the encoder handled best and the {N_WORST} it handled worst, judged '
     'by how high it ranked the descriptions with exactly the same labels (at most one example per label combination). '
     f'For each query we list the {TOP_K} descriptions it retrieved first. We would value your judgement on whether the '
     'retrieved '
     'descriptions are geologically equivalent to the query, especially in the worst cases, where the labels differ '
     'only in part.')

doc.add_heading('How to read the tables', level=2)
bullet([('Same labels: ', True), ('the retrieved description has exactly the same annotated labels as the query.', False)])
if any(s.get('relevant') for s in data['samples']):
    bullet([('Rank: ', True), ('position in the list of all descriptions ranked by the encoder (1 = most similar to '
                               'the query). The second table of an example lists descriptions with the same labels '
                               'that the encoder ranked lower.', False)])
bullet([('Label overlap: ', True), ('share of labels the two descriptions have in common (1.00 = identical, '
                                   '0.50 = half of the labels in common, 0 = none).', False)])
bullet([('Similarity: ', True), ("the encoder's similarity score (cosine, at most 1.00). Only the order matters: the "
                                 'scores are high for all of these similarly worded texts.', False)])
bullet([('Average precision: ', True), ('1.00 when all the descriptions with the same labels are ranked before any '
                                        'other; close to 0 when they are ranked far down the list.', False)])
p = para('Row colours: ', size=10, color=MUTED, after=2)
for label, fill in (('same labels', GOOD), ('labels in common in part', PART), ('no labels in common', BAD)):
    r = p.add_run(f' {label} ')
    r.font.size = Pt(9)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), fill)
    r._r.get_or_add_rPr().append(shd)
    p.add_run('   ')

doc.add_heading(f'Overall results ({CAMPO}, {SPLIT} set)', level=2)
widths = [12.0, 5.0]
t = make_table(widths, ['Measure', 'Value'])
for a, b in ((f'{CAMPO.capitalize()} descriptions in the {SPLIT} set', f'{data["n_items"]}'),
             ('Queries with at least one other description with the same labels', f'{data["n_queries_with_match"]}'),
             ('First retrieved description has the same labels (R@1)', f'{agg["R@1"]:.0%}'),
             ('A description with the same labels among the first 5 (R@5)', f'{agg["R@5"]:.0%}'),
             ('Mean average precision', f'{agg["mAP"]:.2f}'),
             ('Label-overlap quality of the first 10 (nDCG@10, 1.00 = best possible)', f'{agg["nDCG@10"]:.2f}')):
    add_row(t, widths, [a, b])
para('Descriptions with no other description of exactly the same labels are not used as queries for R@1, R@5 and '
     'average precision; they are included in nDCG@10.', size=9, color=MUTED, italic=True, after=6)

WIDTHS = [0.8, 8.5, 4.4, 1.2, 1.2, 0.9]
HEADER = ['#', 'Retrieved description', 'Its labels', 'Label overlap', 'Similarity', 'Same labels']
REL_WIDTHS = [1.2, 14.6, 1.2]  # same total width (17 cm)
REL_HEADER = ['Rank', 'Description with the same labels', 'Similarity']
CENTER = WD_ALIGN_PARAGRAPH.CENTER
n = 0
for group, title, intro in (
        ('best', 'Best-handled queries', 'For these queries the descriptions with the same labels were ranked first.'),
        ('worst', 'Worst-handled queries', 'For these queries the descriptions with the same labels were ranked far down '
                                          'the list; the ones ranked first share only part of the labels.')):
    doc.add_page_break()
    doc.add_heading(title, level=1)
    para(intro, color=MUTED)
    for s in [s for s in data['samples'] if s['group'] == group]:
        n += 1
        doc.add_heading(f'Example {n}', level=2)
        q = doc.add_table(rows=1, cols=1)
        q.alignment = WD_TABLE_ALIGNMENT.CENTER
        q.autofit = False
        cell_margins(q, 0.2)
        row_flag(q.rows[0], 'cantSplit')
        c = q.rows[0].cells[0]
        c.width = Cm(17.0)
        shade(c, 'F3F5F8')
        write(c, 'Query', size=8.5, bold=True, color=MUTED)
        p = c.add_paragraph(); p.paragraph_format.space_after = Pt(4)
        p.add_run(s['query']).font.size = Pt(10.5)
        p = c.add_paragraph(); p.paragraph_format.space_after = Pt(0)
        r = p.add_run('Labels: '); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = MUTED
        p.add_run('; '.join(labels_lines(s['labels']))).font.size = Pt(9)
        others = s['n_exact']
        first = s['first_exact_rank']
        rank_note = ('a description with the same labels was ranked first' if first == 1 else
                     f'the first description with the same labels was at rank {first}')
        note = para(f'Average precision {s["ap"]:.2f} · {rank_note} · {others} other description'
                    f'{"s" if others != 1 else ""} with the same labels in the {SPLIT} set', size=9, color=MUTED, after=4)
        note.paragraph_format.space_before = Pt(4)
        note.paragraph_format.keep_with_next = True
        t = make_table(WIDTHS, HEADER)
        for ret in s['retrieved']:
            fill = GOOD if ret['exact'] else (PART if ret['jaccard'] > 0 else BAD)
            add_row(t, WIDTHS, [str(ret['rank']), ret['text'], labels_lines(ret['labels']), f'{ret["jaccard"]:.2f}',
                                f'{ret["cosine"]:.3f}', 'yes' if ret['exact'] else 'no'],
                    fill=fill, aligns={0: CENTER, 3: CENTER, 4: CENTER, 5: CENTER})
        if s.get('relevant'):  # retrieval_examples.py --relevant
            shown = len(s['relevant'])
            rest = s['n_exact'] - sum(r['exact'] for r in s['retrieved'])  # same labels, outside the first rows
            which = (f'{shown} of the {rest}' if shown < rest else 'the only one' if rest == 1 else f'all {rest}')
            note = para(f'Descriptions with the same labels as the query that were not among the first '
                        f'{len(s["retrieved"])} ({which}), with the rank the encoder gave them:',
                        size=9, color=MUTED, after=4)
            note.paragraph_format.space_before = Pt(6)
            note.paragraph_format.keep_with_next = True
            t = make_table(REL_WIDTHS, REL_HEADER)
            for rel in s['relevant']:
                add_row(t, REL_WIDTHS, [str(rel['rank']), rel['text'], f'{rel["cosine"]:.3f}'], fill=GOOD,
                        aligns={0: CENTER, 2: CENTER})

doc.add_heading('Method notes', level=1)
for text in (f'Encoder: {data["encoder"]}' + ('.' if MODEL.startswith('vlm:') else
                                               ', used as released (not trained on these data).'),
             f'Data: {CAMPO} descriptions of the {SPLIT} split. A description repeated for several images with the '
             'same labels is used once, so a query never finds a copy of itself.',
             'Labels are the annotated (category, value) pairs of each description; "same labels" means all pairs are '
             'equal.'):
    bullet([(text, False)])

# page number in the footer
fp = sec.footer.paragraphs[0]
fp.alignment = CENTER
for kind, text in (('begin', None), (None, 'PAGE'), ('end', None)):
    r = fp.add_run()
    r.font.size, r.font.color.rgb = Pt(8), MUTED
    if kind:
        el = OxmlElement('w:fldChar'); el.set(qn('w:fldCharType'), kind)
    else:
        el = OxmlElement('w:instrText'); el.set(qn('xml:space'), 'preserve'); el.text = text
    r._r.append(el)
doc.core_properties.title = TITLE
zoom = doc.settings.element.find(qn('w:zoom'))  # python-docx's template omits the required percent
if zoom is not None:
    zoom.set(qn('w:percent'), '100')
doc.save(out)
print('saved', out)
