"""Reproduce the final main figures from aggregate source data."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import argparse, csv, hashlib, html, json, math, os, shutil, subprocess
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'outputs/figure_reproduction/main'
SOURCE_ROOT = REPO / 'source_data/main'
POPPLER = shutil.which('pdftoppm')
RENDER_REVIEW_PNGS = False

def font_file(name):
    override = os.environ.get('NMH_ARIAL_FONT_DIR')
    candidates = ([Path(override)] if override else []) + [
        Path('/System/Library/Fonts/Supplemental'),
        Path('/Library/Fonts'),
    ]
    for folder in candidates:
        path = folder / name
        if path.exists():
            return path
    fallback = {
        'Arial.ttf': [
            Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'),
            Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),
        ],
        'Arial Bold.ttf': [
            Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'),
            Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'),
        ],
    }
    for path in fallback.get(name, []):
        if path.exists():
            return path
    raise FileNotFoundError(
        f'{name} was not found. Set NMH_ARIAL_FONT_DIR to a directory containing Arial.ttf and Arial Bold.ttf.'
    )

pdfmetrics.registerFont(TTFont('ArialLocal', str(font_file('Arial.ttf'))))
pdfmetrics.registerFont(TTFont('ArialLocalBold', str(font_file('Arial Bold.ttf'))))
W = 180 * 72 / 25.4
BLACK = '#000000'
GREY = '#666666'
BLUE = '#156082'  # source Hamilton Fig. 1 line RGB approximately (0.082, .376, .510)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def nstr(r): return f"{int(float(r['n'])):,}"
def num(v):
    x = Decimal(str(v)).quantize(Decimal('0.0001'))
    if x == 0: x = abs(x)
    return f'{x:.4f}'.replace('-', '−')

class Figure:
    def __init__(self, key, height):
        self.key, self.h = key, height
        self.stem = OUT / 'figures' / key
        self.c = canvas.Canvas(str(self.stem.with_suffix('.pdf')), pagesize=(W, height), invariant=1)
        self.c.setTitle(key.replace('_', ' '))
        self.c.setAuthor('Huiyun Yu and Bo Li')
        self.c.setSubject('Aggregate source-data figure for the accompanying manuscript.')
        self.c.setCreator('Aggregate source-data figure renderer')
        self.svg = []
        self.text_boxes = []
        self.points = []
        self.box(0, 0, W, height, stroke=None)
    def text(self, x, y, text, size=7.5, bold=False, anchor='start', color=BLACK):
        text = str(text)
        font = 'ArialLocalBold' if bold else 'ArialLocal'
        width = pdfmetrics.stringWidth(text, font, size)
        left = x - (width / 2 if anchor == 'middle' else width if anchor == 'end' else 0)
        assert left >= -0.01 and left + width <= W + .01, (self.key, text, left, width)
        assert size <= y <= self.h - 1, (self.key, text, y)
        self.c.setFont(font, size)
        self.c.setFillColor(HexColor(color))
        {'start': self.c.drawString, 'middle': self.c.drawCentredString, 'end': self.c.drawRightString}[anchor](x, self.h-y, text)
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>')
        self.text_boxes.append({'text': text, 'left': left, 'right': left+width, 'baseline': y, 'size': size})
    def line(self, x1, y1, x2, y2, color=BLACK, width=.5, dash=False):
        self.c.setStrokeColor(HexColor(color)); self.c.setLineWidth(width)
        self.c.setDash([2, 2] if dash else [])
        self.c.line(x1, self.h-y1, x2, self.h-y2)
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"' + (' stroke-dasharray="2,2"' if dash else '') + '/>')
    def box(self, x, y, w, h, stroke=BLACK):
        self.c.setDash([]); self.c.setLineWidth(.5)
        self.c.setFillColor(HexColor('#FFFFFF')); self.c.setStrokeColor(HexColor(stroke or '#FFFFFF'))
        self.c.rect(x, self.h-y-h, w, h, fill=1, stroke=bool(stroke))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="{stroke or "none"}" stroke-width="0.5"/>')
    def arrow(self, x1, y1, x2, y2):
        self.line(x1, y1, x2, y2, BLUE)
        a=math.atan2(y2-y1, x2-x1)
        for da in [-.45, .45]: self.line(x2, y2, x2-3.5*math.cos(a+da), y2-3.5*math.sin(a+da), BLUE)
    def point(self, x, y):
        self.c.setDash([]); self.c.setFillColor(HexColor(BLACK))
        self.c.circle(x, self.h-y, 1.65, fill=1, stroke=0)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="1.65" fill="black"/>')
    def save(self, source_records):
        self.c.showPage(); self.c.save()
        meta = {'source_records': source_records, 'plotted_rows': self.points, 'aggregate_source_data_only': True}
        self.stem.with_suffix('.svg').write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{self.h*25.4/72:.4f}mm" viewBox="0 0 {W} {self.h}">'
            f'<metadata>{html.escape(json.dumps(meta))}</metadata>' + ''.join(self.svg) + '</svg>',
            encoding='utf-8',
        )
        if RENDER_REVIEW_PNGS:
            if POPPLER is None:
                raise RuntimeError('pdftoppm is required for PNG rendering and was not found on PATH.')
            for dpi, dst in [(600, self.stem), (180, OUT/'review'/self.key)]:
                subprocess.run(
                    [POPPLER, '-singlefile', '-r', str(dpi), '-png', str(self.stem.with_suffix('.pdf')), str(dst)],
                    check=True,
                    capture_output=True,
                )
            (OUT/'review'/f'{self.key}_layout.json').write_text(
                json.dumps(self.text_boxes, indent=2), encoding='utf-8'
            )
        return {
            'figure': self.key,
            'width_mm': 180,
            'height_mm': self.h*25.4/72,
            'plotted_source_rows': self.points,
            'source_record_count': len(source_records),
        }

def count(rr, panel, cohort, outcome=None):
    g=[r for r in rr if r['panel']==panel and r['cohort']==cohort and (outcome is None or r['outcome']==outcome)]
    assert len(g)==1
    return nstr(g[0])

def design(rr):
    d=Figure('Figure_1', 448)
    d.box(8,10,W-16,48)
    d.text(13,21,'Cohorts: isolated primary adversity episodes',7.5)
    d.text(W/2,36,f"UKHLS: {count(rr,'Trajectory cohort','UKHLS')} people",8,anchor='middle')
    d.text(W/2,48,f"HRS: {count(rr,'Trajectory cohort','HRS')} people",8,anchor='middle')
    pw=(W-32)/3; xs=[8,16+pw,24+2*pw]; centers=[x+pw/2 for x in xs]
    d.line(W/2,58,W/2,71,BLUE); d.line(centers[0],71,centers[-1],71,BLUE)
    headings=['a  One prior, different adversity','b  Recurrent caregiving','c  Multiple prior events']
    texts=[
      [('Earlier adversity',114),('Completed acute /',131),('two-year response',142),('Later adversity',183),('of a different type',194),('Acute / two-year response',220)],
      [('Earlier caregiving episode',114),('Completed acute /',131),('two-year response',142),('Observed no-caregiving reset',183),('Later caregiving episode',205),('Acute / two-year response',220)],
      [('At least two prior episodes',114),('Completed acute responses',135),('before target pre1',146),('Mean prior acute response',183),('Earliest eligible target event',205),('Acute response',220)]
    ]
    for k,(x,c) in enumerate(zip(xs,centers)):
        d.arrow(c,71,c,84); d.box(x,84,pw,209)
        d.text(x+5,97,headings[k],7.3,bold=True)
        for text,y in texts[k]:d.text(c,y,text,7.3,anchor='middle')
        d.arrow(c,153,c,169)
        if k==0:d.arrow(c,197,c,209)
        if k==1:
            d.text(c,240,'Acute | Two-year persistence',7.1,anchor='middle')
            for co,y in [('UKHLS',253),('HRS',265)]:
                d.text(c,y,f"{co}: {count(rr,'B Recurrence',co,'acute')} | {count(rr,'B Recurrence',co,'persistence')}",7.5,anchor='middle')
        else:
            panel='A Cross domain' if k==0 else 'C Multiple histories'
            for co,y in [('UKHLS',247),('HRS',260)]: d.text(c,y,f"{co}: n = {count(rr,panel,co)}",7.5,anchor='middle')
        if k>0:d.text(c,282,'Registered before estimation',6.9,anchor='middle')
        d.line(c,293,c,308,BLUE)
    d.line(centers[0],308,centers[-1],308,BLUE); d.arrow(W/2,308,W/2,323)
    d.box(8,323,W-16,48)
    d.text(13,335,'Current-state benchmark',7.5)
    d.text(W/2,350,'Mental health measured immediately before the target event (pre1)',8,anchor='middle')
    d.text(W/2,362,'Compare prior-response information before and after adding current state',7.5,anchor='middle')
    d.arrow(W/2,371,W/2,391)
    d.box(8,391,W-16,42)
    d.text(13,403,'Held-out evaluation',7.5)
    d.text(W/2,418,'Person-grouped cross-validation and temporal holdout',8,anchor='middle')
    d.text(W/2,429,'Information flow, not a causal diagram; counts refer to eligible people',7,anchor='middle')
    d.points=[int(r['display_row']) for r in rr]
    return d.save(rr)

def panel_height(groups, compact=False):
    return 60+sum((9+10*len(g[1])+3) if compact else (13+12*len(g[1])+5) for g in groups)

def forest_panel(d, top, letter, title, groups, lim, ticks, subtitle=None, compact=False):
    left, ax, end, right=8,181,387,W-8
    d.text(left,top+10,letter,10,bold=True)
    d.text(left+14,top+10,title,8.5,bold=True)
    if subtitle:d.text(right,top+10,subtitle,7,anchor='end')
    d.text(left,top+27,'Comparison / cohort (n)',7.5)
    d.text(right,top+27,'ΔR² (95% CI)',7.5,anchor='end')
    d.line(left,top+31,right,top+31)
    yy=top+43
    ybottom=top+panel_height(groups,compact)-20
    sx=lambda v:ax+(float(v)-lim[0])/(lim[1]-lim[0])*(end-ax)
    d.line(sx(0),top+34,sx(0),ybottom,GREY,.4,True)
    for label, gg in groups:
        d.text(left,yy,label,7.5); yy+=9 if compact else 13
        for r in gg:
            d.text(left+8,yy,f"{r['cohort']} ({nstr(r)})",7.5)
            est,lo,hi=[float(r[k]) for k in ['estimate','ci_lower','ci_upper']]
            assert lim[0]<=lo<=est<=hi<=lim[1], (d.key,r['display_row'],lo,hi,lim)
            py=yy-2.6
            d.line(sx(lo),py,sx(hi),py,GREY,.55)
            for v in [lo,hi]:d.line(sx(v),py-2.2,sx(v),py+2.2,GREY,.5)
            d.point(sx(est),py)
            d.text(right,yy,f"{num(r['estimate'])} ({num(r['ci_lower'])}, {num(r['ci_upper'])})",7.2,anchor='end')
            d.points.append(int(r['display_row'])); yy+=10 if compact else 12
        yy+=3 if compact else 5
    ay=yy+1
    d.line(ax,ay,end,ay)
    for tick in ticks:
        x=sx(tick);d.line(x,ay,x,ay+3)
        d.text(x,ay+12,f'{tick:.2f}'.replace('-','−'),7,anchor='middle')
    d.text((ax+end)/2,ay+25,'Change in held-out predictive R²',7.5,anchor='middle')
    return ay+32

def select(rr, **kw):
    g=[r for r in rr if all(r[k]==v for k,v in kw.items())]
    return sorted(g, key=lambda r: ['UKHLS','HRS'].index(r['cohort']))

def forests(rr2,rr3,rr4):
    audit=[]
    groups2=[]
    for outcome in ['acute','persistence']:
        groups2.append([(label,select(rr2,outcome=outcome,label=label)) for label in ['Prior response without current state','Current pre1 without history','Prior response after current pre1']])
    gap=35
    d=Figure('Figure_2',sum(panel_height(g) for g in groups2)+gap+25)
    top=0
    for k,g in enumerate(groups2):
        forest_panel(d,top,'ab'[k],['Acute response','Two-year persistence'][k],g,(-.03,.38),[0,.1,.2,.3], 'Person-grouped ten-fold CV')
        top+=panel_height(g)+gap
    audit.append(d.save(rr2))

    groups3=[]
    for panel in ['A Caregiving acute','B Caregiving persistence','C Multiple acute histories']:
        gg=[]
        for val,vtitle in [('grouped_cv','Grouped CV'),('temporal','Temporal holdout')]:
            for suffix,title in [('without','without current state'),('after','after current state')]:
                found=[r for r in select(rr3,panel=panel,validation=val) if f' {suffix} ' in r['label']]
                gg.append((f'{vtitle}: history {title}',found))
        groups3.append(gg)
    gap=20
    d=Figure('Figure_3',sum(panel_height(g,True) for g in groups3)+gap*2+33)
    top=0
    for k,g in enumerate(groups3):
        forest_panel(d,top,'abc'[k],['Recurrent caregiving: acute response','Recurrent caregiving: two-year persistence','Multiple prior events: mean acute-response history'][k],g,(-.14,.10),[-.1,-.05,0,.05,.1],compact=True)
        top+=panel_height(g,True)+gap
    d.text(8,d.h-8,'UKHLS: primary; HRS: secondary. Error bars show person-cluster bootstrap 95% confidence intervals.',7)
    audit.append(d.save(rr3))

    groups4=[]
    for val in ['grouped_cv','temporal']:
        groups4.append([(lab,select(rr4,validation=val,label=raw)) for lab,raw in [
            ('Current pre1 instead of earlier pre2','Current pre1 instead of earlier pre2'),
            ('Add earlier pre2 after current pre1','Add pre2 after current pre1'),
            ('Add history after both states','History after both states')]])
    gap=35
    d=Figure('Figure_4',sum(panel_height(g) for g in groups4)+gap+39)
    top=0
    for k,g in enumerate(groups4):
        forest_panel(d,top,'ab'[k],['Person-grouped cross-validation','Temporal holdout'][k],g,(-.10,.31),[-.1,0,.1,.2,.3])
        top+=panel_height(g)+gap
    d.text(8,d.h-10,'Multi-event acute-history sample; all analyses secondary; first contrast compares non-nested models.',7)
    audit.append(d.save(rr4))
    return audit

def verify_source(r):
    original_rel = Path(r['source_file'])
    src = REPO / original_rel
    if not src.is_file():
        raise AssertionError(f'Missing source file: {original_rel}')
    original=rows(src)[int(r['source_row_0based'])]
    for k,v in json.loads(r['source_filter']).items(): assert str(original[k])==str(v),(src,k)
    for target,col in [('estimate',r['source_estimate_column']),('n',r['source_n_column'])]:
        # Previously published aggregate CSVs may differ by floating-point serialization.
        tolerance = Decimal('0') if target=='n' else Decimal('1e-14')
        assert abs(Decimal(r[target])-Decimal(original[col]))<=tolerance,(src,target)
    for target in ['ci_lower','ci_upper']:
        if r[target]:assert abs(Decimal(r[target])-Decimal(original[target]))<Decimal('1e-14'),(src,target)

def main():
    global OUT, RENDER_REVIEW_PNGS
    parser=argparse.ArgumentParser(description='Reproduce final main figures from aggregate source data only.')
    parser.add_argument('--output-dir', type=Path, default=OUT, help='Output directory; expected outputs are never overwritten.')
    parser.add_argument('--render-review-pngs', action='store_true', help='Also render review PNGs with pdftoppm.')
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    RENDER_REVIEW_PNGS=args.render_review_pngs
    for f in ['figures','source_data','review']: (OUT/f).mkdir(parents=True, exist_ok=True)
    srcs=[SOURCE_ROOT/f'Figure_{k}_source.csv' for k in range(1,5)]
    original_hashes={str(p.relative_to(REPO)):sha(p) for p in srcs}
    datasets=[]
    for p in srcs:
        shutil.copy2(p, OUT/'source_data'/p.name)
        datasets.append(rows(p))
    datasets[3]=[r for r in datasets[3] if r['panel']=='C Multiple acute histories']
    assert [len(g) for g in datasets]==[10,12,24,12]
    for g in datasets:
        for r in g: verify_source(r)
    audit=[design(datasets[0])]+forests(*datasets[1:])
    for a,g in zip(audit,datasets):
        assert sorted(a['plotted_source_rows'])==sorted(int(r['display_row']) for r in g)
    for p,v in original_hashes.items():assert sha(REPO/p)==v
    report={'created_at':datetime.now(timezone.utc).isoformat(),'input':'repository aggregate source data','source_values_verified':True,'original_inputs_unchanged':original_hashes,'figures':audit,'font':'Embedded Arial when available; Liberation Sans fallback otherwise','review_pngs_rendered':RENDER_REVIEW_PNGS,'figure4_display_filter':"panel == 'C Multiple acute histories'"}
    (OUT/'review/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (OUT/'source_data/README.md').write_text('# Figure source data\n\nThe four CSV files are copied from the repository source-data folder. Figure 4 uses the 12 rows where `panel == C Multiple acute histories`; the full 36-row source file is retained. Source-row indices are zero-based and display-row indices are one-based. The SVG metadata contain the source records used in each display.\n', encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
