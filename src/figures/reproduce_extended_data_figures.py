"""Draw Extended Data figures from the archived aggregate results.

Only presentation is changed. Source rows, estimates, intervals and counts are
read without alteration. Participant-level data are neither needed nor read.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

PAGE_W = 180 / 25.4 * 72
MARGIN = 18.0
GAP = 20.0
INK = '#000000'
GRAY = '#666666'
GRID = '#DDDDDD'
FONT = 'EDSans'
BOLD = 'EDSansBold'
FAMILIES = ['widowhood', 'unemployment', 'caregiving', 'health']
FAMILY_NAMES = {'widowhood':'Widowhood', 'unemployment':'Unemployment',
                'caregiving':'Caregiving', 'health':'Health adversity',
                'financial_strain':'Financial strain'}
# Match each curve, marker and legend key; shapes provide a second cue.
FAMILY_STYLES = {
    'widowhood': ('#17496B', 'circle'),
    'unemployment': ('#B45E27', 'square'),
    'caregiving': ('#357862', 'triangle'),
    'health': ('#81569E', 'diamond'),
}
TITLES = {
    1:'Event-centred mental-health trajectories',
    2:'Episode construction and follow-up availability',
    3:'Event-specific versus common trajectory models',
    4:'Cross-adversity prediction: robustness checks',
    5:'Item completeness and reliability',
    6:'Response history within recurrent adversity domains',
    7:'Multiple acute-response histories: alternative representations',
    8:'Attrition weighting and complete-case sensitivities',
}
PANEL_NAMES = {
    'acute change': ('Acute response','Change-score model'),
    'acute ancova_level': ('Acute response','ANCOVA model'),
    'persistence change': ('Two-year persistence','Change-score model'),
    'persistence ancova_level': ('Two-year persistence','ANCOVA model'),
    'acute Without current state': ('Acute response','Without current mental health'),
    'acute After current state': ('Acute response','After current mental health'),
    'persistence Without current state': ('Two-year persistence','Without current mental health'),
    'persistence After current state': ('Two-year persistence','After current mental health'),
    'Without current state': ('Without current mental health',''),
    'After current state': ('After current mental health',''),
    'UKHLS persistence_2y_z': ('UKHLS','Two-year persistence'),
    'UKHLS persistence_4y_z': ('UKHLS','Four-year persistence'),
    'HRS persistence_2y_z': ('HRS','Two-year persistence'),
    'HRS persistence_4y_z': ('HRS','Four-year persistence'),
}
FOREST_LABELS = {
    'Event specific minus universal': 'Event-specific minus common trajectory',
    'Prior response without current state': 'Prior response without current mental health',
    'Prior response after current pre1': 'Prior response after current mental health',
    'Split form context': 'Split-half response without current mental health',
    'Split form current_state': 'Split-half response after current mental health',
    'Split form ancova_current_state': 'Split-half response after current mental health',
    **FAMILY_NAMES,
}
HISTORY_NAMES = {
    'between_history_sd': 'Variability of prior responses',
    'mean': 'Mean prior response',
    'most_recent': 'Most recent prior response',
    'residualized_mean': 'Residualized mean prior response',
}
METHOD_NAMES = {'unweighted':'Unweighted', 'IPOW_truncated_1_99':'IPOW',
                'complete_four_points':'Complete case'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fonts():
    override=os.environ.get('NMH_ARIAL_FONT_DIR')
    folders=([Path(override)] if override else []) + [Path('/System/Library/Fonts/Supplemental'),Path('/Library/Fonts')]
    pairs=[(p/'Arial.ttf',p/'Arial Bold.ttf') for p in folders]
    # Approved Extended Data PDFs used Liberation Sans. Pin their default font
    # rather than silently choosing a different face on macOS.
    vendored=Path(__file__).resolve().parents[2]/'assets/fonts'
    if not override:
        pairs.insert(0,(vendored/'LiberationSans-Regular.ttf',vendored/'LiberationSans-Bold.ttf'))
    pairs += [(Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf')),
              (Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'),Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))]
    pair=next(((a,b) for a,b in pairs if a.is_file() and b.is_file()),None)
    if pair is None:
        raise FileNotFoundError('Set NMH_ARIAL_FONT_DIR to a folder containing Arial.ttf and Arial Bold.ttf, or install Liberation Sans.')
    pdfmetrics.registerFont(TTFont(FONT,str(pair[0])))
    pdfmetrics.registerFont(TTFont(BOLD,str(pair[1])))
    return pair[0].name


def finite(v):
    try:return math.isfinite(float(v))
    except (ValueError,TypeError):return False


def num(v):return float(v)

def format_number(v,precision=4):
    value=float(v)
    if abs(value)<0.5*10**(-precision):value=0.0
    return f'{value:.{precision}f}'.replace('-', '\u2212')

def estimate_text(row,ci=True):
    if not finite(row.get('estimate')):return 'Not available'
    s=format_number(row['estimate'])
    if ci and finite(row.get('ci_lower')) and finite(row.get('ci_upper')):
        s += f" ({format_number(row['ci_lower'])}, {format_number(row['ci_upper'])})"
    return s


def group(rows,key):
    result=OrderedDict()
    for row in rows:result.setdefault(row[key],[]).append(row)
    return result


def wrapped_lines(text,width,size,bold=False):
    fn=BOLD if bold else FONT
    lines=[];line=''
    for word in str(text).split():
        proposal=(line+' '+word).strip()
        if line and pdfmetrics.stringWidth(proposal,fn,size)>width:
            lines.append(line);line=word
        else:line=proposal
    if line:lines.append(line)
    if any(pdfmetrics.stringWidth(s,fn,size)>width+0.1 for s in lines):
        raise ValueError(f'Unbreakable label exceeds available width: {text}')
    return lines


class Drawing:
    """Vector PDF/SVG pair with a per-mark numeric audit."""
    def __init__(self,stem,w,h,title):
        self.stem=stem;self.w=w;self.h=h
        self.c=canvas.Canvas(str(stem.with_suffix('.pdf')),pagesize=(w,h),invariant=1)
        self.c.setTitle(title);self.c.setAuthor('Huiyun Yu and Bo Li')
        self.c.setSubject('Extended Data figure generated from archived aggregate results')
        self.c.setCreator('Aggregate-data figure renderer')
        self.svg=[];self.text_boxes=[];self.marks=[];self.labels=[]
        self.rect(0,0,w,h,'#FFFFFF')
    def rect(self,x,y,w,h,fill,stroke=None):
        self.c.setFillColor(HexColor(fill));self.c.setStrokeColor(HexColor(stroke or fill));self.c.setDash([])
        self.c.rect(x,self.h-y-h,w,h,stroke=bool(stroke),fill=1)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke or fill}"/>')
    def line(self,x1,y1,x2,y2,color=GRID,width=.5,dash=False):
        self.c.setStrokeColor(HexColor(color));self.c.setLineWidth(width);self.c.setDash([2.4,2.2] if dash else [])
        self.c.line(x1,self.h-y1,x2,self.h-y2)
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="2.4,2.2"' if dash else '')+'/>')
    def text(self,x,y,text,size=8,color=INK,bold=False,anchor='start'):
        text=str(text);fn=BOLD if bold else FONT
        w=pdfmetrics.stringWidth(text,fn,size)
        xx=x-w/2 if anchor=='middle' else x-w if anchor=='end' else x
        ascent,descent=pdfmetrics.getAscentDescent(fn,size)
        if xx<-.05 or xx+w>self.w+.05 or y-ascent<-.05 or y-descent>self.h+.05:
            raise ValueError(f'Text outside canvas: {text!r}')
        self.c.setFillColor(HexColor(color));self.c.setFont(fn,size)
        self.c.drawString(xx,self.h-y,text)
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial,Liberation Sans,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>')
        self.text_boxes.append({'text':text,'bbox':[xx,y-ascent,xx+w,y-descent],'size':size})
    def rich_r2(self,x,y,prefix='',suffix='',size=7.1,anchor='middle'):
        # Normal 2 with a raised baseline; no pre-composed superscript character.
        parts=[(prefix+'R',size,0),('2',size*.72,-size*.36),(suffix,size,0)]
        widths=[pdfmetrics.stringWidth(t,FONT,s) for t,s,_ in parts];total=sum(widths)
        cur=x-total/2 if anchor=='middle' else x-total if anchor=='end' else x
        for (t,s,dy),w in zip(parts,widths):self.text(cur,y+dy,t,s);cur+=w
    def wrap(self,x,y,text,width,size=8,bold=False,color=INK,leading=None):
        lines=wrapped_lines(text,width,size,bold);leading=leading or size+3
        for i,line in enumerate(lines):self.text(x,y+i*leading,line,size,color,bold)
        return max(1,len(lines))*leading
    def mark(self,x,y,color=INK,shape='circle',open=False,r=2.3):
        self.c.setDash([]);self.c.setStrokeColor(HexColor(color));self.c.setLineWidth(.75)
        fill='#FFFFFF' if open else color;self.c.setFillColor(HexColor(fill))
        if shape=='circle':
            self.c.circle(x,self.h-y,r,stroke=1,fill=1)
            self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{color}" stroke-width=".75"/>')
        elif shape=='square':
            self.c.rect(x-r,self.h-y-r,2*r,2*r,stroke=1,fill=1)
            self.svg.append(f'<rect x="{x-r}" y="{y-r}" width="{2*r}" height="{2*r}" fill="{fill}" stroke="{color}" stroke-width=".75"/>')
        else:
            pts=[(x,y-r*1.2),(x-r*1.12,y+r*.8),(x+r*1.12,y+r*.8)] if shape=='triangle' else [(x,y-r*1.28),(x-r*1.1,y),(x,y+r*1.28),(x+r*1.1,y)]
            p=self.c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
            for px,py in pts[1:]:p.lineTo(px,self.h-py)
            p.close();self.c.drawPath(p,stroke=1,fill=1)
            self.svg.append('<polygon points="'+' '.join(f'{px},{py}' for px,py in pts)+f'" fill="{fill}" stroke="{color}" stroke-width=".75"/>')
    def record(self,row,x,y,x_min=None,x_max=None,plot_left=None,plot_right=None,**extra):
        self.marks.append({'display_row':int(row['display_row']),'panel':row['panel'],'label':row['label'],
                           'cohort':row['cohort'],'estimate':row['estimate'],'ci_lower':row.get('ci_lower',''),
                           'ci_upper':row.get('ci_upper',''),'n':row.get('n',''),'x':x,'y':y,
                           'axis_min':x_min,'axis_max':x_max,'plot_left':plot_left,'plot_right':plot_right,**extra})
    def mapping(self,kind,old,new):
        r={'kind':kind,'source_label':old,'display_label':new}
        if r not in self.labels:self.labels.append(r)
    def save(self,source,rows):
        ids=sorted(int(r['display_row']) for r in rows)
        assert sorted(m['display_row'] for m in self.marks)==ids
        overlaps=[]
        for i,a in enumerate(self.text_boxes):
            ax0,ay0,ax1,ay1=a['bbox']
            for b in self.text_boxes[i+1:]:
                bx0,by0,bx1,by1=b['bbox']
                if min(ax1,bx1)-max(ax0,bx0)>.1 and min(ay1,by1)-max(ay0,by0)>.1:
                    overlaps.append([a['text'],b['text']])
        if overlaps:raise ValueError(f'Text overlap in {self.stem.name}: {overlaps[:10]}')
        self.c.showPage();self.c.save()
        meta=html.escape(json.dumps({'source_csv':source,'plotted_source_rows':ids}))
        self.stem.with_suffix('.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" viewBox="0 0 {self.w} {self.h}"><metadata>{meta}</metadata>'+''.join(self.svg)+'</svg>',encoding='utf-8')
        audit={'figure':self.stem.name,'width_pt':self.w,'height_pt':self.h,'source_rows':len(rows),'plotted_rows':len(self.marks),
               'text_overlap_pairs':overlaps,'minimum_text_size_pt':min(b['size'] for b in self.text_boxes),
               'marks':self.marks,'label_mapping':self.labels,'text_boxes':self.text_boxes}
        self.stem.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        return audit


def nice_bounds(rows,with_ci=True,ticks=3,include_zero=True):
    cols=['estimate','ci_lower','ci_upper'] if with_ci else ['estimate']
    vs=[float(r[c]) for r in rows for c in cols if finite(r.get(c))]
    low=min(vs+[0.0]) if include_zero else min(vs)
    high=max(vs+[0.0]) if include_zero else max(vs)
    span=max(high-low,.005)
    raw=span/(ticks-1)
    exp=10**math.floor(math.log10(raw))
    step=next((v*exp for v in [1,2,2.5,5,10] if v*exp>=raw-1e-12),10*exp)
    lo=math.floor((low-.03*span)/step)*step
    hi=math.ceil((high+.03*span)/step)*step
    # Keep zero at the edge rather than inventing a negative range for positive data.
    if low==0 and min(vs)>=0:lo=0
    if high==0 and max(vs)<=0:hi=0
    ts=[];t=lo
    while t<=hi+step*.01:ts.append(t);t+=step
    if len(ts)>6:ts=ts[::2]+([ts[-1]] if len(ts)%2==0 else [])
    return lo,hi,ts


def forest_bounds(rows):
    values=[float(r[c]) for r in rows for c in ('estimate','ci_lower','ci_upper') if finite(r.get(c))]
    low=min([0.0,*values]);high=max([0.0,*values]);span=max(high-low,.005)
    lo=low-.07*span;hi=high+.07*span
    if low==0 and min(values)>=0:lo=0
    if high==0 and max(values)<=0:hi=0
    raw=(hi-lo)/3
    exponent=10**math.floor(math.log10(raw))
    step=next(v*exponent for v in (1,2,2.5,5,10) if v*exponent>=raw-1e-12)
    start=math.ceil((lo-1e-12)/step);end=math.floor((hi+1e-12)/step)
    ticks=[j*step for j in range(start,end+1)]
    if len(ticks)<2:ticks=[lo,hi]
    return lo,hi,ticks


def tick_format(value,step):
    prec=max(0,min(4,-int(math.floor(math.log10(abs(step)))) + (1 if abs(step)/10**math.floor(math.log10(abs(step)))==2.5 else 0)))
    return format_number(value,prec)


def axis(d,left,right,y,lo,hi,ticks,label='r2',top=None):
    sx=lambda v:left+(float(v)-lo)/(hi-lo)*(right-left)
    d.line(left,y,right,y,GRAY,.45)
    if top is not None and lo<=0<=hi:d.line(sx(0),top,sx(0),y,'#AAAAAA',.45,True)
    step=ticks[1]-ticks[0] if len(ticks)>1 else hi-lo
    for t in ticks:
        x=sx(t);d.line(x,y,x,y+3,GRAY,.45);d.text(x,y+12,tick_format(t,step),7,anchor='middle')
    if label=='r2':d.rich_r2((left+right)/2,y+26,prefix='Change in predictive ',size=7.1)
    elif label:d.text((left+right)/2,y+26,label,7.1,anchor='middle')


def trajectory(rows,stem,source):
    d=Drawing(stem,PAGE_W,519,TITLES[1]);d.text(MARGIN,22,TITLES[1],11,bold=True)
    available=PAGE_W-2*MARGIN
    for j,fam in enumerate(FAMILIES):
        x=MARGIN+j*available/4;color,shape=FAMILY_STYLES[fam]
        d.line(x,43,x+18,43,color,1.1);d.mark(x+9,43,color,shape,r=2.3)
        d.text(x+23,46,FAMILY_NAMES[fam],7.7)
        d.mapping('event',fam,FAMILY_NAMES[fam])
    for k,co in enumerate(['UKHLS','HRS']):
        g=[r for r in rows if r['cohort']==co];head=75+k*211
        d.text(MARGIN,head,f'{chr(97+k)}  {co}',9.4,bold=True)
        d.text(104,head,f"{int(float(g[0]['n'])):,} people; {int(float(g[0]['n_episodes'])):,} episodes",7.7)
        left=59;right=PAGE_W-24;top=head+30;bottom=top+131
        lo,hi,ticks=nice_bounds(g,True,6,False)
        sy=lambda v:bottom-(float(v)-lo)/(hi-lo)*(bottom-top)
        d.text(left,top-11,'Mental-health score (pre-event SD units)',7.6)
        for t in ticks:
            yy=sy(t);d.line(left,yy,right,yy,GRID,.4);d.text(left-8,yy+2.5,tick_format(t,ticks[1]-ticks[0]),7.3,anchor='end')
        times=['pre2','pre1','event','year2','year4'];time_labels=['Earlier pre-event','Immediate pre-event','Event','2 years','4 years']
        for fi,fam in enumerate(FAMILIES):
            subset={r['time_point']:r for r in g if r['label']==fam};prev=None;color,shape=FAMILY_STYLES[fam]
            for j,t in enumerate(times):
                if t not in subset:continue
                r=subset[t];x=left+j*(right-left)/4+(fi-1.5)*1.65;y=sy(r['estimate'])
                if prev:d.line(*prev,x,y,color,1.05)
                if finite(r.get('ci_lower')) and finite(r.get('ci_upper')):
                    yl=sy(r['ci_lower']);yu=sy(r['ci_upper']);d.line(x,yl,x,yu,color,.65)
                    d.line(x-1.3,yl,x+1.3,yl,color,.6);d.line(x-1.3,yu,x+1.3,yu,color,.6)
                d.mark(x,y,color,shape,r=2.1);prev=(x,y)
                d.record(r,x,y,color=color,shape=shape,axis_y_min=lo,axis_y_max=hi,plot_top=top,plot_bottom=bottom)
        for j,lab in enumerate(time_labels):
            d.text(left+j*(right-left)/4,bottom+16,lab,7.0,anchor='middle');d.mapping('time',times[j],lab)
    d.text(MARGIN,488,'Higher scores indicate worse mental health. Error bars show 95% confidence intervals.',7.3,GRAY)
    d.text(MARGIN,503,'Time points are categories; horizontal spacing does not represent elapsed time.',7.3,GRAY)
    return d.save(source,rows)


FLOW_NAMES={
'all_collapsed_candidate_episodes':'Collapsed candidate episodes',
'any_primary_episode':'Episodes containing a primary adversity',
'isolated_primary_episode':'Isolated primary-adversity episodes',
'valid_pre1_and_event_self_report':'Valid self-report before and at the event',
'valid_age_and_binary_sex':'Final trajectory sample',
'has_pre2':'Additional earlier pre-event assessment available',
'has_uncensored_year2':'Observed, uncensored two-year follow-up',
'has_uncensored_year4':'Observed, uncensored four-year follow-up',
'complete_four_timepoints':'Complete pre-event, event, two- and four-year observations',
'isolated_primary':'Isolated primary-adversity episodes',
'compound_primary':'Multiple primary adversities at the same transition',
'primary_with_any_secondary_concurrent':'Primary adversity with a concurrent secondary event',
'year2_censored_by_later_primary':'Two-year follow-up censored by a later primary adversity',
'year4_censored_by_later_primary':'Four-year follow-up censored by a later primary adversity',
}

def flow(rows,stem,source):
    d=Drawing(stem,PAGE_W,405,TITLES[2]);d.text(MARGIN,23,TITLES[2],11,bold=True)
    d.text(MARGIN,44,'Counts refer to episodes, not unique people.',7.7,GRAY)
    x1=PAGE_W-104;x2=PAGE_W-MARGIN
    d.text(MARGIN,65,'Stage or subset',8,bold=True);d.text(x1,65,'UKHLS',8,bold=True,anchor='end');d.text(x2,65,'HRS',8,bold=True,anchor='end')
    d.line(MARGIN,71,PAGE_W-MARGIN,71,INK,.6)
    sections=[('Sample construction',list(FLOW_NAMES)[:5]),('Follow-up availability',list(FLOW_NAMES)[5:9]),('Concurrent events',list(FLOW_NAMES)[9:12]),('Follow-up censoring',list(FLOW_NAMES)[12:])]
    y=87
    for title,keys in sections:
        d.text(MARGIN,y,title,8.2,bold=True);y+=16
        for key in keys:
            d.text(MARGIN+5,y,FLOW_NAMES[key],7.6);d.mapping('flow',key,FLOW_NAMES[key])
            for co,x in [('UKHLS',x1),('HRS',x2)]:
                r=next(r for r in rows if r['cohort']==co and r['label']==key)
                d.text(x,y,f"{int(float(r['estimate'])):,}",7.8,anchor='end');d.record(r,x,y)
            y+=14
        y+=8
    d.line(MARGIN,y-7,PAGE_W-MARGIN,y-7,INK,.5)
    d.wrap(MARGIN,y+6,'Availability, concurrent-event and censoring rows are overlapping subsets, not sequential exclusions. Compound episodes are retained once for sensitivity analyses. A later primary adversity censors follow-up at or after its transition.',PAGE_W-2*MARGIN,7.2,color=GRAY,leading=10)
    return d.save(source,rows)


def forest_label(label):
    if label in FOREST_LABELS:return FOREST_LABELS[label]
    for key,value in HISTORY_NAMES.items():
        for suffix,display in [(' nonoverlap','non-overlapping histories'),(' overlap_allowed','overlap allowed')]:
            if label==key+suffix:return f'{value} ({display})'
    raise KeyError(f'Unmapped forest label: {label}')


def ordered_rows(rows):
    return sorted(rows,key=lambda r:({'UKHLS':0,'HRS':1}.get(r['cohort'],2),1 if 'temporal' in r['validation'] else 0))


def panel_forest_height(g,pwidth):
    h=40
    for label,rs in group(g,'label').items():
        lines=wrapped_lines(forest_label(label),pwidth,7.5,True)
        h+=len(lines)*9+3+len(rs)*10.5+4
    return h+21


def forest(number,rows,stem,source):
    panels=group(rows,'panel');pw=(PAGE_W-2*MARGIN-GAP)/2
    panel_keys=list(panels);blocks=[];top=59
    for i in range(0,len(panel_keys),2):
        pair=panel_keys[i:i+2];height=max(panel_forest_height(panels[p],pw) for p in pair)
        for j,p in enumerate(pair):blocks.append((p,MARGIN+j*(pw+GAP),top,i+j))
        top+=height+14
    footnotes={3:[],
        4:['Robustness analyses. Current mental health is the assessment immediately before the later event.'],
        6:['UKHLS recurrent caregiving was primary; other domains and HRS analyses were secondary.'],
        7:['The UKHLS mean with non-overlapping histories was primary; other representations and HRS were secondary.']}
    extra=footnotes[number]+['Axis ranges differ between panels. Error bars show person-cluster bootstrap 95% confidence intervals.']
    h=top+len(extra)*11+4
    d=Drawing(stem,PAGE_W,h,TITLES[number]);d.text(MARGIN,22,TITLES[number],10.6,bold=True)
    d.mark(MARGIN+2,42,open=False,r=2);d.text(MARGIN+10,45,'CV: person-grouped cross-validation',7.5)
    d.mark(212,42,open=True,r=2);d.text(220,45,'T: temporal holdout',7.5)
    for panel,x,y,idx in blocks:
        g=panels[panel];a,b=PANEL_NAMES.get(panel,(panel,''))
        d.text(x,y,f'{chr(97+idx)}  {a}',8.4,bold=True)
        if b:d.text(x,y+12,b,7.6)
        d.mapping('panel',panel,'; '.join(z for z in (a,b) if z))
        head=y+25
        d.text(x,head,'Cohort / validation (n)',6.8)
        d.rich_r2(x+pw,head,prefix='\u0394',suffix=' (95% CI)',size=6.8,anchor='end')
        d.line(x,head+5,x+pw,head+5,INK,.5)
        # Reserve separate columns; headings may span the full panel width.
        label_w=70;value_w=90;left=x+label_w;right=x+pw-value_w
        if right-left<60:raise ValueError('Forest plot column is too narrow')
        lo,hi,ticks=forest_bounds(g);sx=lambda v:left+(float(v)-lo)/(hi-lo)*(right-left)
        # Draw only in row bands, so the zero line does not pass through headings.
        yy=head+15
        for label,rs in group(g,'label').items():
            disp=forest_label(label);d.mapping('comparison',label,disp)
            used=d.wrap(x,yy,disp,pw,7.5,True,leading=9);yy+=used+3
            row_top=yy-7
            for r in ordered_rows(rs):
                short='T' if 'temporal' in r['validation'] else 'CV'
                n=f"{int(float(r['n'])):,}" if finite(r.get('n')) else 'not available'
                left_label=f"{r['cohort']} {short} ({n})"
                if pdfmetrics.stringWidth(left_label,FONT,7)>label_w-4:raise ValueError(left_label)
                d.text(x+3,yy,left_label,7.0)
                py=yy-2.2
                if lo<=0<=hi:d.line(sx(0),py-5,sx(0),py+5,'#AAAAAA',.45,True)
                if finite(r.get('ci_lower')) and finite(r.get('ci_upper')):
                    l=sx(r['ci_lower']);u=sx(r['ci_upper']);d.line(l,py,u,py,'#777777',.55)
                    d.line(l,py-1.8,l,py+1.8,'#777777',.5);d.line(u,py-1.8,u,py+1.8,'#777777',.5)
                d.mark(sx(r['estimate']),py,open=short=='T',r=2)
                value=estimate_text(r)
                if pdfmetrics.stringWidth(value,FONT,7.0)>value_w-3:raise ValueError(f'Value column overflow: {value}')
                d.text(x+pw,yy,value,7.0,anchor='end')
                d.record(r,sx(r['estimate']),py,lo,hi,left,right,display_comparison=disp,display_count_label=left_label,display_value=value)
                yy+=10.5
            yy+=4
        axis(d,left,right,yy+1,lo,hi,ticks,label=None)
    for i,s in enumerate(extra):d.text(MARGIN,top+i*11,s,7.0,GRAY)
    return d.save(source,rows)


def measurement_label(label,panel):
    parts=label.split();episode='Earlier episode' if parts[0]=='earlier' else 'Later episode'
    if panel=='C Change reliability':
        outcome='acute response' if parts[1]=='acute' else 'two-year persistence'
        metric='alpha' if parts[-1]=='alpha' else 'omega total'
        return f'{episode}, {outcome}: {metric}'
    time={'pre1':'pre-event','event':'event','year2':'two-year follow-up'}[parts[1]]
    extra='' if len(parts)==2 else ': '+('alpha' if parts[2]=='alpha' else 'omega total')
    return f'{episode}, {time}{extra}'


def measurement(rows,stem,source):
    panel_order=['A Complete items','B Full scale reliability','C Change reliability']
    heights=[47+len(group([r for r in rows if r['panel']==p],'label'))*12.5+31 for p in panel_order]
    h=67+sum(heights)+11*2+31
    d=Drawing(stem,PAGE_W,h,TITLES[5]);d.text(MARGIN,22,TITLES[5],11,bold=True)
    d.mark(MARGIN+3,41,shape='circle',r=2);d.text(MARGIN+11,44,'UKHLS',7.6)
    d.mark(90,41,shape='square',open=True,r=2);d.text(99,44,'HRS',7.6)
    d.text(144,44,'Descriptive estimates; no confidence intervals were estimated.',7.4,GRAY)
    top=67
    for i,p in enumerate(panel_order):
        g=[r for r in rows if r['panel']==p]
        title=['Complete-item proportion','Full-scale reliability','Change-score reliability'][i]
        d.text(MARGIN,top,f'{chr(97+i)}  {title}',8.7,bold=True);d.mapping('panel',p,title)
        left=185;right=296;x_uk=397;x_hr=PAGE_W-MARGIN
        d.text(MARGIN,top+19,'Assessment',7.1)
        d.text(x_uk,top+19,'UKHLS: estimate (n)',7.1,anchor='end')
        d.text(x_hr,top+19,'HRS: estimate (n)',7.1,anchor='end')
        d.line(MARGIN,top+25,PAGE_W-MARGIN,top+25,INK,.5)
        yy=top+42
        for label,rs in group(g,'label').items():
            disp=measurement_label(label,p)
            if pdfmetrics.stringWidth(disp,FONT,7.1)>left-MARGIN-9:raise ValueError(f'Measurement label overflow: {disp}')
            d.text(MARGIN,yy,disp,7.1);d.mapping('measure',label,disp)
            for co,xv,dy,shape,op in [('UKHLS',x_uk,-2.5,'circle',False),('HRS',x_hr,1.8,'square',True)]:
                r=next(r for r in rs if r['cohort']==co)
                xx=left+num(r['estimate'])*(right-left);py=yy-2+dy
                d.mark(xx,py,shape=shape,open=op,r=1.8)
                value=f"{estimate_text(r,False)} ({int(float(r['n'])):,})"
                d.text(xv,yy,value,7.1,anchor='end')
                d.record(r,xx,py,0,1,left,right,display_measure=disp,display_value=value)
            yy+=12.5
        axis(d,left,right,yy+3,0,1,[0,.5,1],'Complete-item proportion' if i==0 else 'Reliability coefficient')
        top+=heights[i]+11
    d.text(MARGIN,h-14,'n is the completeness denominator or the complete-item sample used to estimate reliability.',7.2,GRAY)
    return d.save(source,rows)


def weighting(rows,stem,source):
    pw=(PAGE_W-2*MARGIN-GAP)/2
    panel_order=['UKHLS persistence_2y_z','UKHLS persistence_4y_z','HRS persistence_2y_z','HRS persistence_4y_z']
    top=70;ph=280;h=top+ph*2+49
    d=Drawing(stem,PAGE_W,h,TITLES[8]);d.text(MARGIN,22,TITLES[8],10.8,bold=True)
    d.text(MARGIN,43,'Points are descriptive estimates; confidence intervals were not estimated.',7.5,GRAY)
    lo,hi,ticks=nice_bounds(rows,False,4,True)
    for i,p in enumerate(panel_order):
        g=[r for r in rows if r['panel']==p];x=MARGIN+(i%2)*(pw+GAP);y=top+(i//2)*ph
        a,b=PANEL_NAMES[p];d.text(x,y,f'{chr(97+i)}  {a}: {b.lower()}',8.3,bold=True);d.mapping('panel',p,f'{a}: {b.lower()}')
        d.text(x,y+19,'Method (n episodes)',7.0);d.text(x+pw,y+19,'Estimate',7.0,anchor='end');d.line(x,y+24,x+pw,y+24,INK,.5)
        left=x+98;right=x+pw-38;sx=lambda v:left+(float(v)-lo)/(hi-lo)*(right-left)
        yy=y+39
        for fam in FAMILIES:
            d.text(x,yy,FAMILY_NAMES[fam],7.5,bold=True);d.mapping('event',fam,FAMILY_NAMES[fam]);yy+=13
            for method,name in METHOD_NAMES.items():
                r=next(r for r in g if r['label']==fam+' '+method)
                lab=f"{name} ({int(float(r['n'])):,})";d.text(x+4,yy,lab,7.1)
                py=yy-2.3
                if lo<=0<=hi:d.line(sx(0),py-4.5,sx(0),py+4.5,'#AAAAAA',.45,True)
                d.mark(sx(r['estimate']),py,r=2)
                val=estimate_text(r,False);d.text(x+pw,yy,val,7.1,anchor='end')
                d.record(r,sx(r['estimate']),py,lo,hi,left,right,display_measure=FAMILY_NAMES[fam]+'; '+name,display_value=val)
                d.mapping('method',method,name);yy+=11
            yy+=4
        axis(d,left,right,yy+1,lo,hi,ticks,'Persistence (pre-event SD)')
    d.text(MARGIN,h-30,'IPOW: inverse-probability-of-observation weighting; weights truncated at the 1st and 99th percentiles.',7.1,GRAY)
    d.text(MARGIN,h-17,'Complete cases have pre-event, event, two-year and four-year observations. Higher values indicate worse persistence.',7.1,GRAY)
    return d.save(source,rows)


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def validate_source_rows(root,rows,cache):
    for r in rows:
        p=(root/r['source_file']).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('Source file is outside the repository')
        if p not in cache:cache[p]=read_rows(p)
        orig=cache[p][int(r['source_row_0based'])]
        for k,v in json.loads(r['source_filter']).items():
            if str(orig[k])!=str(v):raise AssertionError((p,k,orig[k],v))
        for target,col in [('estimate',r.get('source_estimate_column','')),('n',r.get('source_n_column',''))]:
            if r.get(target) and col:
                tol=Decimal('0') if target=='n' else Decimal('1e-14')
                assert abs(Decimal(r[target])-Decimal(orig[col]))<=tol,(p,target,r[target],orig[col])
        for target,col in zip(['ci_lower','ci_upper'],filter(None,r.get('source_ci_columns','').split(';'))):
            if r.get(target):assert abs(Decimal(r[target])-Decimal(orig[col]))<=Decimal('1e-14')


def run(root,out,render_pngs=False):
    root=root.resolve();out=out.resolve();font_name=fonts()
    figdir=out/'extended_figures';figdir.mkdir(parents=True,exist_ok=True)
    cache={};audits=[];mappings=[]
    for i in range(1,9):
        key=f'Extended_Data_Figure_{i}';p=root/'source_data/extended_data'/f'{key}_source.csv'
        before=sha256(p);rows=read_rows(p);validate_source_rows(root,rows,cache)
        if i in [1,2,5,8]:f={1:trajectory,2:flow,5:measurement,8:weighting}[i];audit=f(rows,figdir/key,str(p.relative_to(root)))
        else:audit=forest(i,rows,figdir/key,str(p.relative_to(root)))
        assert sha256(p)==before
        for mapping in audit['label_mapping']:mappings.append({'figure':key,**mapping})
        audits.append({'display':key,'source_file':str(p.relative_to(root)),'source_sha256':before,'source_rows':len(rows),
                       'plotted_rows':audit['plotted_rows'],'all_source_rows_plotted':'PASS','numeric_provenance':'PASS',
                       'text_overlap':'PASS','width_pt':audit['width_pt'],'height_pt':audit['height_pt'],'font_file':font_name})
        if render_pngs:
            pdftoppm=shutil.which('pdftoppm')
            if not pdftoppm:raise RuntimeError('pdftoppm is required for --render-review-pngs')
            review=out/'review';review.mkdir(exist_ok=True)
            subprocess.run([pdftoppm,'-singlefile','-r','200','-png',str(figdir/f'{key}.pdf'),str(review/key)],check=True,capture_output=True)
    for name,records in [('extended_figure_style_audit.csv',audits),('display_label_mapping.csv',mappings)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'README_EXTENDED_FIGURES.txt').write_text('Extended Data figures were drawn from archived aggregate results. Every estimate, interval and count was checked against its source row. Fonts are embedded; PDF and SVG elements remain vector-based. All figures are 180 mm wide.\n',encoding='utf-8')
    print(json.dumps(audits,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--archive',type=Path,help='Use the unchanged FINAL_no_git.zip as the data source')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--render-review-pngs',action='store_true')
    args=parser.parse_args()
    if args.archive:
        with tempfile.TemporaryDirectory(prefix='nmh_aggregate_') as tmp:
            base=Path(tmp)
            with zipfile.ZipFile(args.archive) as z:
                for entry in z.infolist():
                    target=(base/entry.filename).resolve()
                    if not target.is_relative_to(base.resolve()):raise ValueError('Unsafe archive entry')
                z.extractall(base)
            matches=list(base.glob('*/source_data/extended_data'))
            if len(matches)!=1:raise ValueError('Could not uniquely locate aggregate source data')
            run(matches[0].parents[1],args.output_dir,args.render_review_pngs)
    else:run(args.source_root,args.output_dir,args.render_review_pngs)

if __name__=='__main__':main()
