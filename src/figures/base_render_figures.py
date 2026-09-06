"""Shared drawing functions for the aggregate-data figure scripts."""
from pathlib import Path
import argparse,json,math,html,subprocess,os,textwrap
import pandas as pd
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth

PKG=Path(__file__).resolve().parents[2]
RENDER_PNGS=False
COLORS={'UKHLS':'#17496B','HRS':'#B45E27'}
BLACK='#20252B';GRAY='#7E8994';LIGHT='#DBE1E6'
class Dual:
    def __init__(self,stem,w,h):
        self.w=w;self.h=h;self.stem=stem;self.elements=[]
        self.c=canvas.Canvas(str(stem.with_suffix('.pdf')),pagesize=(w,h),invariant=1)
        self.c.setTitle(stem.name.replace('_',' '));self.c.setAuthor('Huiyun Yu and Bo Li');self.c.setSubject('Aggregate source-data figure for the accompanying manuscript.');self.c.setCreator('Aggregate source-data figure renderer')
        self.rect(0,0,w,h,'#FFFFFF')
    def line(self,x1,y1,x2,y2,color=LIGHT,width=.6,dash=False):
        c=self.c;c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setDash([3,2] if dash else [])
        c.line(x1,self.h-y1,x2,self.h-y2)
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="3,2"' if dash else '')+'/>')
    def rect(self,x,y,w,h,fill,stroke=None):
        self.c.setFillColor(HexColor(fill));self.c.setStrokeColor(HexColor(stroke or fill));self.c.rect(x,self.h-y-h,w,h,stroke=bool(stroke),fill=1)
        self.elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke or fill}"/>')
    def text(self,x,y,s,size=9,color=BLACK,bold=False,anchor='start'):
        s=str(s).replace('−','-').replace('≥','>=')
        fn='Helvetica-Bold' if bold else 'Helvetica';c=self.c;c.setFillColor(HexColor(color));c.setFont(fn,size)
        {'start':c.drawString,'middle':c.drawCentredString,'end':c.drawRightString}[anchor](x,self.h-y,s)
        self.elements.append(f'<text x="{x}" y="{y}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{html.escape(s)}</text>')
    def wrap(self,x,y,s,width,size=9,bold=False):
        words=str(s).split();lines=[];line=''
        for word in words:
            if stringWidth((line+' '+word).strip(),'Helvetica-Bold' if bold else 'Helvetica',size)>width and line:lines.append(line);line=word
            else:line=(line+' '+word).strip()
        if line:lines.append(line)
        for j,l in enumerate(lines):self.text(x,y+j*(size+3),l,size,bold=bold)
        return len(lines)*(size+3)
    def point(self,x,y,color,open=False,r=2.5):
        c=self.c;c.setDash([]);c.setLineWidth(1.1);c.setStrokeColor(HexColor(color));c.setFillColor(HexColor('#FFFFFF' if open else color));c.circle(x,self.h-y,r,stroke=1,fill=1)
        self.elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{color}" stroke-width="1.1" fill="{"#FFFFFF" if open else color}"/>')
    def save(self,source,points):
        self.c.showPage();self.c.save()
        meta=html.escape(json.dumps({'source_csv':source,'plotted_source_rows':points},ensure_ascii=False,allow_nan=False))
        self.stem.with_suffix('.svg').write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" viewBox="0 0 {self.w} {self.h}"><metadata>{meta}</metadata>'+''.join(self.elements)+'</svg>',
            encoding='utf-8',
        )
        if RENDER_PNGS:
            cmd=['pdftoppm','-singlefile','-r','300','-png',str(self.stem.with_suffix('.pdf')),str(self.stem)]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def finite(x):return pd.notna(x) and math.isfinite(float(x))
def axis(d,x,y,w,lo,hi,label,gridtop=None):
    d.line(x,y,x+w,y,GRAY)
    for a in np.linspace(lo,hi,5):
        px=x+(a-lo)/(hi-lo)*w
        d.line(px,y,px,y+3,GRAY)
        prec=3 if hi-lo<.02 else 2
        d.text(px,y+14,f'{0 if abs(a)<.5*10**(-prec) else a:.{prec}f}',8.5,anchor='middle')
    if lo<=0<=hi:
        zx=x-lo/(hi-lo)*w;d.line(zx,gridtop or y-10,zx,y,GRAY,.7,True)
    d.text(x+w/2,y+29,label,8.5,anchor='middle')
def bounds(g):
    vals=[float(v) for c in ['estimate','ci_lower','ci_upper'] for v in g[c] if finite(v)]
    lo=min([0,*vals]);hi=max([0,*vals]);span=max(hi-lo,.005)
    return lo-.08*span,hi+.08*span
def panel_height(g):
    return 52+len(g)*14+sum(18+12*(len(str(label))>39) for label in g.label.unique())

def forest(key,meta,df,stem):
    panels=list(df.panel.unique());ncol=2 if len(panels)>2 or (key.startswith('Extended') and len(panels)>1) else 1
    pw=350 if ncol==2 else 560;w=pw*ncol+40;tops=[];y=86
    for start in range(0,len(panels),ncol):
        row=panels[start:start+ncol];rh=max(panel_height(df.loc[df.panel.eq(p)]) for p in row)
        tops.extend((p,20+j*pw,y,pw-18,rh) for j,p in enumerate(row));y+=rh+28
    d=Dual(stem,w,y+36);d.wrap(20,26,meta['title'],w-40,14,True)
    d.text(20,51,'UKHLS = blue   HRS = orange   filled = grouped CV   open = temporal holdout',9)
    d.text(20,66,'P = primary   S = secondary or sensitivity   error bars = 95% CI',8.5)
    plotted=[]
    for panel,x,top,pwidth,h in tops:
        g=df.loc[df.panel.eq(panel)];d.wrap(x,top,panel,pwidth,11,True);yy=top+20
        ax=x+(157 if ncol==2 else 270);aw=pwidth-(165 if ncol==2 else 280);lo,hi=bounds(g)
        for label in g.label.unique():
            gg=g.loc[g.label.eq(label)].sort_values(['cohort','validation'],ascending=[False,True])
            if key=='Extended_Data_Figure_7' and (gg.analysis_role=='primary').any():
                d.rect(x,yy-11,pwidth,20+len(gg)*14,'#EDF3F7',LIGHT)
            yy+=d.wrap(x,yy,label,pwidth,9,True)+4
            for _,r in gg.iterrows():
                co=r.cohort;validation=str(r.validation);temporal='temporal' in validation
                short='T' if temporal else 'CV' if 'grouped' in validation else ''
                role='P' if r.analysis_role=='primary' else 'S' if any(t in str(r.analysis_role) for t in ['secondary','sensitivity']) else ''
                n='' if not finite(r.n) else f'n={int(r.n):,}'
                d.text(x+6,yy,f'{co} {short} {role}  {n}',8.5,COLORS.get(co,BLACK))
                if finite(r.estimate):
                    sx=lambda v:ax+(float(v)-lo)/(hi-lo)*aw
                    if finite(r.ci_lower) and finite(r.ci_upper):
                        a,b=sx(r.ci_lower),sx(r.ci_upper);d.line(a,yy-3,b,yy-3,COLORS.get(co,BLACK),1)
                        d.line(a,yy-5,a,yy-1,COLORS.get(co,BLACK),.7);d.line(b,yy-5,b,yy-1,COLORS.get(co,BLACK),.7)
                    d.point(sx(r.estimate),yy-3,COLORS.get(co,BLACK),temporal,3.5 if r.analysis_role=='primary' else 2.6)
                    plotted.append(int(r.display_row))
                else:d.text(ax,yy,'not available',8.5,GRAY)
                yy+=14
        axis(d,ax,yy+3,aw,lo,hi,'Delta predictive R2' if 'predictive' in str(g.unit.iloc[0]) else ('Proportion' if 'proportion' in str(g.unit.iloc[0]) else 'Coefficient'),top+20)
    d.save(meta['source'],plotted)

def descriptive(key,meta,df,stem):
    panels=list(df.panel.unique());w=740;pw=350;tops=[];y=68
    for start in range(0,len(panels),2):
        row=panels[start:start+2];rh=max(85+24*df.loc[df.panel.eq(p),'label'].nunique() for p in row)
        tops.extend((p,20+j*pw,y,rh) for j,p in enumerate(row));y+=rh+26
    d=Dual(stem,w,y+25);d.wrap(20,27,meta['title'],w-40,14,True);d.text(20,49,'UKHLS = blue   HRS = orange   Points show descriptive estimates; confidence intervals were not estimated.',9)
    pts=[]
    for panel,x,top,rh in tops:
        g=df.loc[df.panel.eq(panel)];d.text(x,top,panel,11,bold=True);yy=top+27;ax=x+192;aw=128;lo,hi=bounds(g)
        for label,gg in g.groupby('label',sort=False):
            short=label.replace('unweighted','Unweighted').replace('IPOW_truncated_1_99','IPOW').replace('complete_four_points','Complete case').replace('omega_total','omega')
            d.wrap(x,yy,short,180,8.5)
            for j,(_,r) in enumerate(gg.iterrows()):
                if finite(r.estimate):d.point(ax+(r.estimate-lo)/(hi-lo)*aw,yy-3+(j-(len(gg)-1)/2)*5,COLORS.get(r.cohort,BLACK),r.cohort=='HRS',2.4);pts.append(int(r.display_row))
            yy+=24
        label='Proportion' if 'proportion' in str(g.unit.iloc[0]) else 'Reliability' if 'reliability' in str(g.unit.iloc[0]) else 'Persistence in cohort SD'
        axis(d,ax,yy,aw,lo,hi,label,top+20)
    d.save(meta['source'],pts)

def trajectory(key,meta,df,stem):
    d=Dual(stem,590,615);d.text(24,28,meta['title'],14,bold=True)
    times=['pre2','pre1','event','year2','year4'];families=list(df.label.unique());colors=['#17496B','#B45E27','#357862','#81569E'];points=[]
    for k,co in enumerate(['UKHLS','HRS']):
        top=75+k*265;g=df.loc[df.cohort.eq(co)];d.text(24,top-10,co,12,bold=True)
        d.text(88,top-10,f'{int(g.n.iloc[0]):,} people; {int(g.n_episodes.iloc[0]):,} episodes',9)
        left=65;right=540;bottom=top+188;lo,hi=bounds(g);xx=lambda j:left+j*(right-left)/4;yy=lambda v:bottom-(v-lo)/(hi-lo)*173
        for tick in np.linspace(lo,hi,5):d.line(left,yy(tick),right,yy(tick));d.text(left-8,yy(tick)+3,f'{tick:.2f}',9,anchor='end')
        for fi,fam in enumerate(families):
            a=g.loc[g.label.eq(fam)].set_index('time_point');prev=None
            for j,t in enumerate(times):
                if t not in a.index:continue
                r=a.loc[t];x=xx(j)+(-3+2*fi);y=yy(r.estimate)
                if prev:d.line(*prev,x,y,colors[fi],1.3)
                if finite(r.ci_lower):d.line(x,yy(r.ci_lower),x,yy(r.ci_upper),colors[fi],.8)
                d.point(x,y,colors[fi],False,2.5);prev=(x,y);points.append(int(r.display_row))
        for j,t in enumerate(['pre2','pre1','event','2 years','4 years']):d.text(xx(j),bottom+18,t,9,anchor='middle')
        d.text(65,top+7,'Standardized outcome',9)
    for j,fam in enumerate(families):d.point(42+j*137,49,colors[j]);d.text(50+j*137,52,fam,9)
    d.text(24,592,'Higher scores indicate worse mental health. Time points are categories, not equal elapsed intervals.',8.5,GRAY)
    d.save(meta['source'],points)

def design(key,meta,df,stem):
    d=Dual(stem,590,565);d.text(24,29,meta['title'],14,bold=True)
    d.text(24,48,'Prospective information structure, not a causal diagram',9,GRAY);points=[]
    top=88
    instructions={
        'Trajectory cohort':['Unique transition','Isolated primary episode','Event-centred observations'],
        'A Cross domain':['Earlier different family','Earlier two-year outcome observed','Later target response'],
        'B Recurrence':['Earlier caregiving episode','Documented no-care reset','Later caregiving response'],
        'C Multiple histories':['At least two acute histories','All known before target pre1','Earliest eligible target']}
    for panel,g in df.groupby('panel',sort=False):
        d.text(24,top,panel,11,bold=True)
        for j,s in enumerate(instructions[panel]):
            d.rect(24+j*185,top+13,172,42,'#F1F4F6',LIGHT);d.wrap(33+j*185,top+29,s,155,9)
            if j<2:d.line(197+j*185,top+34,207+j*185,top+34,GRAY,1.2)
        y=top+74
        for co in ['UKHLS','HRS']:
            ss=g.loc[g.cohort.eq(co)];s='; '.join(f'{r.label}: {int(r.n):,}' for _,r in ss.iterrows())
            d.text(24,y,co,9,COLORS[co],True);d.wrap(74,y,s,490,9)
            y+=16;points.extend(ss.display_row.astype(int).tolist())
        top+=119
    d.save(meta['source'],points)
def flow(key,meta,df,stem):
    d=Dual(stem,680,450);d.text(24,28,meta['title'],14,bold=True)
    labels={'all_collapsed_candidate_episodes':'Collapsed candidate episodes','any_primary_episode':'Contains a primary event','isolated_primary_episode':'Isolated primary event','valid_pre1_and_event_self_report':'Valid pre1 and event self-report','valid_age_and_binary_sex':'Final trajectory sample','has_pre2':'Has additional pre2','has_uncensored_year2':'Observed uncensored year2','has_uncensored_year4':'Observed uncensored year4','complete_four_timepoints':'Complete pre1 event year2 year4','isolated_primary':'Isolated primary audit','compound_primary':'Compound primary audit'}
    points=[]
    for k,co in enumerate(['UKHLS','HRS']):
        x=24+k*336;d.text(x,61,co,12,bold=True);g=df.loc[df.cohort.eq(co)];y=84
        for _,r in g.iterrows():
            d.text(x,y,labels.get(r.label,r.label.replace('_',' ')),8.5)
            d.text(x+299,y,f'{int(r.estimate):,}',9,COLORS[co],True,anchor='end');points.append(int(r.display_row));y+=22
    d.wrap(24,406,'Follow-up availability rows are overlapping subsets, not sequential exclusions. Compound episodes are retained once for sensitivity only. A later primary adversity censors follow-up at or after its transition date.',625,9)
    d.save(meta['source'],points)

def main():
    raise SystemExit("Use reproduce_extended_data_figures.py to generate the Extended Data figures.")


if __name__ == '__main__':
    main()
