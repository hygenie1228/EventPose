"""Redraw the current model with Figure-3-inspired modular visual styling."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools')]
from goal_runtime_control import register_job,write

def main(english=False,mesh_path=None,event_image_path=None,export_callback=None):
    import numpy as np
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Polygon,Circle
    g=json.loads((ROOT/'_agents/research/261028/device3_encoding_goal_12h.json').read_text());register_job(g,os.getpid(),'stereo_paper_style_figure_CPU')
    baseout=Path(g['output_root'])/'encoding_explanation'
    out=baseout/'editable' if english else baseout
    out.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf')
    plt.rcParams.update({'font.family':'UnDotum','axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    receipt=json.loads((baseout/'설명_그림_검증자료.json').read_text());name=receipt['session'];frame=receipt['source_frame']
    if english and (out/'actual_assets_provenance.json').exists():
        actual=json.loads((out/'actual_assets_provenance.json').read_text());name=actual['session'];frame=actual['source_frame']
    with h5py.File(Path(g['output_root'])/'encoding_cache/surface20'/(name+'.h5'),'r',locking=False) as f:
        i=int(np.searchsorted(f['frames'][:],frame));assert int(f['frames'][i])==frame
        rgb=f['rgb3'][i].transpose(1,2,0);events=f['event4'][i].astype(float)/255
    ev=np.ones((224,224,3))*.98
    ev[:,:,0]-=np.minimum(events[1]*2.5,.9);ev[:,:,1]-=np.minimum((events[0]+events[1])*2,.9);ev[:,:,2]-=np.minimum(events[0]*2.5,.9)
    if event_image_path is not None:
        from PIL import Image
        import cv2
        raw=np.asarray(Image.open(event_image_path).convert('RGB'));eh,ew=raw.shape[:2];scale=min(224/ew,224/eh);nw,nh=round(ew*scale),round(eh*scale)
        ev=np.zeros((224,224,3),np.uint8);left,top=(224-nw)//2,(224-nh)//2
        ev[top:top+nh,left:left+nw]=cv2.resize(raw,(nw,nh),interpolation=cv2.INTER_AREA)
    translations={
        '보정 기반 RGB–Event 스테레오 인체 복원':'Calibration-aware RGB–Event Stereo Human Reconstruction',
        'RGB 특징 F':'RGB features F','Event 특징 E':'Event features E','3종 인코딩':'Event encoding',
        '동일 시각의 과거 관측':'Causal observations at the same time',
        '각 센서의 고유 좌표와 보정값 유지':'Sensor-specific coordinates and calibration',
        '① RGB query와 광선':'1. RGB queries and rays','평균 pooling':'Average pooling',
        '→ Linear\n128차원 Q':'Linear\n128-D queries Q',
        '256 ray × 깊이 후보 11개':'256 rays × 11 depth hypotheses',
        'Z = 0.5 … 8 m\n센서 보정 Kp, Ke, R, t':'Z = 0.5 … 8 m\nCalibration: Kp, Ke, R, t',
        '② Event 특징 표본 추출':'2. Event feature sampling','보정값으로 투영':'Calibrated projection',
        '28 × 28 위에서 sampling':'Sample on the 28 × 28 feature map',
        '화면 밖·뒤쪽 후보 제외':'Mask out-of-view / behind-camera points',
        '③ 깊이별 가중합·잔차':'3. Depth attention and residual',
        'Masked softmax\n11후보 가중합':'Masked softmax\nWeighted sum over 11 depths',
        '메시지':'Message','Linear + 확대':'Linear + upsampling',
        '원래 RGB 특징 F 유지: 64 × 64 × 1024':'Preserve the original RGB features F: 64 × 64 × 1024',
        '검출\n+ HPH':'Detection\n+ HPH','자세·체형·거리':'Pose, shape and distance',
        '3D mesh · root\n55관절 → body22 평가':'3D mesh and root\n55 joints → body22 evaluation',
        '공유 구조':'Variants',
        'event3 + event4: 주센서만 event adapter 경로로 교체':'event3 + event4: event adapter before the primary ViT',
        'RGB3 + event3: 보조 센서만 교체':'RGB3 + event3: replace the auxiliary sensor',
        '실제 RGB/event 입력 예시 · 특징 격자/ray/mesh 아이콘은 구조 개념도 · 배치 차원 B 생략 · GT box/pose/depth는 추론 입력 아님':'Actual RGB/event inputs and predicted mesh; feature/ray graphics are schematic. Batch B omitted. No GT box/pose/depth input.'}
    fig,ax=plt.subplots(figsize=(29,12));fig.subplots_adjust(left=.008,right=.992,bottom=.02,top=.96)
    ax.set(xlim=(0,176),ylim=(0,76));ax.axis('off');ink='#26333d';muted='#62717b'
    purple='#ae6bd7';blue='#5998cf';green='#75b98b';orange='#db9865'
    def text(x,y,s,size=13,ha='center',**kw):
        if english:
            import re
            s=translations.get(s,s);assert not re.search('[가-힣]',s),s
            if len(s)>36 and size<=16:size*=.82
        return ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=kw.pop('color',ink),**kw)
    def arrow(a,b,color=ink,ls='-',lw=1.5):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=lw,color=color,linestyle=ls,zorder=12))
    def path(points,color=ink,ls='-',lw=1.4):
        ax.plot(*zip(*points),color=color,ls=ls,lw=lw,zorder=10)
        arrow(points[-2],points[-1],color,ls,lw)
    def panel(x,y,w,h,title,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.2,rounding_size=1.2',fc=color+'13',ec=color,lw=1.4,zorder=0))
        text(x+.5,y+h+2,title,16,ha='left',color=ink)
    def box(x,y,w,h,label,color,fs=13):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.14,rounding_size=.65',fc=color+'28',ec=color,lw=1.2,zorder=4));text(x+w/2,y+h/2,label,fs,zorder=6)
    def stack(x,y,w,h,color,label,sub,grid=False):
        for k in (2,1,0):
            ax.add_patch(Rectangle((x+k*.65,y+k*.65),w,h,fc=color+'45',ec=color,lw=1,zorder=3))
        if grid:
            for t in np.linspace(0,1,6)[1:-1]:
                ax.plot([x+t*w,x+t*w],[y,y+h],color=color,lw=.55,zorder=4)
                ax.plot([x,x+w],[y+t*h,y+t*h],color=color,lw=.55,zorder=4)
        text(x+w/2,y+h+3,label,13);text(x+w/2,y-3,sub,12,color=muted)
    def encoder(x,y,w,h,color,title):
        for k in range(3):
            ax.add_patch(FancyBboxPatch((x+k*w*.3,y+k*.8),w*.18,h-k*1.6,boxstyle='round,pad=.06,rounding_size=.35',fc=color+'45',ec=color,lw=1))
        text(x+w*.4,y-3,title,12)
    # Main pathway and true sensor input thumbnails.
    text(1,74,'보정 기반 RGB–Event 스테레오 인체 복원',22,ha='left')
    ax.imshow(rgb,extent=(1,13,52,64),zorder=3);text(7,67,'device3 RGB',14);text(7,49,'3 × 896 × 896',12,color=muted)
    ax.imshow(ev,extent=(1,13,22,34),zorder=3);text(7,37,'device4 event',14);text(7,19,'동일 시각의 과거 관측',11,color=muted)
    encoder(18,53,9,10,blue,'ViT-L/14');arrow((13,58),(18,58));arrow((26,58),(31,58))
    stack(31,54,8,8,blue,'RGB 특징 F','64 × 64 × 1024',True)
    box(16,24,12,8,'3종 인코딩',purple,12);arrow((13,28),(16,28));arrow((28,28),(31,28))
    encoder(31,23,7,10,purple,'Event CNN');arrow((38,28),(41,28))
    stack(41,24,7,8,purple,'Event 특징 E','28 × 28 × 128',True)
    text(23,14,'count33: 2ch   ·   bins100: 6ch   ·   surface20: 2ch',11)
    text(23,9,'각 센서의 고유 좌표와 보정값 유지',12,color=muted)
    # Three clearly separated fusion submodules, as in a journal overview.
    panel(53,20,25,45,'① RGB query와 광선',purple)
    stack(57,47,7,7,purple,'평균 pooling','16 × 16 × 1024',True)
    path([(40,58),(49,58),(49,51),(57,51)],blue)
    text(70,51,'→ Linear\n128차원 Q',11)
    arrow((65,51),(67,51),purple)
    # One schematic ray only; label states exact full grid count.
    ax.add_patch(Polygon([[57,32],[60,30],[60,34]],closed=True,fc=purple+'44',ec=purple,lw=1.2))
    ax.plot([60,74],[32,39],color=purple,lw=1.1)
    for t in np.linspace(.12,.95,11):ax.add_patch(Circle((60+14*t,32+7*t),.38,fc=purple,ec='white',lw=.3,zorder=8))
    text(66,43,'256 ray × 깊이 후보 11개',11)
    text(65.5,26,'Z = 0.5 … 8 m\n센서 보정 Kp, Ke, R, t',11)
    arrow((65.5,28.5),(65.5,33.8),purple,ls='--',lw=1)
    # Event feature lattice + projected candidates; does not originate rays.
    panel(81,20,23,45,'② Event 특징 표본 추출',blue)
    stack(85,43,13,13,blue,'보정값으로 투영','28 × 28 위에서 sampling',True)
    us=np.linspace(85.7,97.4,11);vs=49+2*np.sin(np.linspace(-1,1.2,11))
    ax.plot(us,vs,color=purple,lw=1.1,zorder=7);ax.scatter(us,vs,s=10,c=purple,zorder=8)
    arrow((76,36),(85,47),purple)
    path([(49,28),(50,28),(50,17),(92.5,17),(92.5,23)],blue)
    text(92.5,34,'화면 밖·뒤쪽 후보 제외',11)
    box(84,23,17,6,'Bilinear sampling',blue,11)
    path([(99,46),(102,46),(102,30),(92.5,30),(92.5,29)],blue)
    # Attention across candidates, followed by a residual at original size.
    panel(107,20,23,45,'③ 깊이별 가중합·잔차',green)
    box(110,48,17,10,'Masked softmax\n11후보 가중합',green,12)
    path([(101,26),(105,26),(105,53),(110,53)],blue)
    path([(74,52),(76,52),(76,63),(118.5,63),(118.5,58)],purple,ls='--')
    stack(112,34,9,7,green,'메시지','16 × 16 × 128')
    arrow((118.5,48),(118.5,42),green)
    box(110,23,17,6,'Linear + 확대',green,11);arrow((117,33),(118.5,29),green)
    # Original-resolution skip bypass. Only a residual is upsampled.
    path([(39,60),(46,60),(46,71),(135,71),(135,58)],blue)
    text(84,72,'원래 RGB 특징 F 유지: 64 × 64 × 1024',12,color=blue)
    path([(127,26),(135,26),(135,54)],green);text(141,35,'ΔF\n64 × 64 × 1024',11,color=green)
    ax.add_patch(Circle((135,56),1.6,fc='white',ec=ink,lw=1.3,zorder=15));text(135,56,'+',20,zorder=16)
    arrow((137,56),(141,56));box(141,49,12,14,'검출\n+ HPH',orange,15)
    arrow((153,56),(157,56));box(157,50,14,12,'SMPL-X',green,14)
    text(147,44,'자세·체형·거리',11,color=muted)
    # Human outline denotes output type; it is intentionally not an inferred mesh.
    joints=np.array([[164,39],[164,36],[160.5,35],[167.5,35],[159,30],[169,30],[158,26],[170,26],[162,29],[166,29],[161,24],[167,24],[160.5,20],[167.5,20]])
    edges=[(0,1),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),(1,8),(1,9),(8,9),(8,10),(9,11),(10,12),(11,13),(2,8),(3,9),(2,9),(3,8)]
    if mesh_path is not None:
        from PIL import Image
        mesh_image=np.asarray(Image.open(mesh_path));mh,mw=mesh_image.shape[:2];scale=min(13/mw,25/mh);dw,dh=mw*scale,mh*scale
        ax.imshow(mesh_image,extent=(164.5-dw/2,164.5+dw/2,31-dh/2,31+dh/2),zorder=4)
    else:
        for a,b in edges:ax.plot(joints[[a,b],0],joints[[a,b],1],color='#8f79b5',lw=2)
        ax.add_patch(Circle((164,40.7),1.5,fc='#c9b9e0',ec='#8f79b5',lw=1.1))
    arrow((164,50),(164,43))
    text(164,16,'3D mesh · root\n55관절 → body22 평가',11)
    # Shared-structure variants and a compact legend outside the network.
    ax.plot([53,171],[13,13],color='#d8dee5',lw=.8)
    text(54,9,'공유 구조',12,ha='left',color=muted)
    text(82,9,'event3 + event4: 주센서만 event adapter 경로로 교체',11)
    text(136,9,'RGB3 + event3: 보조 센서만 교체',11)
    text(1,3,'실제 RGB/event 입력 예시 · 특징 격자/ray/mesh 아이콘은 구조 개념도 · 배치 차원 B 생략 · GT box/pose/depth는 추론 입력 아님',10,ha='left',color=muted)
    stem='Stereo_RGB_Event_Pipeline' if english else '스테레오_모델_논문형_전체구조'
    for ext in ('png','svg','pdf'):fig.savefig(out/(stem+'.'+ext),dpi=170,bbox_inches='tight')
    if export_callback is not None:export_callback(fig,ax,out)
    plt.close(fig)
    if english:return
    report=ROOT/'_agents/report/261028_스테레오_센서융합_모델_개요와_입출력.md'
    text_report=report.read_text().replace('스테레오_모델_입출력_개요.',stem+'.')
    start='발표용 원본:';a=text_report.index(start);b=text_report.index('\n\n## 1.',a)
    text_report=text_report[:a]+f'''발표용 원본: [PNG]({out}/{stem}.png), [SVG]({out}/{stem}.svg), [PDF]({out}/{stem}.pdf).

스타일은 사용자 지정 [논문 Figure 3](https://arxiv.org/pdf/2604.09142)의 입력 thumbnail·특징 stack·옅은 색 모듈 구획·모듈 내부의 작은 연산 도식을 참고했다. 내용과 연결은 현재 실행 중인 Multi-HMR RGB/event 모델의 실제 코드로 다시 구성했다. 입력 thumbnail은 현재 캐시의 같은 시각 RGB3/event4이며, 중간 특징·ray·mesh 아이콘은 구조 개념도다. 새 checkpoint의 실제 activation이나 mesh 예측 결과를 표시한 그림은 아니다.

그림의 **① RGB query와 광선**에서 원래 RGB64×64 특징을16×16으로 평균 pooling한다. `grid=16`은 이 실험에서 계산량을 제한하도록 정한 설정값이다. 주센서 이미지896×896 위의16×16 격자 중심에서 ray256개를 만들고 각각 깊이 후보11개를 놓는다. **② Event 특징 표본 추출**은 이3D 후보점을 보정값으로 event 카메라에 투영해28×28 특징에서 읽는 단계다. event28×28의 각 칸에서 ray를 쏘는 구조가 아니다. **③ 깊이별 가중합·잔차**는11후보를 가중합한16×16 메시지를64×64 잔차로 확대해 원래 RGB 특징에 더한다. skip 경로는 원래64×64 해상도를 유지한다.

주센서가 event3인 조건에서는 event adapter의3채널 표현이 ViT로 들어간다. 보조 event encoder와 뒤쪽 융합·검출·HPH·SMPL-X 경로는 같은 구조다. 두 encoder의 특징 해상도가 같다는 가정은 사용하지 않는다.''' +text_report[b:]
    report.write_text(text_report)
    write(out/'논문형_개요도_출처와_검증.json',dict(reference='https://arxiv.org/pdf/2604.09142',reference_figure=3,reference_pdf_page=4,
        style_only=True,model_content='current Multi-HMR calibrated epipolar event residual fusion',
        actual_input=dict(session=name,source_frame=frame,encoding='surface20',device=4),
        RGB_feature_shape=[64,64,1024],event_feature_shape=[28,28,128],query_grid=[16,16],depth_candidates=11,
        original_RGB_skip_preserved=True,intermediate_features_and_mesh_are_schematic=True,
        paths={ext:str(out/(stem+'.'+ext)) for ext in ('png','svg','pdf')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(report,flush=True)

if __name__=='__main__':main()
