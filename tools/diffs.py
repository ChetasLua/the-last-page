"""frame-to-frame difference per shot: finds pops and dead holds"""
import sys, subprocess, numpy as np
FF='.mm/env/bin/ffmpeg'
for s in sys.argv[1:]:
    w,h=256,144
    p=subprocess.Popen([FF,'-v','error','-i',f'out/{s}.mkv','-vf',f'scale={w}:{h}:flags=area','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    a=np.frombuffer(p.stdout.read(),np.uint8).reshape(-1,h,w,3).astype(np.float32)
    d=np.abs(np.diff(a,axis=0)).mean((1,2,3))
    big=np.argsort(d)[::-1][:8]
    print(s,len(a),'mean %.2f'%d.mean(),'top:',sorted([(int(i+1),round(float(d[i]),1)) for i in big]),'lum %.0f..%.0f'%(a.mean((1,2,3)).min(),a.mean((1,2,3)).max()))
