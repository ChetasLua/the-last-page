"""python tools/master.py [--crf 12] [--jobs 8] -> dist/the-last-page-16x9.mp4
First-generation 16:9 master: every frame rendered from source and piped straight into one H.264 encode, soundtrack muxed."""
import os, sys, time, argparse, importlib, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
for d in ('cache', 'out', 'dist'): os.makedirs(d, exist_ok=True)
import numpy as np, cv2
from multiprocessing import Pool
from lp import comp, W, H
FF = os.path.join(ROOT, '.mm/env/bin/ffmpeg'); S = None


def init(name):
    global S
    cv2.setNumThreads(1); S = importlib.import_module('shots.' + name).Shot(); S.setup()


def one(f):
    return comp.to8(comp.finish(S.frame(f), f, **getattr(S, 'finish', {})), f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--crf', default='12'); ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--out', default='dist/the-last-page-16x9.mp4')
    a = ap.parse_args(); t0 = time.time(); total = 0
    cmd = [FF, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '24', '-i', '-', '-i', 'out/soundtrack.wav', '-map', '0:v', '-map', '1:a',
           '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p,setsar=1',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', a.crf, '-profile:v', 'high', '-level', '5.1', '-g', '48', '-x264-params', 'aq-mode=3:deblock=-2,-2:psy-rd=1.0,0.2',
           '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(1, 10):
        name = f's{k}'; s = importlib.import_module('shots.' + name).Shot(); s.setup(); n = s.n; del s; t = time.time()
        with Pool(a.jobs, init, (name,)) as pool:
            for im in pool.imap(one, range(n), chunksize=2): p.stdin.write(im.tobytes()); total += 1
        print(name, n, 'frames %.0fs' % (time.time() - t), flush=True)
    p.stdin.close(); rc = p.wait()
    print('DONE' if rc == 0 and total == 1560 else 'FAILED', total, 'frames', '%.0fs' % (time.time() - t0), '%.1f MB' % (os.path.getsize(a.out) / 1e6))
