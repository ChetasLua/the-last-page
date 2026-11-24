"""Concatenate out/s1..s9.mkv, mux out/soundtrack.wav, write out/film.mp4 (delivery) and dist/the-last-page.html"""
import os, sys, subprocess, base64, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
FF = '.mm/env/bin/ffmpeg'
ap = argparse.ArgumentParser(); ap.add_argument('--rate', default='8200k'); ap.add_argument('--preset', default='slow'); ap.add_argument('--html', action='store_true'); ap.add_argument('--skip', action='store_true')
a = ap.parse_args()
os.makedirs('dist', exist_ok=True)
if not a.skip:
    with open('out/list.txt', 'w') as fh:
        for i in range(1, 10): fh.write(f"file 's{i}.mkv'\n")
    vf = 'scale=in_range=tv:in_color_matrix=bt601:out_range=tv:out_color_matrix=bt709:flags=lanczos+accurate_rnd,format=yuv420p'
    has = os.path.exists('out/soundtrack.wav')
    common = ['-c:v', 'libx264', '-preset', a.preset, '-b:v', a.rate, '-maxrate', '17M', '-bufsize', '26M', '-profile:v', 'high', '-level', '5.1', '-pix_fmt', 'yuv420p', '-g', '48',
              '-x264-params', 'aq-mode=3:aq-strength=1.0:deblock=-1,-1:psy-rd=1.0,0.15', '-vf', vf, '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-r', '24']
    src = [FF, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', 'out/list.txt']
    subprocess.run(src + ['-map', '0:v'] + common + ['-pass', '1', '-passlogfile', 'out/p', '-an', '-f', 'null', '/dev/null'], check=True)
    cmd = src + (['-i', 'out/soundtrack.wav'] if has else []) + ['-map', '0:v'] + (['-map', '1:a'] if has else []) + common + ['-pass', '2', '-passlogfile', 'out/p', '-movflags', '+faststart']
    if has: cmd += ['-c:a', 'aac', '-b:a', '256k', '-shortest']
    subprocess.run(cmd + ['out/film.mp4'], check=True)
    print('out/film.mp4', os.path.getsize('out/film.mp4') / 1e6, 'MB')
if a.html:
    b = base64.b64encode(open('out/film.mp4', 'rb').read()).decode()
    h = open('tools/page.html').read().replace('/*FILM*/', b)
    open('dist/the-last-page.html', 'w').write(h)
    print('dist/the-last-page.html', len(h) / 1e6, 'MB')
