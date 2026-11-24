// node tools/playtest.mjs : opens dist/the-last-page.html in headless Chrome, clicks play, samples playback, screenshots, seeks to end, quits.
import {spawn} from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const ROOT=path.resolve(path.dirname(new URL(import.meta.url).pathname),'..');
const CH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const ud=fs.mkdtempSync(path.join(os.tmpdir(),'tlp-playtest-'));
const p=spawn(CH,['--headless=new','--remote-debugging-port=0','--user-data-dir='+ud,'--no-first-run','--window-size=1600,900','--use-angle=metal','file://'+ROOT+'/dist/the-last-page.html'],{stdio:['ignore','ignore','pipe']});
let buf=''; const ws=await new Promise((res,rej)=>{p.stderr.on('data',d=>{buf+=d;const m=buf.match(/ws:\/\/[^\s]+/);if(m)res(m[0])});setTimeout(()=>rej(new Error('no devtools '+buf.slice(0,400))),20000)});
const port=new URL(ws).port; const sleep=t=>new Promise(r=>setTimeout(r,t));
let tg; for(let i=0;i<40;i++){const l=await (await fetch(`http://127.0.0.1:${port}/json`)).json(); tg=l.find(x=>x.type==='page'); if(tg)break; await sleep(250)}
const s=new WebSocket(tg.webSocketDebuggerUrl); await new Promise(r=>s.onopen=r); let id=0; const pend={}; const errs=[];
s.onmessage=e=>{const m=JSON.parse(e.data); if(m.id&&pend[m.id]){pend[m.id](m);delete pend[m.id]} if(m.method==='Runtime.exceptionThrown')errs.push(m.params.exceptionDetails.text+' '+(m.params.exceptionDetails.exception?.description||'')); if(m.method==='Log.entryAdded')errs.push(m.params.entry.text)};
const cmd=(method,params={})=>new Promise(r=>{const i=++id;pend[i]=r;s.send(JSON.stringify({id:i,method,params}))});
const ev=async x=>(await cmd('Runtime.evaluate',{expression:x,returnByValue:true,awaitPromise:true})).result?.result?.value;
await cmd('Runtime.enable');await cmd('Log.enable');await cmd('Page.enable');
const shot=async n=>{const r=await cmd('Page.captureScreenshot',{format:'jpeg',quality:85});fs.writeFileSync(ROOT+'/qa/pk/'+n+'.jpg',Buffer.from(r.result.data,'base64'))};
try{
  let t0=Date.now(); for(let i=0;i<120;i++){if(await ev('!document.getElementById("play").disabled&&document.getElementById("film").readyState>=2'))break;await sleep(250)}
  console.log('ready after',Date.now()-t0,'ms', await ev('JSON.stringify({rs:film.readyState,d:film.duration,w:film.videoWidth,h:film.videoHeight})'));
  await shot('play0');
  const c=await ev('(()=>{const r=play.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()');
  for(const type of['mousePressed','mouseReleased'])await cmd('Input.dispatchMouseEvent',{type,x:c[0],y:c[1],button:'left',clickCount:1});
  await sleep(6000);
  console.log('playing',await ev('JSON.stringify({t:film.currentTime,p:film.paused,m:film.muted,v:film.volume,a:film.webkitAudioDecodedByteCount,q:film.getVideoPlaybackQuality().droppedVideoFrames+"/"+film.getVideoPlaybackQuality().totalVideoFrames,btn:getComputedStyle(play).opacity,frame:(()=>{const r=frame.getBoundingClientRect();return [r.width,r.height]})()})'));
  await shot('play1'); await ev('film.currentTime=49'); await sleep(2500); await shot('play2');
  console.log('mid',await ev('JSON.stringify({t:film.currentTime,q:film.getVideoPlaybackQuality().droppedVideoFrames+"/"+film.getVideoPlaybackQuality().totalVideoFrames})'));
  await ev('film.currentTime=62.6'); await sleep(900); await shot('play3'); await sleep(2800);
  console.log('end',await ev('JSON.stringify({t:film.currentTime,ended:film.ended,on:document.documentElement.classList.contains("on"),visible:document.body.innerText.trim()})'));
  await shot('play4'); console.log('errors',JSON.stringify(errs));
}finally{try{await cmd('Browser.close')}catch{} p.kill(); await sleep(800); fs.rmSync(ud,{recursive:true,force:true}); process.exit(0)}
