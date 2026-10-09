// CPU encode comparison; setup, completion and pixel readback are outside the timed region.
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
const args=process.argv.slice(2);
if(args[0]==='--child') console.log(JSON.stringify(await measure(args[1],args[2],Number(args[3]))));
else {
 const entries=[];let runs=7,count=2000,modes=['unique','repeated','alternating','single'];
 for(let i=0;i<args.length;i++) { if(args[i]==='--entry'){const s=args[++i],j=s.indexOf('=');if(j<1)throw Error('Expected label=entry');entries.push({label:s.slice(0,j),path:s.slice(j+1)});}else if(args[i]==='--runs')runs=Number(args[++i]);else if(args[i]==='--count')count=Number(args[++i]);else if(args[i]==='--mode')modes=args[++i].split(',');else throw Error(`Unknown option ${args[i]}`); }
 if(!entries.length||!Number.isInteger(runs)||runs<1||!Number.isInteger(count)||count<1)throw Error('Invalid benchmark arguments');
 const results=[];
 for(const mode of modes){const samples=entries.map(e=>({...e,samples:[],pixels:[],trials:[]}));for(let run=0;run<runs;run++)for(const e of (run%2?[...samples].reverse():samples)){const child=spawnSync(process.execPath,[fileURLToPath(import.meta.url),'--child',e.path,mode,String(count)],{encoding:'utf8',env:process.env});if(child.status!==0)throw Error(`${e.label}/${mode}: ${child.stderr}`);const r=JSON.parse(child.stdout.trim().split('\n').at(-1));e.samples.push(r.usPerDraw);e.trials.push(r);if(r.pixelHash)e.pixels.push(r.pixelHash);}for(const e of samples){const s=[...e.samples].sort((a,b)=>a-b);e.median=s[Math.floor(s.length/2)];e.min=s[0];e.max=s.at(-1);}if(samples.some(e=>e.pixels.some(hash=>hash!==samples[0].pixels[0])))throw Error(`Pixels changed in ${mode}`);results.push({mode,entries:samples});}
 const hash=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
 console.log(JSON.stringify({environment:{node:process.version,platform:`${process.platform}/${process.arch}`,cpu:cpus()[0]?.model,count,runs,trialOrder:'alternating baseline/candidate process order; no samples discarded'},harnessHash:hash(fileURLToPath(import.meta.url)),artifacts:entries.map(e=>({label:e.label,entry:e.path,entryHash:hash(e.path),drawHash:hash(new URL('./draw.js',pathToFileURL(e.path))),frameHash:hash(new URL('./frame.js',pathToFileURL(e.path)))})),results},null,2));
}
async function measure(entry,mode,n){
 const api=await import(pathToFileURL(entry).href);const {prepareShader}=await import(new URL('../packages/wgsl/dist/prepare.js',import.meta.url));
 const {init,draw,geometry,target,frame}=api;const gpu=await init();
 try {
 const source=`struct Frame { viewProjection: mat4x4f } struct Object { world: mat4x4f } struct Material { color: vec4f }
@group(0) @binding(0) var<uniform> scene: Frame; @group(1) @binding(0) var<uniform> own: Object; @group(2) @binding(0) var<uniform> material: Material;
@vertex fn vs_main(@location(0) position:vec3f)->@builtin(position) vec4f{return scene.viewProjection*own.world*vec4f(position,1);}
@fragment fn fs_main()->@location(0) vec4f{return material.color;}`;
 const matrix=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
 const buffer=bytes=>{const b=gpu.device.createBuffer({size:bytes.byteLength,usage:['uniform','copy_dst']});b.write(bytes);return b;};
 const scene=buffer(matrix), material=buffer(new Float32Array([.125,.5,.75,1]));
 const mesh=geometry(gpu,{buffers:[{data:new Float32Array([-1,-1,0,3,-1,0,-1,3,0]),stride:12,attributes:{position:{format:'float32x3',offset:0,location:0}}}],vertexCount:3});
 const count=mode==='single'?1:n, buildCount=mode==='unique'?count:mode==='alternating'?2:1;
 const shader=prepareShader(source), other=prepareShader(source.replace('return material.color;','return material.color * 0.5;'));
 const items=Array.from({length:buildCount},(_,i)=>draw(gpu,{shader:mode==='alternating'&&i===1?other:shader,geometry:mesh,set:{scene,own:buffer(matrix),material}}));
 const out=target(gpu,{size:[8,8],format:'rgba8unorm',depth:true});const passCount=mode==='single'?1:3;
 const run=()=>frame(gpu,f=>{for(let p=0;p<passCount;p++)f.pass(out,encoder=>{for(let i=0;i<count;i++)encoder.draw(items[mode==='unique'?i:mode==='alternating'?i%2:0]);});});
 const warm=mode==='single'?100:15,timed=mode==='single'?1000:60;
 for(let i=0;i<warm;i++){run();await gpu.settled();}
 let millis=0;for(let i=0;i<timed;i++){const t=performance.now();run();millis+=performance.now()-t;await gpu.settled();}
 const clockCalibration = calibrateClock();
 let pixelHash;
 if(entry.endsWith('/node.js')){const bytes=await out.color.read({mipLevel:0,region:'all'});const factor=mode==='alternating'&&(count-1)%2===1?.5:1;const expected=[Math.round(.125*factor*255),Math.round(.5*factor*255),Math.round(.75*factor*255),Math.round(factor*255)];for(let i=0;i<bytes.length;i++)if(bytes[i]!==expected[i%4])throw Error(`Incorrect ${mode} pixel ${i}: ${bytes[i]} vs ${expected[i%4]}`);pixelHash=createHash('sha256').update(bytes).digest('hex');}
 const usPerDraw=millis*1000/(timed*count*passCount);
 return {usPerDraw,pixelHash,drawsPerFrame:count*passCount,passesPerFrame:passCount,timedFrames:timed,warmupFrames:warm,clockCalibration,estimatedObserverUsPerDraw:clockCalibration.meanUs/(count*passCount),estimatedAdjustedUsPerDraw:usPerDraw-clockCalibration.meanUs/(count*passCount)};
 }finally{await gpu.dispose();}
}

// Measures the empty interval seen by this exact pair of clock reads. This estimates
// fixed boundary cost; it cannot remove JIT, GC, scheduling or instrumentation effects.
function calibrateClock() {
 const count=20000, values=[];
 for(let i=0;i<2000;i++){const t=performance.now();performance.now()-t;}
 for(let i=0;i<count;i++){const t=performance.now();values.push((performance.now()-t)*1000);}
 const sorted=[...values].sort((a,b)=>a-b);
 return {samples:count,meanUs:values.reduce((a,b)=>a+b,0)/count,medianUs:sorted[Math.floor(count/2)],p95Us:sorted[Math.floor(count*.95)],minUs:sorted[0],maxUs:sorted.at(-1)};
}
