// IEA Annex 20 2D test room (Nielsen): L 9 m, H 3 m, supply slot 0.168 m under the ceiling, outlet 0.48 m at the floor,
// U0 0.455 m/s. Reference: 106 LDA points (u/U0) on 4 lines, in annex20_reference.csv (column "measured").
// Usage: node validation/annex20.cjs [res=40] [steps=800] [maxRms=0.20]
const fs=require("fs"),path=require("path");
const {makeSim,FACES}=require("../tests/load-sim.cjs");
const res=+(process.argv[2]||40),steps=+(process.argv[3]||800),maxRms=+(process.argv[4]||0.2);
const H=3,L=9,W=3,hs=0.168,t=0.48,U0=0.455;
const sc={room:{L,W,H},boxes:[],fans:[],target:{x:1,y:1,z:1},turb:"fan",
  openings:[{kind:"supply",face:"W",u:W/2,v:H-hs/2,wu:W,wv:hs,speed:U0},{kind:"open",face:"E",u:W/2,v:t/2,wu:W,wv:t}]};
const s=makeSim(sc,res,FACES);
let nIn=0;for(let id=0;id<s.N;id++)if(s.flag[id]===2)nIn++;
sc.openings[0].speed=U0*hs/(nIn/Math.round(W/s.h)*s.h);s.build(sc);   // keep the supply flow rate right on a coarse grid
const j=Math.round(W/2/s.h+0.5),acc=new Float64Array(s.N);let n=0;
for(let st=0;st<steps;st++){s.step(40);if(st>=steps*0.75){n++;for(let k=1;k<=s.nz;k++)for(let i=1;i<=s.nx;i++){const id=i+s.sx*j+s.sxy*k;acc[id]+=0.5*(s.u[id]+s.u[id-1]);}}}
const U=(x,z)=>{const fx=Math.min(Math.max(x/s.h+0.5,1),s.nx-0.0001),fz=Math.min(Math.max(z/s.h+0.5,1),s.nz-0.0001),i=fx|0,k=fz|0,ax=fx-i,az=fz-k;
  const g=(i,k)=>{const id=i+s.sx*j+s.sxy*k;return s.flag[id]===0?acc[id]/n/U0:0;};
  return (1-az)*((1-ax)*g(i,k)+ax*g(i+1,k))+az*((1-ax)*g(i,k+1)+ax*g(i+1,k+1));};
const rows=fs.readFileSync(path.join(__dirname,"annex20_reference.csv"),"utf8").trim().split("\n").slice(1).map(r=>r.split(","));
const err={};let all=[];
for(const [line,a,m] of rows){const c=+a;
  const [x,z]=line==="x=H"?[H,H-c*H]:line==="x=2H"?[2*H,H-c*H]:line==="ceiling"?[c*H,H-hs/2]:[c*H,hs/2];
  const e=U(x,z)-+m;(err[line]=err[line]||[]).push(e);all.push(e);}
const rms=a=>Math.sqrt(a.reduce((q,e)=>q+e*e,0)/a.length);
for(const k in err)console.log(`${k}: RMS ${rms(err[k]).toFixed(3)} (${err[k].length} points)`);
const R=rms(all);console.log(`all: RMS ${R.toFixed(3)} of U0, ${all.length} points, grid ${(s.h*100).toFixed(0)} cm, ${s.time.toFixed(0)} s simulated (limit ${maxRms})`);
process.exit(R<=maxRms?0:1);
