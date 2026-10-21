// Free jet of a 0.4 m pedestal fan. Reference: Buildings 2025, 15(3):308, Fig. 5 (4 household fans on high),
// centreline speed divided by the speed at 0.5 m, read from the figure at 1.0 to 4.0 m.
// Usage: node validation/fan-decay.cjs [res=64] [steps=1000] [maxRms=0.10]
const {makeSim,FACES}=require("../tests/load-sim.cjs");
const res=+(process.argv[2]||64),steps=+(process.argv[3]||1000),maxRms=+(process.argv[4]||0.1);
const D=[1,1.5,2,2.5,3,3.5,4],MEAS=[0.88,0.71,0.56,0.47,0.39,0.33,0.30];
const x0=0.8,zf=1.2,sc={room:{L:6,W:4,H:2.7},boxes:[],openings:[],target:{x:3,y:2,z:1.1},turb:"fan",fans:[{name:"f",x:x0,y:2,z:zf,angle:0,d:0.4,speed:4}]};
const s=makeSim(sc,res,FACES);
const at=(x,y,z)=>{const fx=x/s.h+0.5,fy=y/s.h+0.5,fz=z/s.h+0.5,i=Math.floor(fx),j=Math.floor(fy),k=Math.floor(fz);let r=0;
  for(let q=0;q<8;q++){const di=q&1,dj=(q>>1)&1,dk=q>>2,w=(di?fx-i:1-fx+i)*(dj?fy-j:1-fy+j)*(dk?fz-k:1-fz+k);r+=w*s.speed((i+di)+s.sx*(j+dj)+s.sxy*(k+dk));}return r;};
const acc=new Array(D.length+1).fill(0);let n=0;
for(let t=0;t<steps;t++){s.step(40);if(t>=steps*0.6){n++;[0.5,...D].forEach((d,q)=>acc[q]+=at(x0+d,2,zf));}}
const ref=acc[0]/n,ratio=acc.slice(1).map(a=>a/n/ref);
ratio.forEach((r,q)=>console.log(`${D[q]} m: model ${r.toFixed(2)}, measured ${MEAS[q]}`));
const R=Math.sqrt(ratio.reduce((q,r,i)=>q+(r-MEAS[i])**2,0)/D.length);
console.log(`RMS ${R.toFixed(3)} (limit ${maxRms}), grid ${(s.h*100).toFixed(1)} cm, ${s.time.toFixed(0)} s simulated`);
process.exit(R<=maxRms?0:1);
