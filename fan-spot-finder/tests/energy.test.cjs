// Mass and energy balances of the solver. Coarse grid (res 24) so the suite runs in a few minutes.
const test=require("node:test"),assert=require("node:assert");
const {makeSim,FACES}=require("./load-sim.cjs");
const RHOCP=1.2*1005;
const fluid=s=>{const a=[];for(let id=0;id<s.N;id++)if(s.flag[id]===0)a.push(id);return a;};
const meanT=(s,ids)=>ids.reduce((q,id)=>q+s.tp[id],0)/ids.length;
const heater={name:"Heater",x:2.8,y:1.8,z:0,dx:0.4,dy:0.4,dz:0.5,q:500};
const target={x:4.5,y:2,z:1.1,q:0};

test("sealed room, adiabatic walls: mean temperature rise = Q t / (rho cp V)",()=>{
  const sc={room:{L:6,W:4,H:2.7},boxes:[heater],openings:[],fans:[],target,turb:"fan",heat:{on:true,Tin:20,Tout:20,Twall:20,hw:0}};
  const s=makeSim(sc,24,FACES),ids=fluid(s),V=ids.length*s.h**3;
  for(let n=0;n<600;n++)s.step(40);
  const want=20+500*s.time/(RHOCP*V);
  assert.ok(Math.abs(meanT(s,ids)-want)<0.01,`mean ${meanT(s,ids).toFixed(3)} vs ${want.toFixed(3)}`);
});

test("sealed room, walls at 20 C: at steady state the walls take the 500 W",()=>{
  const sc={room:{L:6,W:4,H:2.7},boxes:[heater],openings:[],fans:[],target,turb:"fan",heat:{on:true,Tin:20,Tout:20,Twall:20}};
  const s=makeSim(sc,24,FACES);let q=0,n=0;
  for(let k=0;k<4000;k++){s.step(40);if(k>=3500){q+=s.qWall;n++;}}
  assert.ok(Math.abs(q/n+500)<15,`walls give ${(q/n).toFixed(1)} W, expected -500 W`);
});

test("window, cold outside: heat out of the window = heat from the walls, and inflow = outflow",()=>{
  const sc={room:{L:5,W:4,H:2.6},boxes:[],openings:[{kind:"open",face:"N",u:1.4,v:1.5,wu:1.0,wv:1.2}],fans:[],
    target:{x:3.75,y:1.25,z:1.1,q:0},turb:"fan",heat:{on:true,Tin:26,Tout:3,Twall:26}};
  const s=makeSim(sc,24,FACES),{nx,ny,nz,sx,sxy,h}=s;
  let Hn=0,qin=0,qout=0,qw=0,n=0;
  for(let k=0;k<3000;k++){s.step(40);if(k>=2500){n++;qw+=s.qWall;
    for(let kk=1;kk<=nz;kk++)for(let i=1;i<=nx;i++){const id=i+sx*ny+sxy*kk;if(s.flag[id+sx]!==3)continue;
      const v=s.v[id];Hn+=v*(v>0?s.tp[id]:3)*h*h*RHOCP;if(v>0)qout+=v*h*h;else qin-=v*h*h;}}}
  Hn/=n;qw/=n;qin/=n;qout/=n;
  assert.ok(Math.abs(qin-qout)/qout<0.01,`in ${qin.toFixed(4)} out ${qout.toFixed(4)} m3/s`);
  assert.ok(Math.abs(Hn-qw)/qw<0.03,`window ${Hn.toFixed(0)} W vs walls ${qw.toFixed(0)} W`);
});

test("default room: air never goes below the coldest input or far above the warmest",()=>{
  const sc={room:{L:5,W:4,H:2.6},boxes:[{name:"Desk",x:3.0,y:0.15,z:0,dx:1.5,dy:0.7,dz:0.75,q:150},{name:"Bed",x:2.6,y:2.3,z:0,dx:2.0,dy:1.5,dz:0.5,q:0}],
    openings:[{kind:"open",face:"N",u:1.4,v:1.5,wu:1.0,wv:1.2}],fans:[{name:"Fan",x:3.65,y:1.2,z:1.0,angle:90,d:0.4,speed:3}],
    target:{x:3.75,y:1.25,z:1.1,q:70},turb:"fan",heat:{on:true,Tin:26,Tout:28,Twall:26}};
  const s=makeSim(sc,24,FACES);let lo=1e9,hi=-1e9;
  for(let k=0;k<3000;k++){s.step(40);if(k%100===0)for(let id=0;id<s.N;id++)if(s.flag[id]===0||s.flag[id]===4){lo=Math.min(lo,s.tp[id]);hi=Math.max(hi,s.tp[id]);}}
  assert.ok(lo>=26-1e-3,`lowest ${lo.toFixed(3)} C`);assert.ok(hi<40,`highest ${hi.toFixed(2)} C`);
});
