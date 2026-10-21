// Oscillating fan: the jet turns with time, energy still balances, and the swing spreads the breeze.
const test=require("node:test"),assert=require("node:assert");
const {makeSim,FACES}=require("./load-sim.cjs");
const room={L:5,W:4,H:2.6},target={x:3.8,y:2,z:1.1,q:0};
const fan=o=>({name:"Fan",x:1,y:2,z:1.1,angle:0,d:0.4,speed:3,...o});
// mean direction (degrees) of the air speed set on the faces of the fan cells
function jetAngle(s){let a=0,b=0;
  for(let id=0;id<s.N;id++)if(s.flag[id]===4){a+=s.u[id]+s.u[id-1];b+=s.v[id]+s.v[id-s.sx];}
  return Math.atan2(b,a)*180/Math.PI;}

test("the jet direction follows angle + sweep/2 sin(2 pi t / cycle)",()=>{
  const s=makeSim({room,boxes:[],openings:[],fans:[fan({osc:true,sweep:90,period:4})],target,turb:"fan"},20,FACES);
  assert.ok(s.osc);
  let seen=[];
  while(s.time<4){s.step(20);const want=s.fanAngle(fan({osc:true,sweep:90,period:4}),s.time);
    assert.ok(Math.abs(jetAngle(s)-want)<0.5,`t ${s.time.toFixed(2)}: jet ${jetAngle(s).toFixed(1)} vs ${want.toFixed(1)}`);seen.push(want);}
  assert.ok(Math.max(...seen)>40&&Math.min(...seen)<-40,"the jet must swing to about +-45 degrees");
});

test("oscillation off: the jet stays at the set direction",()=>{
  const s=makeSim({room,boxes:[],openings:[],fans:[fan({angle:30,osc:false,sweep:90,period:4})],target,turb:"fan"},20,FACES);
  for(let n=0;n<50;n++)s.step(20);
  assert.ok(!s.osc&&Math.abs(jetAngle(s)-30)<0.5);
});

test("sealed room with heater and oscillating fan: mean temperature rise = Q t / (rho cp V)",()=>{
  const sc={room,boxes:[{name:"Heater",x:2.5,y:0.5,z:0,dx:0.4,dy:0.4,dz:0.5,q:500}],openings:[],fans:[fan({osc:true,sweep:120,period:6})],
    target,turb:"fan",heat:{on:true,Tin:20,Tout:20,Twall:20,hw:0}};
  const s=makeSim(sc,22,FACES),ids=[];for(let id=0;id<s.N;id++)if(s.flag[id]===0||s.flag[id]===4)ids.push(id);
  for(let n=0;n<400;n++)s.step(40);
  const mean=ids.reduce((q,id)=>q+s.tp[id],0)/ids.length,want=20+500*s.time/(1.2*1005*ids.length*s.h**3);
  assert.ok(Math.abs(mean-want)<0.01,`mean ${mean.toFixed(3)} vs ${want.toFixed(3)}`);
});

test("a point 40 degrees off the axis gets more breeze from the swinging fan than from a fixed one",()=>{
  const off={x:1+2.2*Math.cos(0.7),y:2+2.2*Math.sin(0.7),z:1.1,q:0};
  const run=osc=>{const s=makeSim({room,boxes:[],openings:[],fans:[fan({osc,sweep:90,period:6})],target:off,turb:"fan"},22,FACES);
    let a=0,n=0;while(s.time<30){s.step(30);if(s.time>12){a+=s.score("breeze",off);n++;}}return a/n;};
  const fixed=run(false),swing=run(true);
  assert.ok(swing>fixed*1.2,`swinging ${swing.toFixed(3)} m/s vs fixed ${fixed.toFixed(3)} m/s`);
});
