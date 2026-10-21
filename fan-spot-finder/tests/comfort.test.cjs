// "Feels like" = air temperature minus the ASHRAE 55 cooling effect of air speed.
// Reference values: pythermalcomfort 4.6.2, cooling_effect(tdb=tr=T, vr=v, rh=50, met=1.1, clo=0.5).
const test=require("node:test"),assert=require("node:assert");
const {coolingEffect}=require("./load-sim.cjs");
const REF=[[22,0.2,0.75],[22,0.5,2.66],[22,1,4.1],[22,1.5,4.91],[26,0.2,0.79],[26,0.5,2.42],[26,1,3.51],[26,1.5,4.14],
  [30,0.2,0.88],[30,0.5,2.65],[30,1,3.7],[30,1.5,4.21]];
test("cooling effect matches pythermalcomfort within 0.02 K",()=>{
  for(const [t,v,ce] of REF){const got=coolingEffect(t,v,50,1.1,0.5);
    assert.ok(Math.abs(got-ce)<=0.02,`T=${t} v=${v}: got ${got.toFixed(3)}, expected ${ce}`);}
});
test("pythermalcomfort documentation example: 25 C, 0.3 m/s, 1.2 met gives 1.68 K",()=>{
  assert.ok(Math.abs(coolingEffect(25,0.3,50,1.2,0.5)-1.68)<=0.02);
});
test("no cooling effect in still air",()=>{assert.strictEqual(coolingEffect(26,0.1,50,1.1,0.5),0);});
