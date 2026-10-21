// Loads the solver straight out of app/index.html, so the tests always check the code the app runs.
const fs=require("fs"),path=require("path"),vm=require("vm");
const html=fs.readFileSync(path.join(__dirname,"..","app","index.html"),"utf8");
const a=html.indexOf("const FACES={"),b=html.indexOf("/* ---------- main-thread state");
if(a<0||b<0)throw new Error("solver markers not found in app/index.html");
module.exports=vm.runInNewContext(html.slice(a,b)+"\n({FACES,makeSim,setTemp,coolingEffect,feelsLike});",{Math,Float32Array,Float64Array,Int32Array,Uint8Array,Set,Array,Object,JSON,Number,isNaN});
