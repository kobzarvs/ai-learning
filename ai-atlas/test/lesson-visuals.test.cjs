const {test}=require('node:test'),assert=require('node:assert/strict');
const V=require('../lesson-visuals.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);

test('a common cause produces correlation but an intervention has zero effect without a direct link',()=>{
 const data=V.observations();
 for(const intervention of [-3,1,3]){
  const result=V.simulate({data,effect:0,intervention});
  assert.ok(result.correlation>0.95);
  assert.deepEqual(result.after,result.swimming);
  close(result.change,0);
  for(const point of result.curve)close(point.y,result.beforeMean);
 }
});
test('causal effects preserve weather and noise and match the known structural equation',()=>{
 const data=V.observations();
 for(const effect of [0.3,0.6])for(const intervention of [-3,1,3]){
  const result=V.simulate({data,effect,intervention});
  close(result.change,effect*(intervention-result.consumptionMean));
  for(let i=0;i<data.weather.length;i++){
   close(result.swimming[i],2*data.weather[i]+effect*data.iceCream[i]+data.noise[i]);
   close(result.after[i]-result.swimming[i],effect*(intervention-data.iceCream[i]));
  }
  for(let i=1;i<result.curve.length;i++)close(result.curve[i].y-result.curve[i-1].y,0.5*effect);
 }
 assert.deepEqual(V.observations(),data);
});
test('charts keep labelled axes and fixed scales, with explicit controls and a simulation caption',()=>{
 assert.equal(V.render('linear'),'');
 for(const effect of [0,0.3,0.6]){
  const markup=V.render('causal',{effect,intervention:1});
  assert.equal((markup.match(/<svg /g)||[]).length,2);
  assert.equal((markup.match(/viewBox="0 0 720 408"/g)||[]).length,2);
  assert.match(markup,/Показатель потребления мороженого/);
  assert.match(markup,/Средняя активность/);
  assert.match(markup,/data-causal-intervention="1" aria-pressed="true"/);
  assert.ok(markup.includes(`data-causal-effect="${effect}" aria-pressed="true"`));
  assert.match(markup,/отдельная симуляция по формулам Python-примера/);
  assert.doesNotMatch(markup,/NaN|undefined|Infinity/);
 }
});
