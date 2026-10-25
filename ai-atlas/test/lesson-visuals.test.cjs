const {test}=require('node:test'),assert=require('node:assert/strict');
const V=require('../lesson-visuals.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);

test('assigning the instruction to harder tasks reverses the observed sign despite a true time saving',()=>{
 const biased=V.simulate({minutesSaved:4,assignment:'biased'});
 assert.ok(biased.observedDifference>0);
 assert.ok(biased.correlation>0);
 close(biased.change,-4);
 const random=V.simulate({minutesSaved:4,assignment:'random'});
 assert.ok(Math.abs(random.observedDifference+4)<2);
 close(random.change,-4);
 const noEffect=V.simulate({minutesSaved:0,assignment:'biased'});
 assert.ok(noEffect.observedDifference>0);
 close(noEffect.change,0);
});
test('paired interventions preserve each task and noise across all control settings',()=>{
 const data=V.observations();
 for(const minutesSaved of [0,4,8])for(const assignment of ['biased','random']){
  const result=V.simulate({data,minutesSaved,assignment});
  close(result.change,-minutesSaved);
  for(let i=0;i<data.difficulty.length;i++){
   close(result.oldTime[i],20+10*data.difficulty[i]+data.noise[i]);
   close(result.newTime[i]-result.oldTime[i],-minutesSaved);
   close(result.observed[i],result.instruction[i]?result.newTime[i]:result.oldTime[i]);
   const probability=assignment==='random'?0.5:0.1+0.2*(data.difficulty[i]-1);
   assert.equal(result.instruction[i],Number(data.assignmentDraw[i]<probability));
  }
 }
 assert.deepEqual(V.observations(),data);
});
test('charts share a horizontal row, labelled fixed axes and reproducible Python parameter hints',()=>{
 assert.equal(V.render('linear'),'');
 for(const minutesSaved of [0,4,8])for(const assignment of ['biased','random']){
  const markup=V.render('causal',{minutesSaved,assignment});
  assert.equal((markup.match(/<svg /g)||[]).length,2);
  assert.match(markup,/viewBox="0 0 660 460"/);
  assert.match(markup,/viewBox="0 0 660 370"/);
  assert.match(markup,/<div class="lesson-chart-row"><figure[\s\S]*<\/figure><figure[\s\S]*<\/figure><\/div>/);
  assert.match(markup,/Сложность задания: от 1 до 5/);
  assert.match(markup,/Разница времени, минуты/);
  assert.match(markup,/Минус — быстрее/);
  assert.ok(markup.includes(`data-causal-assignment="${assignment}" aria-pressed="true"`));
  assert.ok(markup.includes(`data-causal-saving="${minutesSaved}" aria-pressed="true"`));
  assert.ok(markup.includes(`minutes_saved = ${minutesSaved.toFixed(1)}`));
  assert.ok(markup.includes(`random_assignment = ${assignment==='random'?'True':'False'}`));
  assert.match(markup,/не данные предприятия/);
  assert.doesNotMatch(markup,/NaN|undefined|Infinity|морожен|купан/);
 }
});

test('separate distributions account for every task and explain the difference in means',()=>{
 for(const assignment of ['biased','random'])for(const minutesSaved of [0,4,8]){
  const result=V.simulate({assignment,minutesSaved});
  assert.equal(result.groups.reduce((sum,g)=>sum+g.count,0),1000);
  for(const [instruction,group] of result.groups.entries()){
   assert.equal(group.counts.reduce((a,b)=>a+b,0),group.count);
   close(group.shares.reduce((a,b)=>a+b,0),100);
   for(let bin=0;bin<4;bin++){
    const actual=result.difficulty.filter((d,i)=>result.instruction[i]===instruction&&d>=bin+1&&(bin===3?d<=5:d<bin+2)).length;
    assert.equal(group.counts[bin],actual);
   }
  }
  close(result.groups[1].meanTime-result.groups[0].meanTime,result.observedDifference);
 }
 const baseline=V.simulate({minutesSaved:0}),saving=V.simulate({minutesSaved:8});
 assert.deepEqual(baseline.groups.map(g=>g.counts),saving.groups.map(g=>g.counts));
 assert.ok(baseline.groups[0].counts[0]>baseline.groups[1].counts[0]);
 assert.ok(baseline.groups[1].counts[3]>baseline.groups[0].counts[3]);
 close(baseline.groups[1].meanTime-saving.groups[1].meanTime,8);
 const markup=V.render('causal',{minutesSaved:0});
 assert.equal((markup.match(/class="distribution-panel"/g)||[]).length,2);
 assert.match(markup,/55,46 − 43,63 ≈ 11,83 мин/);
 assert.match(markup,/При нулевой экономии/);
 assert.match(markup,/все 1000 заданий/);
 assert.doesNotMatch(markup,/<circle|240 точек/);
});
