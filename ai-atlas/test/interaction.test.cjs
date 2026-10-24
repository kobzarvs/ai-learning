const {test}=require('node:test');
const assert=require('node:assert/strict');
const C=require('../core.js');
const data=require('../../map-data.json');
const down=(pan,options={})=>{pan.freshPointer();return pan.begin({id:1,onControl:true,...options});};
test('Pan is the default tool and reset preserves it',()=>{
 const pan=C.createPanController();assert.equal(pan.tool,'pan');assert.equal(pan.effective,'pan');
 assert.equal(down(pan),true);pan.end(1);pan.reset();assert.equal(pan.effective,'pan');
});
test('Shift release during a card drag keeps the gesture and blocks the resulting click',()=>{
 const pan=C.createPanController();pan.setTool('select');
 pan.modifier('ShiftLeft',true);assert.equal(pan.tool,'select');assert.equal(pan.effective,'pan');
 assert.equal(down(pan,{shiftKey:true}),true);
 pan.move(1,20,0);
 pan.modifier('ShiftLeft',false);assert.equal(pan.effective,'pan');assert.equal(pan.dragging,true);
 pan.end(1);assert.equal(pan.effective,'select');assert.equal(pan.dragging,false);assert.equal(pan.blocksClick(),true);
 pan.end(1);assert.equal(pan.blocksClick(),true,'lostpointercapture must not clear click suppression');
 assert.equal(down(pan),false);pan.end(1);assert.equal(pan.blocksClick(),false,'next ordinary click opens its card');
});
test('mouse-up before Shift-up and repeated Shift cycles never change the selected tool',()=>{
 const pan=C.createPanController();pan.setTool('select');
 for(let i=0;i<4;i++){
  pan.modifier('ShiftLeft',true);assert.equal(down(pan),true);pan.move(1,20,0);pan.end(1);
  assert.equal(pan.effective,'pan');assert.equal(pan.blocksClick(),true);
  pan.modifier('ShiftLeft',false);assert.equal(pan.tool,'select');assert.equal(pan.effective,'select');
 }
 pan.setTool('pan');pan.modifier('ShiftLeft',true);pan.modifier('ShiftLeft',false);
 assert.equal(pan.effective,'pan');assert.equal(pan.tool,'pan');
});
test('two Shift keys, keyboard repeat and Space are independent overrides',()=>{
 const pan=C.createPanController();pan.setTool('select');
 pan.modifier('ShiftLeft',true);pan.modifier('ShiftLeft',true);pan.modifier('ShiftRight',true);
 pan.modifier('ShiftLeft',false);assert.equal(pan.effective,'pan');
 pan.modifier('Space',true);pan.modifier('ShiftRight',false);assert.equal(pan.effective,'pan');
 pan.modifier('Space',false);assert.equal(pan.effective,'select');
});
test('a selection gesture cannot change into pan when Shift is pressed halfway through',()=>{
 const pan=C.createPanController();pan.setTool('select');assert.equal(down(pan),false);
 pan.modifier('ShiftLeft',true);assert.equal(pan.effective,'select');
 pan.end(1);assert.equal(pan.blocksClick(),false);assert.equal(pan.effective,'pan');
 pan.modifier('ShiftLeft',false);assert.equal(pan.effective,'select');
});
test('multi-pointer gestures, cancellation and loss of focus leave no stuck override',()=>{
 const pan=C.createPanController();pan.setTool('select');down(pan,{onControl:false});
 pan.begin({id:2,onControl:false});pan.end(1);assert.equal(pan.dragging,true);
 pan.modifier('ShiftLeft',true);pan.end(2);assert.equal(pan.dragging,false);
 pan.reset();assert.equal(pan.effective,'select');assert.equal(pan.dragging,false);
 pan.setTool('pan');down(pan);pan.reset();assert.equal(pan.effective,'pan');assert.equal(pan.blocksClick(),true);
 assert.equal(pan.blocksClick(true),false,'keyboard activation is not a synthetic mouse click');
});
test('middle-button pan and modifier flags work even if the keydown was outside the page',()=>{
 const pan=C.createPanController();pan.setTool('select');assert.equal(down(pan,{button:1}),true);pan.end(1);
 assert.equal(down(pan,{shiftKey:true}),true);pan.end(1);assert.equal(pan.effective,'select');
 assert.equal(down(pan,{button:2}),false);assert.equal(pan.dragging,false);
});
test('Pan distinguishes a click with hand jitter from a drag, even after returning to the start',()=>{
 const pan=C.createPanController();down(pan,{x:100,y:100});
 assert.equal(pan.dragging,false);assert.equal(pan.move(1,103,104),false);
 pan.end(1);assert.equal(pan.blocksClick(),false);
 down(pan,{x:100,y:100});assert.equal(pan.move(1,106,100),true);
 pan.move(1,100,100);pan.end(1);assert.equal(pan.blocksClick(),true);
 down(pan);pan.end(1);assert.equal(pan.blocksClick(),false);
 down(pan);pan.end(1,{cancelled:true});assert.equal(pan.blocksClick(),true);
});
test('term rendering uses whole terms, longest aliases, safe text, and optional keyboard focus',()=>{
 const render=C.createTermRenderer(data.glossary);
 const html=render('QLoRA, LoRA и CNN; CNNish и нейроCNN. <script>"&</script>',true);
 assert.equal((html.match(/data-term=/g)||[]).length,3);
 assert.match(html,/data-term="QLoRA"/);assert.match(html,/tabindex="0"/);
 assert.match(html,/&lt;script&gt;&quot;&amp;&lt;\/script&gt;/);assert.doesNotMatch(html,/<script>/);
 assert.doesNotMatch(render('CNN'),/tabindex/,'no nested focus targets in card buttons');
 assert.match(render('TF-IDF, QNN, Core ML'),/data-term="Core ML"/);
 assert.match(data.glossary.QNN.description,/квантовая/);assert.match(data.glossary.QNN.description,/точностью/);
});
test('every topic explains purpose, mechanism and an example; every displayed acronym has a definition',()=>{
 const index=C.indexTree(data.root),render=C.createTermRenderer(data.glossary);
 for(const n of index.values()){
  if(n.children)continue;
  for(const field of ['what','how','example','note'])assert.ok(n[field],`${n.id}: missing ${field}`);
  for(const [field,value] of Object.entries(n)){
   if(['sources','aliases','id','parent','branch'].includes(field))continue;
   const text=Array.isArray(value)?value.join(' '):typeof value==='string'?value:'';
   const terms=text.match(/(?<![\p{L}\p{N}])[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?(?![\p{L}\p{N}])/gu)||[];
   for(const term of terms)assert.match(render(term),/data-term=/,`${n.id}/${field}: ${term} is unexplained`);
  }
 }
 const project=index.get('lab-text');assert.ok(project.input&&project.output&&project.training&&project.criteria&&project.steps.length);
});
