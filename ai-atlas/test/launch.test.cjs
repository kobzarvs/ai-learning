const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const L=require('../launch.js'),ctx={window:{}};vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),ctx);
const {topics,meta}=ctx.window.ATLAS_LESSONS;
test('every Rust guide creates a binary project and supplies Cargo.toml and src/main.rs before running',()=>{
 for(const [id,lesson] of Object.entries(topics).filter(([,l])=>l.rust)){
  for(const platform of ['unix','windows']){
   const p=L.plan(id,lesson,meta,{language:'rust',platform}),blocks=p.steps.flatMap(s=>s.blocks||[]);
   assert.match(blocks[0].code,/^cargo new --bin --vcs none /);assert.doesNotMatch(blocks[0].code,/cargo run/);
   assert.deepEqual(blocks.map(b=>b.id),['init','cargo','rust','run']);assert.equal(blocks[2].label,'src/main.rs');assert.equal(blocks[2].code,lesson.rust.code);
   assert.match(blocks[2].code,/fn main\(/);assert.match(blocks[1].code,/\[package\]/);assert.match(blocks[3].code,/cargo run --release$/);
   assert.ok(blocks[3].code.includes('BURN_DEVICE'));assert.ok(blocks[3].code.includes('flex'));
  }
 }
});
test('every Python guide installs the locked CPU profile and runs the displayed repository example',()=>{
 for(const [id,lesson] of Object.entries(topics))for(const platform of ['unix','windows']){
  const p=L.plan(id,lesson,meta,{platform}),blocks=p.steps.flatMap(s=>s.blocks||[]);
  assert.deepEqual(blocks.map(b=>b.id),['init','install','python','run']);assert.match(blocks[1].code,/uv sync --locked --extra cpu/);
  assert.equal(blocks[2].code,lesson.code);assert.equal(blocks[3].code,'uv run --locked --extra cpu python examples/python/'+id+'.py --device cpu');
  assert.equal(L.findBlock(p,'run').code,blocks[3].code,'copy uses the displayed command');
 }
});
test('GPU instructions select a matching package/backend and never silently substitute CPU',()=>{
 for(const platform of ['unix','windows']){
  const p=L.plan('lora',topics.lora,meta,{device:'cuda',platform});
  assert.match(L.findBlock(p,'install').code,/--extra cuda/);
  assert.match(L.findBlock(p,'run').code,/--device cuda$/);
  const rust=L.plan('lora',topics.lora,meta,{language:'rust',device:'cuda',platform});
  assert.match(L.findBlock(rust,'run').code,/--features burn\/cuda/);
  assert.match(L.findBlock(rust,'run').code,/BURN_DEVICE/);
 }
 for(const language of ['python','rust','mlx']){
  const device=language==='python'?'mps':'metal';
  const p=L.plan('lora',topics.lora,meta,{language,device,mode:'container'});
  assert.ok(p.blocked);assert.equal(L.findBlock(p,'run'),undefined);
 }
 const unsupported=L.plan('factorweights',topics.factorweights,meta,{device:'mps'});
 assert.ok(unsupported.blocked);assert.equal(L.findBlock(unsupported,'run'),undefined);
});
test('MLX examples and every Rust counterpart keep the original task and use independent source files',()=>{
 for(const [id,lesson] of Object.entries(topics)){
  if(lesson.rust){
   assert.equal(lesson.rust.goal,lesson.goal);
   assert.doesNotMatch(lesson.rust.limits,/во вкладке Python/);
  }
  if(lesson.mlx){
   const p=L.plan(id,lesson,meta,{language:'mlx',device:'metal'});
   assert.equal(lesson.mlx.goal,lesson.goal);
   assert.equal(L.findBlock(p,'mlx').code,fs.readFileSync(require.resolve('../../examples/mlx/'+id+'.py'),'utf8'));
   assert.doesNotMatch(lesson.mlx.code,/import torch/);
   assert.match(L.findBlock(p,'run').code,new RegExp('examples/mlx/'+id+'\\.py --device metal$'));
  }
 }
});
test('container launches reference existing repository files and the configured VS Code protocol',()=>{
 const target=new URL(L.containerURL),protocol=new URL(target.searchParams.get('url'));
 assert.equal(target.origin,'https://vscode.dev');assert.equal(protocol.protocol,'vscode:');assert.equal(protocol.searchParams.get('url'),'https://github.com/kobzarvs/ai-learning');
 for(const [id,lesson] of Object.entries(topics))for(const language of lesson.rust?['python','rust']:['python']){
  const p=L.plan(id,lesson,meta,{language,mode:'container'}),file=language==='rust'?'../../examples/rust-burn/src/bin/'+id+'.rs':'../../examples/python/'+id+'.py';
  assert.equal(p.steps[1].action.url,L.containerURL);assert.equal(p.source.code,fs.readFileSync(require.resolve(file),'utf8'));
  assert.ok(L.findBlock(p,'run').code.includes(language==='rust'?'--bin '+id:'examples/python/'+id+'.py'));
 }
 assert.throws(()=>L.plan('../escape',topics.lora,meta));
 const config=JSON.parse(fs.readFileSync(require.resolve('../../.devcontainer/devcontainer.json'),'utf8'));
 assert.equal(config.waitFor,'postCreateCommand');assert.match(config.postCreateCommand,/setup.sh/);
});
