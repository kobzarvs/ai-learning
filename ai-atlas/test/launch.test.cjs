const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const L=require('../launch.js'),ctx={window:{}};vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),ctx);
const {topics,meta}=ctx.window.ATLAS_LESSONS;
test('every Rust guide creates a binary project and supplies Cargo.toml and src/main.rs before running',()=>{
 for(const [id,lesson] of Object.entries(topics).filter(([,l])=>l.rust)){
  const p=L.plan(id,lesson,meta,{language:'rust'}),blocks=p.steps.flatMap(s=>s.blocks||[]);
  assert.match(blocks[0].code,/^cargo new --bin --vcs none /);assert.doesNotMatch(blocks[0].code,/cargo run/);
  assert.deepEqual(blocks.map(b=>b.id),['init','cargo','rust','run']);assert.equal(blocks[2].label,'src/main.rs');assert.equal(blocks[2].code,lesson.rust.code);
  assert.match(blocks[2].code,/fn main\(/);assert.match(blocks[1].code,/\[package\]/);assert.match(blocks[3].code,/cargo run --release$/);
  assert.ok(blocks[3].code.includes('BURN_DEVICE'));assert.ok(blocks[3].code.includes('flex'));
  for(const line of blocks[2].code.split('\n'))if(line.trimStart().startsWith('//'))assert.ok(line.length<=88,id+': long comment');
 }
});
test('every Python guide saves the selected source before running its matching filename',()=>{
 for(const [id,lesson] of Object.entries(topics)){
  const p=L.plan(id,lesson,meta),blocks=p.steps.flatMap(s=>s.blocks||[]);
  assert.deepEqual(blocks.map(b=>b.id),['init','install','python','run']);assert.match(blocks[1].code,/uv sync --locked --extra cpu/);
  assert.match(blocks[2].code,/device = torch.device\("cpu"\)/);
  assert.doesNotMatch(blocks[2].code,/argparse|SUPPORTED_DEVICES|requested_device/);
  assert.equal(blocks[2].label,'examples/python/'+id+'-cpu.py');
  assert.match(p.steps[2].body,/Создай файл/);
  assert.equal(blocks[3].code,'uv run --locked --extra cpu python '+blocks[2].label);
  assert.equal(L.findBlock(p,'run').code,blocks[3].code,'copy uses the displayed command');
 }
});
test('GPU instructions select a matching package/backend and never silently substitute CPU',()=>{
 const p=L.plan('lora',topics.lora,meta,{device:'cuda'});
 assert.match(L.findBlock(p,'install').code,/--extra cuda/);
 assert.match(L.findBlock(p,'run').code,/lora-cuda\.py$/);
 assert.match(L.findBlock(p,'python').code,/device = torch.device\("cuda"\)/);
 const rust=L.plan('lora',topics.lora,meta,{language:'rust',device:'cuda'});
 assert.match(L.findBlock(rust,'cargo').code,/"cuda"/);
 assert.doesNotMatch(L.findBlock(rust,'cargo').code,/"metal"|"flex"/);
 assert.match(L.findBlock(rust,'run').code,/BURN_DEVICE/);
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
   const device=lesson.mlx.devices.includes('metal')?'metal':'cpu';
   const p=L.plan(id,lesson,meta,{language:'mlx',device});
   assert.equal(lesson.mlx.goal,lesson.goal);
   assert.match(L.findBlock(p,'mlx').code,new RegExp('mx.set_default_device\\(mx.'+(device==='metal'?'gpu':'cpu')+'\\)'));
   assert.doesNotMatch(L.findBlock(p,'mlx').code,/argparse|requested/);
   assert.doesNotMatch(lesson.mlx.code,/import torch/);
   for(const field of ['exercise','explanation'])assert.doesNotMatch(lesson.mlx[field],/torch\.|PyTorch|scaled_dot_product_attention|is_causal/,id+': instructions must match MLX');
   assert.match(L.findBlock(p,'run').code,new RegExp('examples/mlx/'+id+'-'+device+'\\.py$'));
  }
 }
});
test('alternate framework exercises refer to functions present in their own examples',()=>{
 for(const language of ['rust','mlx']){
  assert.doesNotMatch(topics.transformer[language].exercise,/scaled_dot_product_attention|is_causal/);
  assert.match(topics.transformer[language].exercise,/attention/);
  assert.match(topics.transformer[language].code,/\battention\(/);
 }
 assert.doesNotMatch(topics.derivatives.mlx.exercise,/\bh\b/);
 assert.match(topics.derivatives.mlx.exercise,/numerical/);
 assert.match(topics.derivatives.mlx.code,/numerical =/);
});
test('container launches create selected Python files before running and use the existing Rust binary',()=>{
 const target=new URL(L.containerURL),protocol=new URL(target.searchParams.get('url'));
 assert.equal(target.origin,'https://vscode.dev');assert.equal(protocol.protocol,'vscode:');assert.equal(protocol.searchParams.get('url'),'https://github.com/kobzarvs/ai-learning');
 for(const [id,lesson] of Object.entries(topics))for(const language of lesson.rust?['python','rust']:['python']){
  const p=L.plan(id,lesson,meta,{language,mode:'container'}),file=language==='rust'?'../../examples/rust-burn/src/bin/'+id+'.rs':'../../examples/python/'+id+'.py';
  assert.equal(p.steps[1].action.url,L.containerURL);
  if(language==='rust')assert.equal(p.source.code,fs.readFileSync(require.resolve(file),'utf8'));
  else{
   assert.match(p.steps[2].body,/Создай файл/);
   assert.equal(p.steps[2].blocks[0].id,'python');assert.equal(p.steps[3].blocks[0].id,'run');
  }
  assert.ok(L.findBlock(p,'run').code.includes(language==='rust'?'--bin '+id:'examples/python/'+id+'-cpu.py'));
 }
 assert.throws(()=>L.plan('../escape',topics.lora,meta));
 const config=JSON.parse(fs.readFileSync(require.resolve('../../.devcontainer/devcontainer.json'),'utf8'));
 assert.equal(config.waitFor,'postCreateCommand');assert.match(config.postCreateCommand,/setup.sh/);
});
test('all device-specific listings contain only the selected runtime and short operation comments',()=>{
 const variants=[];
 for(const [id,lesson] of Object.entries(topics))for(const language of ['python','mlx']){
  const example=language==='python'?lesson:lesson.mlx;if(!example)continue;
  for(const [device] of L.deviceChoices[language]){
   const p=L.plan(id,lesson,meta,{language,device});
   if(!(example.devices||[]).includes(device)){
    assert.ok(p.blocked,id);assert.equal(L.findBlock(p,language),undefined);continue;
   }
   const source=L.findBlock(p,language);assert.ok(source,id);
   assert.doesNotMatch(source.code,/argparse|SUPPORTED_DEVICES|requested_device|parse_args|--device/);
   if(language==='python')for(const other of ['cpu','cuda','mps'].filter(x=>x!==device))assert.doesNotMatch(source.code,new RegExp('\\b'+other+'\\b','i'),id+'/'+device);
   else assert.doesNotMatch(source.code,device==='metal'?/mx\.cpu/:/mx\.gpu/);
   for(const line of source.code.split('\n'))if(line.trimStart().startsWith('#'))assert.ok(line.length<=88,id+': long comment '+line);
   assert.ok(source.code.endsWith(example.display.body),id+': preserve the complete experiment');
   variants.push({name:id+'/'+language+'/'+device,code:source.code,body:example.display.body});
  }
 }
 // Parse every displayed program without importing a framework or executing ML.
 const {spawnSync}=require('node:child_process');
 const parsed=spawnSync('python3',['-c',`import ast,json,sys
for item in json.load(sys.stdin):
 tree=ast.parse(item['code'], filename=item['name'])
 body=ast.parse(item['body']).body
 assert [ast.dump(n) for n in tree.body[-len(body):]] == [ast.dump(n) for n in body], item['name']
`],{input:JSON.stringify(variants),encoding:'utf8'});
 assert.equal(parsed.status,0,parsed.stderr);
 const causal=L.findBlock(L.plan('causal',topics.causal,meta,{device:'mps'}),'python').code;
 assert.match(causal,/# Вычисляем корреляции Пирсона[\s\S]*?\ncorrelation = torch.corrcoef/);
 assert.doesNotMatch(causal,/Создадим искусственные наблюдения/);
 assert.doesNotMatch(causal,/import math|import copy|from torch import nn/);
});
