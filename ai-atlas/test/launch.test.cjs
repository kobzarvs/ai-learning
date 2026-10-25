const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const L=require('../launch.js'),ctx={window:{}};vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),ctx);
const {topics:rootTopics,meta}=ctx.window.ATLAS_LESSONS;
const topics=Object.fromEntries(Object.entries(rootTopics).flatMap(([id,lesson])=>L.examples(id,lesson).map(e=>[e.id,e.lesson])));
test('all guides share one clone and launch an existing file from the selected folder',()=>{
 const path=require('node:path'),root=path.resolve(__dirname,'../..');
 for(const [id,lesson] of Object.entries(topics))for(const language of ['python','rust','mlx']){
  const example=language==='python'?lesson:lesson[language];if(!example)continue;
  for(const [device] of L.deviceChoices[language])for(const mode of ['local','container']){
   const p=L.plan(id,lesson,meta,{language,device,mode});if(p.blocked)continue;
   const folder=language==='rust'?'examples/rust-burn':`examples/${language}/${device}`;
   const relative=folder+(language==='rust'?'/src/bin/':'/')+id+(language==='rust'?'.rs':'.py');
   if(mode==='local')assert.equal(L.findBlock(p,'init').code,'git clone https://github.com/kobzarvs/ai-learning.git\ncd ai-learning');
   assert.equal(L.findBlock(p,'folder')?.code,'cd '+folder);
   assert.ok(fs.existsSync(path.join(root,relative)),relative+' must exist after cloning');
   assert.equal(L.findBlock(p,language).label,relative);
   assert.equal(L.findBlock(p,language).code,fs.readFileSync(path.join(root,relative),'utf8'));
   assert.equal(p.steps.some(s=>s.createFile),false);
   const run=L.findBlock(p,'run').code;
   if(language==='rust'){
    assert.match(run,new RegExp('--bin '+id+'$'));
    assert.match(run,/cargo run --locked --release/);
    assert.equal(L.findBlock(p,'install').code,'cargo fetch --locked');
   }else{
    assert.ok(run.endsWith('python '+id+'.py'));
    assert.match(L.findBlock(p,'install').code,/^uv sync --locked /);
   }
  }
 }
});
test('GPU instructions select the matching dependency profile and backend',()=>{
 const python=L.plan('lora',topics.lora,meta,{device:'cuda'});
 assert.equal(L.findBlock(python,'folder').code,'cd examples/python/cuda');
 assert.equal(L.findBlock(python,'install').code,'uv sync --locked --extra cuda');
 assert.equal(L.findBlock(python,'run').code,'uv run --locked --extra cuda python lora.py');
 assert.match(L.findBlock(python,'python').code,/device = torch.device\("cuda"\)/);
 for(const [device,backend] of [['cpu','flex'],['cuda','cuda'],['metal','metal']]){
  const p=L.plan('lora',topics.lora,meta,{language:'rust',device});
  assert.equal(L.findBlock(p,'run').code,'BURN_DEVICE='+backend+' cargo run --locked --release'+(device==='cpu'?'':' --features '+device)+' --bin lora');
 }
 const mlx=L.plan('lora',topics.lora,meta,{language:'mlx',device:'metal'});
 assert.equal(L.findBlock(mlx,'install').code,'uv sync --locked --extra mlx');
 const cpu=L.plan('lora',topics.lora,meta,{language:'mlx',device:'cpu'});
 assert.equal(L.findBlock(cpu,'install').code,'uv sync --locked --extra mlx --extra mlx-cpu');
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
   assert.doesNotMatch(lesson.mlx.code,/import torch/);
   for(const field of ['exercise','explanation'])assert.doesNotMatch(lesson.mlx[field],/torch\.|PyTorch|scaled_dot_product_attention|is_causal/,id+': instructions must match MLX');
   assert.equal(L.findBlock(p,'run').code.split(' ').at(-1),id+'.py');
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
test('container guides preserve the prepared CPU environment and reference the shared repository',()=>{
 const target=new URL(L.containerURL),protocol=new URL(target.searchParams.get('url'));
 assert.equal(target.origin,'https://vscode.dev');assert.equal(protocol.protocol,'vscode:');assert.equal(protocol.searchParams.get('url'),'https://github.com/kobzarvs/ai-learning');
 const p=L.plan('lora',topics.lora,meta,{language:'mlx',mode:'container'});
 assert.equal(p.steps[1].action.url,L.containerURL);
 assert.equal(L.findBlock(p,'install').code,'uv sync --locked --extra cpu --extra mlx-cpu');
 assert.equal(L.findBlock(p,'run').code,'uv run --locked --extra cpu --extra mlx-cpu python lora.py');
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
 assert.match(causal,/observed_rows = torch.stack\(\[instruction_flag, observed_time\]\)/);
 assert.match(causal,/\n\n# corrcoef[\s\S]*?correlation_matrix = torch.corrcoef\(observed_rows\)/);
 assert.match(causal,/correlation = correlation_matrix\[0, 1\]/);
 assert.doesNotMatch(causal,/Создадим искусственные наблюдения/);
 assert.doesNotMatch(causal,/import math|import copy|from torch import nn/);
});
test('the Python causal model changes the instruction while preserving task difficulty and noise',()=>{
 // Execute the actual arithmetic function with scalar inputs, without requiring
 // an installed ML framework in the frontend test job.
 const {spawnSync}=require('node:child_process');
 const checked=spawnSync('python3',['-c',`import ast,sys,math
tree=ast.parse(sys.stdin.read())
model=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='completion_time')
module=ast.Module(body=[model],type_ignores=[])
namespace={}
exec(compile(module,'causal-model','exec'),namespace)
for difficulty in [1.0,3.0,5.0]:
 for saving in [0.0,4.0,8.0]:
  namespace.update(task_difficulty=difficulty,time_noise=0.17,minutes_saved=saving)
  before=namespace['completion_time'](0.0)
  after=namespace['completion_time'](1.0)
  assert math.isclose(after-before,-saving,abs_tol=1e-12)
for node in ast.walk(tree):
 if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.corrcoef':
  assert len(node.args)==1 and isinstance(node.args[0],ast.Name)
`],{input:topics.causal.display.body,encoding:'utf8'});
 assert.equal(checked.status,0,checked.stderr);
});

test('the optional creation command saves the displayed file relative to its launch folder',()=>{
 const {spawnSync}=require('node:child_process'),os=require('node:os'),path=require('node:path');
 const folder=fs.mkdtempSync(path.join(os.tmpdir(),'atlas-create-'));
 try{
  for(const [language,device,file] of [['python','mps','lora.py'],['mlx','metal','lora.py'],['rust','cpu','src/bin/lora.rs']]){
   const p=L.plan('lora',topics.lora,meta,{language,device});
   const saved=spawnSync('bash',['-c',L.createFileCommand(p,language)],{cwd:folder,encoding:'utf8'});
   assert.equal(saved.status,0,saved.stderr);
   assert.equal(fs.readFileSync(path.join(folder,file),'utf8'),L.findBlock(p,language).code);
  }
  const lesson={...topics.lora,display:{prefix:'',body:'text = """\nATLAS_SOURCE\n$(touch injected)\n`touch injected-too`\n"""\n'}};
  const p=L.plan('lora',lesson,meta);
  const saved=spawnSync('bash',['-c',L.createFileCommand(p,'python')],{cwd:folder,encoding:'utf8'});
  assert.equal(saved.status,0,saved.stderr);
  assert.equal(fs.readFileSync(path.join(folder,'lora.py'),'utf8'),L.findBlock(p,'python').code);
  assert.equal(fs.existsSync(path.join(folder,'injected')),false);
  assert.equal(fs.existsSync(path.join(folder,'injected-too')),false);
  assert.throws(()=>L.createFileCommand(p,'run'));
 }finally{fs.rmSync(folder,{recursive:true,force:true});}
});
