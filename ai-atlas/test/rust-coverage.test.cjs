const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{spawnSync}=require('node:child_process');
const A=require('../core.js'),C=require('../workspace-core.js'),L=require('../launch.js');
const root=path.resolve(__dirname,'../..'),map=require('../../map-data.json');
const leaves=[...A.indexTree(map.root).values()].filter(n=>!n.children);
const ctx={window:{}};vm.runInNewContext(fs.readFileSync(path.join(root,'ai-atlas/lesson-data.js'),'utf8'),ctx);
const {topics,meta}=ctx.window.ATLAS_LESSONS;
const examples=leaves.flatMap(node=>{const lesson=topics[node.id];return [{id:node.id,lesson,node},...(lesson?.variants||[]).map(v=>({...v,node}))];});
function python(code){const result=spawnSync('python3',['-c',code],{cwd:root,encoding:'utf8',env:{...process.env,PYTHONPATH:path.join(root,'lessons')}});assert.equal(result.status,0,result.stderr||result.stdout);return result.stdout;}
test('every map leaf and Python variant has complete Rust metadata, matching source, launch and reader card',()=>{
 assert.deepEqual(Object.keys(topics).sort(),leaves.map(n=>n.id).sort());
 const variants=JSON.parse(python("import json; from game_catalog import GAME_VARIANTS; print(json.dumps([id for id, title in GAME_VARIANTS]))"));
 const expected=[...new Set([...leaves.map(n=>n.id),...variants])].sort();
 assert.deepEqual(examples.map(e=>e.id).sort(),expected,'all Python variants must remain represented');
 const gaps=[];
 for(const {id,lesson,node} of examples){
  if(!lesson?.code){gaps.push(id+': Python lesson missing');continue;}
  if(!lesson.rust){gaps.push(id+': Rust missing');continue;}
  const rust=lesson.rust;
  assert.equal(rust.goal,lesson.goal,id+': task parity');
  for(const key of ['code','goal','explanation','expected','exercise','limits','parity','kind','cargo'])assert.ok(typeof rust[key]==='string'&&rust[key].trim(),id+': '+key);
  assert.match(rust.cargo,/version = "=0\.22\.1"/,id);
  const file=path.join(root,'examples/rust-burn/src/bin',id+'.rs');
  assert.ok(fs.existsSync(file),id+': standalone Rust file');
  assert.equal(rust.code,fs.readFileSync(file,'utf8'),id+': checked-in source');
  assert.equal(rust.display?.body,rust.code,id+': displayed source');
  for(const [device] of L.deviceChoices.rust){
   const plan=L.plan(id,lesson,meta,{language:'rust',device,mode:'local'});
   assert.ok(!plan.blocked,id+'/'+device+': launch blocked');
   assert.equal(L.findBlock(plan,'rust').code,rust.code,id+': reader source');
   assert.equal(L.findBlock(plan,'run').code,'BURN_DEVICE='+(device==='cpu'?'flex':device)+' cargo run --locked --release'+(device==='cpu'?'':' --features '+device)+' --bin '+id);
  }
  assert.ok(C.defaultBoard(node,topics).cards.some(card=>card.type==='rust'&&card.ref===node.id),id+': Rust reader card');
 }
 assert.deepEqual(gaps,[],`${gaps.length} coverage gaps:\n${gaps.join('\n')}`);
});
test('catalog preserves fallback exercises, accepts Rust exercises, and rejects duplicate registration without overwriting',()=>{
 python(`import burn_catalog as b\nfrom catalog import LESSONS\noriginal=b.RUST['linear'].copy()\ntry:\n b.add('linear', 'fn main() {}', 'explanation', 'expected')\nexcept ValueError:\n pass\nelse:\n raise AssertionError('duplicate topic accepted')\nassert b.RUST['linear']==original\n# Remove entries only in this isolated process to exercise compatibility.\nfor topic in ['arrays','optimization','derivatives']: del b.RUST[topic]\nb.add('arrays', 'fn main() {}', 'explanation', 'expected')\nassert b.RUST['arrays']['exercise']==LESSONS['arrays']['exercise']\nb.add('optimization', 'fn main() {}', 'explanation', 'expected')\nassert b.RUST['optimization']['exercise']==b.EXERCISES['optimization']\nb.add('derivatives', 'fn main() {}', 'explanation', 'expected', exercise='Измени Rust-пример')\nassert b.RUST['derivatives']['exercise']=='Измени Rust-пример'\n`);
});
test('pure Rust tasks omit unused Burn device scaffolding while Tensor tasks retain their source',()=>{
 python(`from textwrap import dedent\nimport burn_catalog as b\npure='\\n    fn main() { assert_eq!(2+2,4); }\\n'\nassert b.rust_source(pure)=='fn main() { assert_eq!(2+2,4); }\\n'\nfor body in ['fn main() { let d=device(); let x=Tensor::<1>::zeros([1], &d); }', 'fn main() { let d=device(); let x=Tensor::<1>::random([1], Distribution::Normal(0.0,1.0), &d); }']:\n header=b.HEADER if 'Distribution::' in body else b.HEADER.replace('Distribution, ', '')\n assert b.rust_source(body)==header+dedent(body).strip()+'\\n'\n`);
});
