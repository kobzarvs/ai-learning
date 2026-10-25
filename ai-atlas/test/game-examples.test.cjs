const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),{spawnSync}=require('node:child_process');
const L=require('../launch.js'),context={window:{}};
vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),context);
const {topics,meta}=context.window.ATLAS_LESSONS;
test('the game lesson offers three independently runnable examples in Python and Burn',()=>{
 assert.ok(topics['lab-game'].rust,'the corridor needs a real Burn implementation');
 const examples=L.examples('lab-game',topics['lab-game']);
 assert.deepEqual(examples.map(e=>e.id),['lab-game','lab-game-grid','lab-game-items']);
 for(const example of examples)for(const language of ['python','rust']){
  const p=L.plan(example.id,example.lesson,meta,{language});
  const source=L.findBlock(p,language);
  assert.equal(source.code,fs.readFileSync(path.resolve(__dirname,'../..',source.label),'utf8'));
  assert.ok(L.findBlock(p,'run').code.endsWith(language==='rust'?'--bin '+example.id:'python '+example.id+'.py'));
  assert.ok(example.lesson[language==='rust'?'rust':'goal']);
 }
 assert.equal(L.selectExample('lab-game',topics['lab-game'],'lab-game-items').id,'lab-game-items');
 assert.equal(L.selectExample('lab-game',topics['lab-game'],'missing').id,'lab-game');
});
test('grid movement obeys rectangular bounds, walls, terminal states and one-time item rewards',()=>{
 const example=topics['lab-game'].variants?.find(v=>v.id==='lab-game-items');
 assert.ok(example,'the items example must be published in the lesson');
 const checked=spawnSync('python3',['-c',`
import ast,json,sys
source=json.load(sys.stdin)
tree=ast.parse(source)
# Exercise the actual environment functions without importing a machine-learning runtime.
namespace={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'game-environment','exec'),namespace)
world=namespace['make_world'](3,5,walls={(1,3)},items={(1,2):('Монета',2.0),(2,1):('Алмаз',7.0)})
step=namespace['step']
for action,position in enumerate([(0,1),(2,1),(1,0),(1,2)]):
    state,reward,done=step(world,(1,1,0),action)
    assert state[:2]==position and not done,(action,state)
assert step(world,(0,0,0),0)==((0,0,0),-0.1,False)
assert step(world,(0,0,0),2)==((0,0,0),-0.1,False)
assert step(world,(2,0,0),1)==((2,0,0),-0.1,False)
assert step(world,(0,4,0),3)==((0,4,0),-0.1,False)
assert step(world,(1,2,0),3)==((1,2,0),-0.1,False)
state,reward,done=step(world,(1,1,0),3)
assert state==(1,2,1) and abs(reward-1.96)<1e-9
state,_,_=step(world,state,2)
state,reward,_=step(world,state,3)
assert state==(1,2,1) and reward==-0.04,'an item must pay only once'
state,reward,_=step(world,(1,1,1),1)
assert state==(2,1,3) and abs(reward-6.96)<1e-9
state,reward,done=step(world,(2,3,3),3)
assert state==(2,4,3) and done and abs(reward-1.96)<1e-9
assert step(world,state,2)==(state,0.0,True),'terminal cells cannot pay again'
for args in [(0,4),(-1,5),(3,0)]:
    try: namespace['make_world'](*args)
    except ValueError: pass
    else: raise AssertionError(args)
for options in [dict(walls={(0,0)}),dict(walls={(2,4)}),dict(walls={(3,0)}),dict(items={(1,3):('X',1)},walls={(1,3)}),dict(items={(0,0):('X',1)}),dict(walls={(1,c) for c in range(5)})]:
    try: namespace['make_world'](3,5,**options)
    except ValueError: pass
    else: raise AssertionError(options)
try:
    namespace['make_world'](3,4,walls={(0,3),(1,2)},items={(1,3):('X',1)})
except ValueError: pass
else: raise AssertionError('An item reached only through the terminal exit is unreachable')
for rows,cols in [(2,7),(7,2),(1,5),(5,1)]:
    world=namespace['make_world'](rows,cols)
    state=(0,0,0)
    for action in [3]*(cols-1)+[1]*(rows-1): state,_,done=step(world,state,action)
    assert done and state[:2]==(rows-1,cols-1)
print('environment checks passed')
`],{input:JSON.stringify(example.lesson.display.body),encoding:'utf8'});
 assert.equal(checked.status,0,checked.stderr);
});
