(function(scope){
'use strict';
const repository='https://github.com/kobzarvs/ai-learning';
const containerURL='https://vscode.dev/redirect?url='+encodeURIComponent('vscode://ms-vscode-remote.remote-containers/cloneInVolume?url='+repository);
const block=(id,label,language,code)=>({id,label,language,code});
const deviceChoices={
 python:[['cpu','CPU'],['cuda','NVIDIA CUDA'],['mps','Apple Metal · MPS']],
 rust:[['cpu','CPU · Flex'],['cuda','NVIDIA CUDA'],['metal','Apple Metal']],
 mlx:[['cpu','CPU'],['metal','Apple Metal']],
};
function examples(id,lesson){
 return [{id,title:lesson.exampleTitle||'Основной пример',lesson},...(lesson.variants||[])];
}
function selectExample(id,lesson,selected){
 return examples(id,lesson).find(example=>example.id===selected)||examples(id,lesson)[0];
}
function sourceCode(example,meta,language,device){
 const display=example.display;if(!display)throw Error('Missing display source');
 const runtime=language==='rust'?'':meta.deviceRuntimes?.[language]?.[device];
 if(language!=='rust'&&!runtime)throw Error('Missing device runtime');
 return [display.prefix,runtime,display.body].filter(Boolean).map(s=>s.trim()).join('\n\n')+'\n';
}
function plan(id,lesson,meta,{language='python',mode='local',device='cpu'}={}){
 if(!/^[a-z0-9-]+$/.test(id))throw Error('Invalid topic ID');
 const rust=language==='rust',mlx=language==='mlx',example=rust?lesson.rust:mlx?lesson.mlx:lesson;
 if(!example||!deviceChoices[language])throw Error('This topic has no example in the selected language');
 if(!deviceChoices[language].some(([key])=>key===device))throw Error('Invalid device');
 const allowed=example.devices||deviceChoices[language].map(([key])=>key);
 let blocked='';
 if(!allowed.includes(device))blocked=example.deviceNote;
 else if(mode==='container'&&device!=='cpu')blocked='Этот Linux Dev Container настроен для CPU. Для выбранного GPU переключись на «На компьютере»: Metal/MPS/MLX запускаются нативно на Mac, CUDA — на машине с NVIDIA.';
 if(blocked)return {steps:[],device,blocked};
 const folder=rust?'examples/rust-burn':'examples/'+language+'/'+device;
 const file=folder+(rust?'/src/bin/':'/')+id+(rust?'.rs':'.py');
 const source=block(language,file,rust?'rust':'python',sourceCode(example,meta,language,device));
 const profile=mlx?(mode==='container'?'--extra cpu --extra mlx-cpu':device==='metal'?'--extra mlx':'--extra mlx --extra mlx-cpu'):device==='cuda'?'--extra cuda':'--extra cpu';
 const install=rust?'cargo fetch --locked':'uv sync --locked '+profile;
 const run=rust?'BURN_DEVICE='+(device==='cpu'?'flex':device)+' cargo run --locked --release'+(device==='cpu'?'':' --features '+device)+' --bin '+id
  :'uv run --locked '+profile+' python '+id+'.py';
 const setup=mode==='container'?[
  {title:'Подготовь редактор и контейнеры',body:'Установи VS Code и запусти Docker либо совместимый движок, например OrbStack. Python, Rust и библиотеки будут установлены внутри отдельного окружения.',links:[['Установить VS Code','https://code.visualstudio.com/download'],['Настроить контейнеры','https://code.visualstudio.com/docs/devcontainers/containers#_getting-started']]},
  {title:'Открой готовый проект',body:'Подтверди открытие VS Code и дождись окончания подготовки. Все примеры уже есть в одном репозитории. Контейнер выполняет расчёты на CPU.',action:{label:'Открыть Dev Container в VS Code',url:containerURL}}
 ]:[
  {title:'Склонируй репозиторий один раз',body:'Одна копия ai-learning содержит все примеры Python, Rust и MLX. Если проект уже скачан, перейди в его корневую папку и пропусти эти команды. Для следующих уроков используй ту же копию.',links:[['Репозиторий с примерами',repository],...(rust?[['Установка Rust','https://rustup.rs/']]:[['Установка uv','https://docs.astral.sh/uv/getting-started/installation/']]),['Установка Git','https://git-scm.com/downloads']],blocks:[block('init','Терминал','bash','git clone '+repository+'.git\ncd ai-learning')]}
 ];
 return {device,folder,source,steps:[
  ...setup,
  {title:'Перейди в папку примера',body:'Выполни из корня ai-learning'+(mode==='container'?' в терминале VS Code':'')+'. Файл '+file+' уже находится в репозитории. Его можно сразу запускать и редактировать.',blocks:[block('folder','Терминал','bash','cd '+folder)]},
  {title:'Установи зависимости',body:rust?'Cargo скачает зависимости из Cargo.lock. При первом запуске он соберёт выбранный пример.':'В этой же папке выполни команду. uv найдёт общий pyproject.toml и uv.lock в корне ai-learning, создаст там .venv и установит закреплённые версии.'+(device==='mps'?' Профиль cpu на macOS включает поддержку MPS.':''),blocks:[block('install','Терминал','bash',install)]},
  {title:'Запусти пример',body:'Выполни в той же папке. После изменений повторяй эту команду. Чтобы открыть другой урок, вернись в корень ai-learning и перейди в указанную для него папку.',blocks:[block('run','Терминал','bash',run)]}
 ]};
}
function findBlock(plan,id){return plan.steps.flatMap(step=>step.blocks||[]).concat(plan.source||[]).find(block=>block.id===id);}
function createFileCommand(plan,id){
 const source=plan.source;
 if(!source||source.id!==id||!source.label.startsWith(plan.folder+'/'))throw Error('Invalid program path');
 const file=source.label.slice(plan.folder.length+1);
 if(!/^(?:[a-z0-9-]+\.py|src\/bin\/[a-z0-9-]+\.rs)$/.test(file))throw Error('Invalid program path');
 let delimiter='ATLAS_SOURCE';const lines=source.code.split(/\r?\n/);
 while(lines.includes(delimiter))delimiter+='_';
 const prepare=file.startsWith('src/bin/')?"mkdir -p 'src/bin' && ":'';
 return `# Выполни из ${plan.folder}; существующий ${file} будет заменён.\n${prepare}cat > '${file}' <<'${delimiter}'\n${source.code.endsWith('\n')?source.code:source.code+'\n'}${delimiter}\n`;
}
const api={plan,findBlock,sourceCode,createFileCommand,containerURL,deviceChoices,examples,selectExample};if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLaunch=api;
})(typeof window!=='undefined'?window:globalThis);
