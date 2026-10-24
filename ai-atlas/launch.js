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
function sourceCode(example,meta,language,device){
 const display=example.display;if(!display)throw Error('Missing display source');
 const runtime=language==='rust'?'':meta.deviceRuntimes?.[language]?.[device];
 if(language!=='rust'&&!runtime)throw Error('Missing device runtime');
 return [display.prefix,runtime,display.body].filter(Boolean).map(s=>s.trim()).join('\n\n')+'\n';
}
function selectedCargo(cargo,device){
 return cargo.replace(/\[features\]\n[\s\S]*?\[dependencies\]/,'[dependencies]')
  .replace('"flex"',JSON.stringify(device==='cpu'?'flex':device));
}
function plan(id,lesson,meta,{language='python',mode='local',device='cpu'}={}){
 if(!/^[a-z0-9-]+$/.test(id))throw Error('Invalid topic ID');
 const rust=language==='rust',mlx=language==='mlx',example=rust?lesson.rust:mlx?lesson.mlx:lesson;
 if(!example||!deviceChoices[language])throw Error('This topic has no example in the selected language');
 if(!deviceChoices[language].some(([key])=>key===device))throw Error('Invalid device');
 const shell='bash',folder='atlas-'+id+'-'+language;
 const file='examples/'+(rust?'rust-burn/src/bin/':mlx?'mlx/':'python/')+id+(rust?'.rs':'-'+device+'.py');
 const allowed=example.devices||deviceChoices[language].map(([key])=>key);
 let blocked='';
 if(!allowed.includes(device))blocked=example.deviceNote;
 else if(mode==='container'&&device!=='cpu')blocked='Этот Linux Dev Container настроен для CPU. Для выбранного GPU переключись на «На компьютере»: Metal/MPS/MLX запускаются нативно на Mac, CUDA — на машине с NVIDIA.';
 if(blocked)return {steps:[],device,blocked,prerequisite:''};
 const source=block(language,rust?'src/main.rs':file,rust?'rust':'python',sourceCode(example,meta,language,device));
 const profile=mlx?'--extra mlx --extra mlx-cpu':device==='cuda'?'--extra cuda':'--extra cpu';
 const pyRun='uv run --locked '+profile+' python '+file;
 const burnDevice=device==='cpu'?'flex':device;
 const env=command=>'BURN_DEVICE='+burnDevice+' '+command;
 const note=rust?{cpu:'Burn выполняет пример на процессоре через Flex.',cuda:'Нужны видеокарта NVIDIA, драйвер и CUDA Toolkit. Команда запуска явно выбирает CUDA.',metal:'Нужны macOS и Xcode Command Line Tools. Команда запуска явно выбирает Metal.'}[device]
  :mlx?{cpu:'MLX выполняет пример на процессоре.',metal:'Нужен нативный запуск на Mac с Apple silicon. В коде выбран Metal.'}[device]
  :{cpu:'Модель и данные создаются на процессоре.',cuda:'Нужны видеокарта NVIDIA, совместимый драйвер CUDA 13 и CUDA-сборка PyTorch. В коде выбрана CUDA.',mps:'Нужен совместимый Mac и нативный запуск. Модель и данные создаются на MPS.'}[device];
 if(mode==='container')return {device,steps:[
  {title:'Подготовь редактор и контейнеры',body:'Установи VS Code и запусти Docker либо совместимый движок, например OrbStack. Python, Rust и библиотеки будут установлены внутри отдельного окружения.',links:[['Установить VS Code','https://code.visualstudio.com/download'],['Настроить контейнеры','https://code.visualstudio.com/docs/devcontainers/containers#_getting-started']]},
  {title:'Открой готовый проект',body:'Подтверди открытие VS Code и дождись окончания подготовки. Все файлы уже есть в репозитории. Этот контейнер выполняет расчёты на CPU; зависимости восстанавливаются из закреплённых lock-файлов.',action:{label:'Открыть Dev Container в VS Code',url:containerURL}},
  ...(!rust?[{title:'Сохрани выбранный вариант программы',body:'Создай файл '+file+' в открытом проекте и скопируй в него код ниже. Устройство уже указано в программе.',blocks:[source]}]:[]),
  {title:'Запусти выбранный пример',body:'В VS Code открой «Терминал → Создать терминал». Рабочая папка — корень ai-learning. Первая сборка Rust или установка MLX может занять несколько минут.',blocks:[block('run','Терминал Dev Container','bash',rust?'BURN_DEVICE=flex cargo run --locked --release --manifest-path examples/rust-burn/Cargo.toml --bin '+id:mlx?'uv run --locked --extra cpu --extra mlx-cpu python '+file:'.venv/bin/python '+file)]}
 ],...(rust?{source}:{}),prerequisite:note};
 if(rust)return {device,prerequisite:'Нужен Rust 1.95 или новее с Cargo; установщик — rustup.rs. '+note,steps:[
  {title:'Создай проект с файлом программы',body:'Выполни команды по порядку. Cargo создаст папку, Cargo.toml и src/main.rs. Если имя папки занято, выбери другое в обеих командах.',blocks:[block('init','Терминал',shell,'cargo new --bin --vcs none '+folder+'\ncd '+folder)]},
  {title:'Запиши зависимости в Cargo.toml',body:'Полностью замени содержимое Cargo.toml в корне проекта этим блоком и сохрани. Это файл, не команда терминала.',blocks:[block('cargo','Cargo.toml','toml',selectedCargo(example.cargo,device))]},
  {title:'Запиши программу в src/main.rs',body:'В папке src открой main.rs, замени его содержимое и сохрани. Скачанный main.rs нужно поместить именно в src, не рядом с Cargo.toml.',blocks:[source]},
  {title:'Собери и запусти',body:'Выполни команду из папки с Cargo.toml. Она выберет указанное устройство, скачает Burn и соберёт программу. Первый запуск дольше последующих.',blocks:[block('run','Терминал',shell,env('cargo run --release'))]}
 ]};
 return {device,prerequisite:'Нужны Git и uv. uv установит Python 3.14 и отдельное окружение проекта; Miniconda для этого способа не требуется. '+note,steps:[
  {title:'Скачай готовый учебный проект',body:'В терминале выполни команды. Если репозиторий уже скачан, просто перейди в его корневую папку и пропусти клонирование.',links:[['Установка uv','https://docs.astral.sh/uv/getting-started/installation/'],['Установка Git','https://git-scm.com/downloads']],blocks:[block('init','Терминал',shell,'git clone '+repository+'.git '+folder+'\ncd '+folder)]},
  {title:'Подготовь зависимости для выбранного устройства',body:'uv создаст .venv и восстановит совместимые версии из uv.lock.'+(device==='mps'?' Профиль cpu на macOS устанавливает стандартную сборку PyTorch с поддержкой MPS.':''),blocks:[block('install','Терминал',shell,'uv sync --locked '+profile)]},
  {title:'Сохрани выбранный вариант программы',body:'Создай файл '+file+' и скопируй в него код ниже либо сохрани скачанный файл по этому пути. Папка уже есть в проекте. Устройство указано прямо в программе.',blocks:[source]},
  {title:'Запусти и прочитай результат',body:'Из корня проекта выполни команду. В начале программа напечатает выбранное устройство.',blocks:[block('run','Терминал',shell,pyRun)]}
 ]};
}
function findBlock(plan,id){return plan.steps.flatMap(step=>step.blocks||[]).concat(plan.source||[]).find(block=>block.id===id);}
const api={plan,findBlock,sourceCode,containerURL,deviceChoices};if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLaunch=api;
})(typeof window!=='undefined'?window:globalThis);
