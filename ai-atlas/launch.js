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
function plan(id,lesson,meta,{language='python',mode='local',platform='unix',device='cpu'}={}){
 if(!/^[a-z0-9-]+$/.test(id))throw Error('Invalid topic ID');
 const rust=language==='rust',mlx=language==='mlx',example=rust?lesson.rust:mlx?lesson.mlx:lesson;
 if(!example||!deviceChoices[language])throw Error('This topic has no example in the selected language');
 if(!deviceChoices[language].some(([key])=>key===device))throw Error('Invalid device');
 const shell=platform==='windows'?'powershell':'bash',folder='atlas-'+id+'-'+language;
 const file='examples/'+(rust?'rust-burn/src/bin/':mlx?'mlx/':'python/')+id+(rust?'.rs':'.py');
 const source=block(language,rust?'src/main.rs':file,rust?'rust':'python',example.code);
 const allowed=language==='python'?(lesson.devices||['cpu']):deviceChoices[language].map(([key])=>key);
 let blocked='';
 if(!allowed.includes(device))blocked=lesson.deviceNote+' Выбери CPU или CUDA для этого опыта.';
 else if(mode==='container'&&device!=='cpu')blocked='Этот Linux Dev Container настроен для CPU. Для выбранного GPU переключись на «На компьютере»: Metal/MPS/MLX запускаются нативно на Mac, CUDA — на машине с NVIDIA.';
 else if(platform==='windows'&&(mlx||device==='metal'||device==='mps')&&mode!=='container')blocked='Этот вариант требует нативного запуска на macOS. Для Windows выбери PyTorch или Rust с CPU/CUDA.';
 if(blocked)return {steps:[],source,device,blocked,prerequisite:''};
 const profile=mlx?'--extra mlx --extra mlx-cpu':device==='cuda'?'--extra cuda':'--extra cpu';
 const pyRun='uv run --locked '+profile+' python '+file+' --device '+device;
 const burnDevice=device==='cpu'?'flex':device;
 const features=device==='cpu'?'':' --features burn/'+device;
 const env=command=>platform==='windows'?"$env:BURN_DEVICE = '"+burnDevice+"'\n"+command:'BURN_DEVICE='+burnDevice+' '+command;
 const note=rust
  ?'Burn выбирает CPU, CUDA или Metal через BURN_DEVICE. Для GPU команда включает соответствующую возможность библиотеки при сборке. MLX — отдельный Python-фреймворк, а не устройство Burn.'
  :mlx?'MLX — самостоятельный фреймворк. На Apple silicon выбери Metal для GPU или CPU для процессора. Нужны нативные macOS и Python; Linux поддерживается здесь для CPU. Код не импортирует PyTorch.'
  :'CPU работает на процессоре; CUDA требует NVIDIA и совместимого драйвера CUDA 13; Metal/MPS использует GPU Mac при нативном запуске. Модель и данные размещаются на выбранном устройстве. Недоступный GPU выдаёт ошибку, а не скрытый запуск на CPU.';
 if(mode==='container')return {device,steps:[
  {title:'Подготовь редактор и контейнеры',body:'Установи VS Code и запусти Docker либо совместимый движок, например OrbStack. Python, Rust и библиотеки будут установлены внутри отдельного окружения.',links:[['Установить VS Code','https://code.visualstudio.com/download'],['Настроить контейнеры','https://code.visualstudio.com/docs/devcontainers/containers#_getting-started']]},
  {title:'Открой готовый проект',body:'Подтверди открытие VS Code и дождись окончания подготовки. Все файлы уже есть в репозитории. Этот контейнер выполняет расчёты на CPU; зависимости восстанавливаются из закреплённых lock-файлов.',action:{label:'Открыть Dev Container в VS Code',url:containerURL}},
  {title:'Запусти выбранный пример',body:'В VS Code открой «Терминал → Создать терминал». Рабочая папка — корень ai-learning. Первая сборка Rust или установка MLX может занять несколько минут.',blocks:[block('run','Терминал Dev Container','bash',rust?'BURN_DEVICE=flex cargo run --locked --release --manifest-path examples/rust-burn/Cargo.toml --bin '+id:mlx?'uv run --locked --extra cpu --extra mlx-cpu python '+file+' --device cpu':'.venv/bin/python '+file+' --device cpu')]}
 ],source,prerequisite:note};
 if(rust)return {device,prerequisite:'Нужен Rust 1.95 или новее с Cargo; установщик — rustup.rs. Для Metal нужны macOS и Xcode Command Line Tools; CUDA требует NVIDIA, драйвер и CUDA Toolkit. '+note,steps:[
  {title:'Создай проект с файлом программы',body:'Выполни команды по порядку. Cargo создаст папку, Cargo.toml и src/main.rs. Если имя папки занято, выбери другое в обеих командах.',blocks:[block('init','Терминал',shell,'cargo new --bin --vcs none '+folder+'\ncd '+folder)]},
  {title:'Запиши зависимости в Cargo.toml',body:'Полностью замени содержимое Cargo.toml в корне проекта этим блоком и сохрани. Это файл, не команда терминала.',blocks:[block('cargo','Cargo.toml','toml',example.cargo)]},
  {title:'Запиши программу в src/main.rs',body:'В папке src открой main.rs, замени его содержимое и сохрани. Скачанный main.rs нужно поместить именно в src, не рядом с Cargo.toml.',blocks:[source]},
  {title:'Собери и запусти',body:'Выполни команду из папки с Cargo.toml. Она выберет указанное устройство, скачает Burn и соберёт программу. Первый запуск дольше последующих.',blocks:[block('run','Терминал',shell,env('cargo run --release'+features))]}
 ]};
 return {device,prerequisite:'Нужны Git и uv. uv установит Python 3.14 и отдельное окружение проекта; Miniconda для этого способа не требуется. '+note,steps:[
  {title:'Скачай готовый учебный проект',body:'В терминале выполни команды. Если репозиторий уже скачан, просто перейди в его корневую папку и пропусти клонирование.',links:[['Установка uv','https://docs.astral.sh/uv/getting-started/installation/'],['Установка Git','https://git-scm.com/downloads']],blocks:[block('init','Терминал',shell,'git clone '+repository+'.git '+folder+'\ncd '+folder)]},
  {title:'Подготовь зависимости для выбранного устройства',body:'uv создаст .venv и восстановит совместимые версии из uv.lock. CPU и CUDA используют разные сборки PyTorch; на Mac обычная сборка поддерживает MPS. Для MLX устанавливается отдельный набор пакетов.',blocks:[block('install','Терминал',shell,'uv sync --locked '+profile)]},
  {title:'Открой готовый файл примера',body:'Файл '+file+' уже есть в проекте. Ниже его полное содержимое. Для своего опыта редактируй этот файл; скачанную копию можно положить по тому же пути.',blocks:[source]},
  {title:'Запусти и прочитай результат',body:'Из корня проекта выполни команду. В начале программа напечатает выбранное устройство. На маленьких данных GPU может быть медленнее CPU; сравнивай качество и время отдельно.',blocks:[block('run','Терминал',shell,pyRun)]}
 ]};
}
function findBlock(plan,id){return plan.steps.flatMap(step=>step.blocks||[]).concat(plan.source||[]).find(block=>block.id===id);}
const api={plan,findBlock,containerURL,deviceChoices};if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLaunch=api;
})(typeof window!=='undefined'?window:globalThis);
