"""Generate browser lesson data and standalone runnable files from one source."""
import json
from pathlib import Path
from catalog import LESSONS, PREFIX, HELPERS
from burn_catalog import RUST
from torch_runtime import runtime, device_info
from mlx_catalog import MLX
ROOT = Path(__file__).resolve().parent.parent
map_data = json.loads((ROOT/'map-data.json').read_text())
def leaves(node):
    if 'children' not in node:
        yield node
    for child in node.get('children', []):
        yield from leaves(child)

source_links = [
 ['PyTorch: обучение и оптимизация', 'https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html'],
 ['PyTorch: текущий стабильный релиз', 'https://github.com/pytorch/pytorch/releases/latest'],
]
meta = dict(date='24.10.2026', python='3.14', torch='2.14.1', burn='0.22.1', mlx='0.32.4',
 validation='Проверки CPU и сборки перечислены в GitHub Actions. Наличие команды CUDA/Metal не означает, что каждый опыт проверен на физическом GPU. При выборе недоступного устройства программа завершится с объяснением, не перейдёт скрыто на CPU.', validationUrl='https://github.com/kobzarvs/ai-learning/actions/workflows/check-examples.yml')
topics = {}
for node in leaves(map_data['root']):
    id=node['id']; original=LESSONS[id]; lesson={k:v for k,v in original.items() if k!='body'}
    helpers=''.join(text for name,text in HELPERS.items() if name+'(' in original['body'])
    lesson['code'] = PREFIX+runtime(id)+'\n'+helpers+'\n# '+lesson['goal']+'\n'+original['body']+'\n'
    lesson['devices'], lesson['deviceNote'] = device_info(id)
    lesson['steps'] = ['Прочитай условие выше. Запиши своими словами, что программа получает на вход и какой результат должна выдать.',
        'Создай окружение, установи закреплённую версию библиотеки и запусти код без изменений.',
        'Сравни вывод с ожидаемым результатом: '+lesson['expected'],
        'Проведи второй опыт: '+lesson['exercise'],
        'Создай заметку на холсте: запиши изменение, оба результата и свой вывод.']
    lesson['sources'] = list(source_links)
    if id in RUST:
        lesson['rust']=RUST[id]
        lesson['sources'] += [['Burn: текущая версия', 'https://docs.rs/crate/burn/latest'],
          ['Burn 0.22: изменения API', 'https://burn.dev/books/burn/migrating-to-0.22.html']]
    if id in MLX:
        lesson['mlx'] = MLX[id]
        lesson['sources'] += [['MLX: обучение и устройства', 'https://ml-explore.github.io/mlx/build/html/index.html']]
    topics[id]=lesson
    (ROOT/'examples/python'/f'{id}.py').write_text(lesson['code'])
# Rust binaries share exactly one pinned manifest; the reader exports an independent manifest.
bins=ROOT/'examples/rust-burn/src/bin';bins.mkdir(parents=True, exist_ok=True)
for id,lesson in RUST.items():
    (bins/f'{id}.rs').write_text(lesson['code'])
mlx_dir = ROOT/'examples/mlx'; mlx_dir.mkdir(parents=True, exist_ok=True)
for id, lesson in MLX.items():
    (mlx_dir/f'{id}.py').write_text(lesson['code'])
result={'meta':meta,'topics':topics}
(ROOT/'ai-atlas/lesson-data.js').write_text('window.ATLAS_LESSONS = '+json.dumps(result,ensure_ascii=False,separators=(',',':'))+';\n')
print(f'Generated {len(topics)} PyTorch, {len(RUST)} Rust, {len(MLX)} MLX examples.')
