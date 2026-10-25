"""Generate browser lesson data and standalone runnable files from one source."""
import json
from pathlib import Path
from catalog import LESSONS, PREFIX, HELPERS
from burn_catalog import RUST
from causal_guide import GUIDE
from torch_runtime import runtime, device_info
from mlx_catalog import MLX
from game_catalog import GAME_VARIANTS
from display_source import listing, annotate_rust, TORCH_RUNTIMES, MLX_RUNTIMES
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
meta['deviceRuntimes'] = {'python': TORCH_RUNTIMES, 'mlx': MLX_RUNTIMES}
compiled = {}
example_ids = [node['id'] for node in leaves(map_data['root'])]
example_ids += [id for id, title in GAME_VARIANTS if id not in example_ids]
for id in example_ids:
    original=LESSONS[id]; lesson={k:v for k,v in original.items() if k!='body'}
    helpers=''.join(text for name,text in HELPERS.items() if name+'(' in original['body'])
    lesson['devices'], lesson['deviceNote'] = device_info(id)
    lesson['display'] = listing(PREFIX, helpers+'\n'+original['body'],
                                {d: TORCH_RUNTIMES[d] for d in lesson['devices']})
    lesson['code'] = lesson['display']['prefix']+'\n\n'+runtime(id).strip()+'\n\n'+lesson['display']['body']
    lesson['steps'] = ['Прочитай условие выше. Запиши своими словами, что программа получает на вход и какой результат должна выдать.',
        'Создай окружение, установи закреплённую версию библиотеки и запусти код без изменений.',
        'Сравни вывод с ожидаемым результатом: '+lesson['expected'],
        'Проведи второй опыт: '+lesson['exercise'],
        'Создай заметку на холсте: запиши изменение, оба результата и свой вывод.']
    lesson['sources'] = list(source_links)
    if id == 'causal':
        lesson['guide'] = GUIDE
    if id in RUST:
        code = annotate_rust(RUST[id]['code'])
        lesson['rust']={**RUST[id], 'code': code, 'display': {'body': code}}
        lesson['sources'] += [['Burn: текущая версия', 'https://docs.rs/crate/burn/latest'],
          ['Burn 0.22: изменения API', 'https://burn.dev/books/burn/migrating-to-0.22.html']]
    if id in MLX:
        lesson['mlx'] = MLX[id]
        lesson['sources'] += [['MLX: обучение и устройства', 'https://ml-explore.github.io/mlx/build/html/index.html']]
    compiled[id]=lesson
    (ROOT/'examples/python'/f'{id}.py').write_text(lesson['code'])
# Rust binaries share one pinned manifest and lockfile.
bins=ROOT/'examples/rust-burn/src/bin';bins.mkdir(parents=True, exist_ok=True)
for id,lesson in RUST.items():
    (bins/f'{id}.rs').write_text(compiled[id]['rust']['code'])
mlx_dir = ROOT/'examples/mlx'; mlx_dir.mkdir(parents=True, exist_ok=True)
for id, lesson in MLX.items():
    (mlx_dir/f'{id}.py').write_text(lesson['code'])
# Save exactly the device-specific programs shown in the lesson reader.
# Subfolders keep the existing --device batch runners independent of these listings.
listing_count = 0
for id, lesson in compiled.items():
    for language, example, runtimes in [
        ('python', lesson, TORCH_RUNTIMES),
        ('mlx', lesson.get('mlx'), MLX_RUNTIMES),
    ]:
        if example is None:
            continue
        for device in example['devices']:
            folder = ROOT/'examples'/language/device
            folder.mkdir(parents=True, exist_ok=True)
            display = example['display']
            code = '\n\n'.join(part.strip() for part in [
                display.get('prefix'), runtimes[device], display['body']
            ] if part) + '\n'
            (folder/f'{id}.py').write_text(code)
            listing_count += 1

# A browseable index connects every topic to its checked-in runnable variants.
index = ['# Каталог примеров AI Atlas', '',
         'Сначала [склонируй репозиторий и установи зависимости](README.md).',
         'Python/MLX: перейди в папку файла и запусти `uv run` с профилем из инструкции.',
         'Rust: перейди в `rust-burn` и укажи имя файла без `.rs` в `--bin`.', '',
         '| Тема | PyTorch | Rust / Burn | MLX |', '|---|---|---|---|']
index_entries = [(node['id'], node['name']) for node in leaves(map_data['root'])]
index_entries += [(id, 'Игровой агент: '+title) for id, title in GAME_VARIANTS[1:]]
for id, title in index_entries:
    lesson = compiled[id]
    links = []
    for language, example in [('python', lesson), ('mlx', lesson.get('mlx'))]:
        links.append(' · '.join(
            f'[{device.upper()}]({language}/{device}/{id}.py)'
            for device in example['devices']) if example else '—')
    rust = f'[{id}](rust-burn/src/bin/{id}.rs)' if lesson.get('rust') else '—'
    index.append(f"| {title} (`{id}`) | {links[0]} | {rust} | {links[1]} |")
(ROOT/'examples/INDEX.md').write_text('\n'.join(index)+'\n')

topics = {node['id']: compiled[node['id']] for node in leaves(map_data['root'])}
topics['lab-game']['exampleTitle'] = GAME_VARIANTS[0][1]
topics['lab-game']['variants'] = [
    dict(id=id, title=title, lesson=compiled[id]) for id, title in GAME_VARIANTS[1:]
]
result={'meta':meta,'topics':topics}
(ROOT/'ai-atlas/lesson-data.js').write_text('window.ATLAS_LESSONS = '+json.dumps(result,ensure_ascii=False,separators=(',',':'))+';\n')
print(f'Generated {len(compiled)} PyTorch, {len(RUST)} Rust, {len(MLX)} MLX examples; {listing_count} device-specific files.')
