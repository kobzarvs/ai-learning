"""Generate browser lesson data and standalone runnable files from one source."""
import json
from pathlib import Path
from catalog import LESSONS, PREFIX, HELPERS
from burn_catalog import RUST
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
meta = dict(date='24.10.2026', torch='2.14.1', burn='0.22.1',
 validation='Все 113 Python-примеров выполнены на PyTorch 2.14.1, все 25 Rust-примеров собраны и выполнены на Burn 0.22.1 в проверке GitHub от 24.10.2026. Проверка выполнялась на CPU. Численные результаты на другом устройстве могут немного различаться.', validationUrl='https://github.com/kobzarvs/ai-learning/actions/runs/40081392239')
topics = {}
for node in leaves(map_data['root']):
    id=node['id']; original=LESSONS[id]; lesson={k:v for k,v in original.items() if k!='body'}
    helpers=''.join(text for name,text in HELPERS.items() if name+'(' in original['body'])
    lesson['code'] = PREFIX+'\n'+helpers+'\n# '+lesson['goal']+'\n'+original['body']+'\n'
    lesson['steps'] = ['Прочитай задачу и сначала предположи результат: '+lesson['goal'],
        'Создай окружение, установи закреплённую версию библиотеки и запусти код без изменений.',
        'Сравни вывод с ожидаемым результатом: '+lesson['expected'],
        'Проведи второй опыт: '+lesson['exercise'],
        'Создай заметку на холсте: запиши изменение, оба результата и свой вывод.']
    lesson['sources'] = list(source_links)
    if id in RUST:
        lesson['rust']=RUST[id]
        lesson['sources'] += [['Burn: текущая версия', 'https://docs.rs/crate/burn/latest'],
          ['Burn 0.22: изменения API', 'https://burn.dev/books/burn/migrating-to-0.22.html']]
    topics[id]=lesson
    (ROOT/'examples/python'/f'{id}.py').write_text(lesson['code'])
# Rust binaries share exactly one pinned manifest; the reader exports an independent manifest.
bins=ROOT/'examples/rust-burn/src/bin';bins.mkdir(parents=True, exist_ok=True)
for id,lesson in RUST.items():
    (bins/f'{id}.rs').write_text(lesson['code'])
result={'meta':meta,'topics':topics}
(ROOT/'ai-atlas/lesson-data.js').write_text('window.ATLAS_LESSONS = '+json.dumps(result,ensure_ascii=False,separators=(',',':'))+';\n')
print(f'Generated {len(topics)} Python lessons, {len(RUST)} Rust examples.')
