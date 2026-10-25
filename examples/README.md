# Практика AI Atlas

Все примеры находятся в одном репозитории: по 115 программ PyTorch и Rust/Burn
для всех 113 тем и двух дополнительных игровых вариантов, а также 27 программ MLX. [Каталог всех примеров](INDEX.md) содержит прямые
ссылки на файлы для каждого поддерживаемого устройства.
Данные создаются внутри программ; ключи API и внешние датасеты не нужны.

## Склонируй один раз

Установи [Git](https://git-scm.com/downloads) и выполни:

```sh
git clone https://github.com/kobzarvs/ai-learning.git
cd ai-learning
```

Эта копия подходит для всех уроков и фреймворков. Если она уже есть,
перейди в корень `ai-learning` и выбирай нужную папку ниже.
Все команды `cd examples/...` выполняются из корня репозитория.

## Python / PyTorch

Установи [uv](https://docs.astral.sh/uv/getting-started/installation/).
Из корня `ai-learning`:

```sh
cd examples/python/cpu
uv sync --locked --extra cpu
uv run --locked --extra cpu python lora.py
```

Заменяй `lora.py` именем нужного файла из [каталога](INDEX.md).
Остальные примеры в этой папке используют то же окружение.

Для NVIDIA CUDA, из корня `ai-learning`:

```sh
cd examples/python/cuda
uv sync --locked --extra cuda
uv run --locked --extra cuda python lora.py
```

Нужны NVIDIA и совместимый драйвер CUDA 13. Для Apple Metal / MPS:

```sh
cd examples/python/mps
uv sync --locked --extra cpu
uv run --locked --extra cpu python lora.py
```

Нужен совместимый Mac и нативный запуск. Профиль `cpu` на macOS устанавливает
обычный PyTorch с поддержкой MPS. Некоторые операции не поддерживают MPS:
для таких тем файла в папке `mps` нет; используй CPU или CUDA.

`uv` находит общий `pyproject.toml` и `uv.lock` в корне репозитория даже при
запуске из подпапки. Он установит Python 3.14, создаст общую `.venv` и восстановит
закреплённые зависимости, включая PyTorch 2.14.1. Miniconda/mamba не нужны.
Профили `cpu` и `cuda` взаимоисключающие; переключение может переустановить пакеты.

## Python / MLX

MLX — отдельный фреймворк, его примеры не импортируют PyTorch.
Для GPU на Mac с Apple silicon, из корня `ai-learning`:

```sh
cd examples/mlx/metal
uv sync --locked --extra mlx
uv run --locked --extra mlx python lora.py
```

Для CPU на Mac с Apple silicon или в Linux:

```sh
cd examples/mlx/cpu
uv sync --locked --extra mlx --extra mlx-cpu
uv run --locked --extra mlx --extra mlx-cpu python lora.py
```

Маркеры платформы выбирают подходящий пакет MLX 0.32.4.
Варианты с SVD и решением линейной системы находятся только в папке `cpu`.

## Rust / Burn

Установи [Rust с Cargo](https://rustup.rs/), нужен Rust 1.95 или новее.
Из корня `ai-learning`:

```sh
cd examples/rust-burn
cargo fetch --locked
BURN_DEVICE=flex cargo run --locked --release --bin lora
```

Все 115 Rust-программ находятся в `src/bin/` и используют общие `Cargo.toml`
и `Cargo.lock` с Burn 0.22.1. Меняй `--bin lora` на имя нужного примера без `.rs`.
Создавать новый Cargo-проект или копировать код вручную не нужно.

Из той же папки, для GPU:

```sh
# NVIDIA: нужны драйвер и CUDA Toolkit
BURN_DEVICE=cuda cargo run --locked --release --features cuda --bin lora

# Apple: нужны macOS и Xcode Command Line Tools
BURN_DEVICE=metal cargo run --locked --release --features metal --bin lora
```

Первый запуск собирает зависимости и занимает больше времени.
Команды с `BURN_DEVICE=...` рассчитаны на bash/zsh; в PowerShell сначала задай
`$env:BURN_DEVICE = "flex"` (либо `"cuda"` / `"metal"`), затем выполни `cargo run`.
SVD в Burn 0.22.1 использует CPU. Наличие GPU-команды не означает проверку
каждого опыта на физическом GPU.

Rust-программа решает ту же учебную задачу, что Python-пример: это не обещание
полной реализации промышленной архитектуры, названной в теме. Классические
алгоритмы могут использовать обычный Rust; тензорные задачи используют Burn.

## Обучение игрового агента

В разделе три отдельных опыта с одинаковой задачей на Python и Rust/Burn:

| Пример | Файл Python в `examples/python/cpu/` | Имя Rust для `--bin` |
|---|---|---|
| Коридор из пяти клеток | `lab-game.py` | `lab-game` |
| Поле N×M, четыре направления | `lab-game-grid.py` | `lab-game-grid` |
| Стены и предметы ценой 1, 5 и 9 | `lab-game-items.py` | `lab-game-items` |

На сайте выбирай опыт кнопками «Коридор», «Поле N×M», «Стены и предметы».
Выбор сохраняется при переключении Python, Rust и задания. Для всех опытов
есть программы для поддерживаемых CPU/GPU и команды запуска из общей копии репозитория.

Размер поля задаётся `N` и `M`. Для готовой карты с предметами нужны N ≥ 4 и M ≥ 5;
пустое поле может быть любым прямоугольником минимум из двух клеток.
В маленьких опытах поддерживается не более 2048 нетерминальных состояний
и шести предметов. Каждый предмет собирается один раз: набор собранных предметов
входит в состояние. Стены и границы блокируют движение; за лишние шаги есть штраф.
Программа показывает карту, маршрут стрелками и полученную награду.

В двумерных опытах обучается линейный слой с отдельным входом для каждого
состояния — эквивалент таблицы Q. Известны все переходы маленькой среды.
Это учебное обучение оценок действий, без исследования неизвестной карты
и без обещания перенести навык на другой уровень без переобучения.
Попробуй поднять цену монеты до 20: изменится порядок сбора и маршрут.

## Файлы в уроках

Выбор устройства на сайте меняет папку, команды и показываемую программу.
Файлы в `python/cpu`, `python/cuda`, `python/mps`, `mlx/cpu` и `mlx/metal`
совпадают с кодом в уроке: каждый содержит только выбранное устройство.
Всего здесь 388 таких Python-файла. «Копировать» и «Скачать» доступны через
иконки в шапке кода; рядом — необязательная кнопка «Команда создания файла».
Она копирует команду записи показанного кода; выполняй её из папки примера.
Повторная запись заменяет существующий файл.

Универсальные `python/<тема>.py` и `mlx/<тема>.py` также сохранены.
Они принимают `--device cpu`, `--device cuda` / `mps` / `metal` или `auto`
и используются для пакетных проверок. Явно выбранный недоступный GPU
даёт ошибку, а не скрытый запуск на CPU.

## Dev Container

[Открыть в VS Code](https://vscode.dev/redirect?url=vscode%3A%2F%2Fms-vscode-remote.remote-containers%2FcloneInVolume%3Furl%3Dhttps%3A%2F%2Fgithub.com%2Fkobzarvs%2Fai-learning).
Нужны VS Code и запущенный Docker/OrbStack. После подготовки перейди
из корня проекта в нужную папку и выполни CPU-команды выше.
Для MLX внутри контейнера используй профиль `--extra cpu --extra mlx-cpu`
в обеих командах `uv`, чтобы сохранить подготовленное окружение PyTorch.

Этот контейнер использует CPU. Для Apple GPU нужен нативный запуск:
Linux-контейнер не получает доступ к Metal. NVIDIA-контейнер здесь не настроен.
[Подробная инструкция](../.devcontainer/README.md).

## Проверки и генерация

[GitHub Actions](https://github.com/kobzarvs/ai-learning/actions/workflows/check-examples.yml)
проверяет сайт и примеры на CPU, а также сборку GPU-вариантов Burn и Dev Container.
Смотри статус конкретного прогона; сборка GPU-варианта не равна запуску на GPU.
25.10.2026 все 115 Rust-программ собраны и успешно выполнены на Linux x86_64
harness с Rust 1.98.1, Burn 0.22.1 и CPU Flex; также прошли 184 Node-теста
и сборка сайта. [Результаты и хеши исходников](../work/rust-coverage/final-verification.json).
GPU runtime не проверялся. GitHub Actions для новых файлов ещё не запускался;
публикация этих изменений не выполнялась.

Для повторения CPU-проверки с Rust 1.98.1 из корня репозитория:

```sh
cd examples/rust-burn
cargo build --locked --bins
for source in src/bin/*.rs; do
  bin="$(basename "$source" .rs)"
  BURN_DEVICE=flex "target/debug/$bin" || exit 1
done
```

Отдельный пример: `BURN_DEVICE=flex cargo run --locked --bin lora`.
Проверка полноты исходников и команд запуска дополняет численное выполнение.

Источники уроков: `lessons/catalog.py`, `lessons/burn_catalog.py`, `lessons/mlx_catalog.py`.
Доменные модули `lessons/burn_*.py` регистрируют Rust-примеры через `register(add)`;
их подключение находится в конце `burn_catalog.py`.
Выбор устройства универсальных файлов: `lessons/torch_runtime.py`.
Программы для выбранного устройства и пояснения операций: `lessons/display_source.py`.

Из корня репозитория:

```sh
uv run --locked --extra cpu python lessons/generate.py
node --test ai-atlas/test/*.test.cjs
node ai-atlas/build.mjs
```

Генератор обновляет файлы примеров, каталог и данные сайта. Постоянные правки
вноси в источники уроков: повторная генерация заменяет сгенерированные файлы.
