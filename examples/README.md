# Практика AI Atlas

113 PyTorch-примеров; 25 самостоятельных Rust/Burn-вариантов и 27 MLX-вариантов.
Близкие темы могут использовать один опыт. Python 3.14, PyTorch 2.14.1,
Burn 0.22.1, MLX 0.32.4.

Rust решает ту же учебную задачу, что Python: LoRA обучает обе матрицы адаптера
при неизменной базе, свёрточная сеть учится различать полосы, резервуарная сеть обучает выход и
проверяет будущий участок сигнала. Демонстрации механизма внимания остаются демонстрациями
во всех вариантах. Данные создаются внутри программы; ключи API и датасеты не нужны.
Генераторы случайных чисел отличаются, поэтому побитовое совпадение не ожидается.

## Устройства

| Фреймворк | CPU | NVIDIA CUDA | Apple Metal |
|---|---|---|---|
| PyTorch | --device cpu | --device cuda | --device mps |
| Burn | BURN_DEVICE=flex | BURN_DEVICE=cuda, feature cuda | BURN_DEVICE=metal, feature metal |
| MLX | --device cpu | Здесь не настроен | --device metal |

MLX — отдельный фреймворк. Его код не импортирует PyTorch.
Metal/MPS требует нативного Mac, MLX/Metal — Apple silicon.
Явно выбранный недоступный GPU даёт ошибку, а не скрытый запуск на CPU.
Файлы в репозитории также поддерживают --device auto.

В окне урока выбор CPU/CUDA/MPS (или Metal для Burn/MLX) сразу меняет сам код.
Копирование и скачивание дают программу с одним выбранным устройством, без
разбора аргументов и ветвей для остальных устройств. Для Python/MLX сохраняй
её в указанный в инструкции файл, например examples/python/lora-mps.py:
команда запуска использует это же имя. Отдельного выбора ОС нет.

Для ряда PyTorch-опытов MPS отключён из-за используемых операций: это указано
в интерфейсе. Сингулярное разложение (SVD) в Burn 0.22.1 использует реализацию
на CPU. В MLX-примерах с SVD и решением линейной системы предлагается только
CPU, поскольку эти операции требуют процессора. Малые учебные опыты могут
работать на CPU быстрее.

## Python / uv

Установи [Git](https://git-scm.com/downloads) и
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Miniconda/mamba для этого проекта не требуются.

~~~sh
git clone https://github.com/kobzarvs/ai-learning.git
cd ai-learning

# CPU
uv sync --locked --extra cpu
uv run --locked --extra cpu python examples/python/lora.py --device cpu

# GPU Mac: на macOS профиль cpu устанавливает обычный PyTorch с MPS
uv run --locked --extra cpu python examples/python/lora.py --device mps

# NVIDIA: CUDA 13 и совместимый драйвер
uv sync --locked --extra cuda
uv run --locked --extra cuda python examples/python/lora.py --device cuda

# MLX на Apple silicon
uv sync --locked --extra mlx
uv run --locked --extra mlx python examples/mlx/lora.py --device metal
~~~

uv установит Python 3.14, создаст .venv и восстановит версии из uv.lock.
Профили cpu и cuda взаимоисключающие. При переключении профиля uv может
переустановить пакеты, сохраняя файлы опытов. Для MLX на CPU в Linux используется
профиль mlx-cpu, на Mac — mlx с аргументом --device cpu.

## Rust / Burn

Установи [Rust](https://rustup.rs/), нужен 1.95+, в CI используется 1.98.1.
Из корня репозитория:

~~~sh
BURN_DEVICE=flex cargo run --locked --release --manifest-path examples/rust-burn/Cargo.toml --bin lora
BURN_DEVICE=cuda cargo run --locked --release --manifest-path examples/rust-burn/Cargo.toml --features cuda --bin lora
BURN_DEVICE=metal cargo run --locked --release --manifest-path examples/rust-burn/Cargo.toml --features metal --bin lora
~~~

CUDA требует NVIDIA, драйвер и CUDA Toolkit; Metal — macOS и Xcode Command Line Tools.
Cargo.lock закреплён в репозитории вместе с дополнительными GPU-зависимостями.
Сайт также объясняет создание отдельного проекта: cargo new, затем Cargo.toml
и **src/main.rs**, затем запуск. Такой проект создаёт собственный Cargo.lock
при первой сборке; его нужно сохранять вместе с исходниками.

## Dev Container

[Открыть в VS Code](https://vscode.dev/redirect?url=vscode%3A%2F%2Fms-vscode-remote.remote-containers%2FcloneInVolume%3Furl%3Dhttps%3A%2F%2Fgithub.com%2Fkobzarvs%2Fai-learning).
Нужны VS Code и запущенный Docker/OrbStack. Этот контейнер использует CPU.
Для GPU Mac нужен нативный запуск; Linux-контейнер не получает доступ к Metal.
NVIDIA-контейнер с пробросом GPU здесь пока не настроен.
[Подробная инструкция](../.devcontainer/README.md).

## Проверки и исходники

[GitHub Actions](https://github.com/kobzarvs/ai-learning/actions/workflows/check-examples.yml)
проверяет сайт, 113 PyTorch, 25 Rust и 27 MLX-примеров на CPU, а также сборку
Dev Container на x86_64 и ARM64. Смотри статус конкретного прогона.
Сборка GPU-варианта не равна проверке исполнения на физическом GPU.

Источники уроков находятся в lessons/catalog.py, burn_catalog.py, mlx_catalog.py.
Выбор устройства для файлов репозитория — lessons/torch_runtime.py.
Короткие варианты для окна чтения и пояснения операций — lessons/display_source.py.
Комментарии стоят над операциями, имеют строки до 88 символов и переносятся
внутри окна чтения. Условие опыта показывается отдельно от кода.

~~~sh
uv run --locked --extra cpu python lessons/generate.py
node --test ai-atlas/test/*.test.cjs
node ai-atlas/build.mjs
~~~

Генератор обновляет запускаемые файлы и данные сайта. Не редактируй сгенерированные
копии по отдельности.
