# Окружение AI Atlas

Dev Container — отдельная среда с Python 3.14, PyTorch 2.14.1 и Rust 1.98.1. Burn 0.22.1 закреплён в Cargo.toml. uv 0.13.5 восстанавливает Python-пакеты по uv.lock; Cargo.lock фиксирует Rust-зависимости. Примеры выполняются на CPU, в том числе на Mac: доступ к Apple GPU через MPS из Linux-контейнера не предоставляется.

1. Установи VS Code и запусти Docker или совместимый движок, например OrbStack.
2. [Открой проект в Dev Container](https://vscode.dev/redirect?url=vscode%3A%2F%2Fms-vscode-remote.remote-containers%2FcloneInVolume%3Furl%3Dhttps%3A%2F%2Fgithub.com%2Fkobzarvs%2Fai-learning). Подтверди открытие VS Code. Расширение Dev Containers будет предложено установить при необходимости.
3. Дождись завершения подготовки окружения. Первый запуск скачивает образ, языки и библиотеки.
4. В VS Code открой «Терминал → Создать терминал». Из корня проекта запусти нужный пример:

```sh
.venv/bin/python examples/python/lora.py --device cpu
```

Или вариант на Rust:

```sh
BURN_DEVICE=flex cargo run --locked --release --manifest-path examples/rust-burn/Cargo.toml --bin lora
```

На сайте на вкладке «Dev Container» показана команда именно выбранной темы. Все файлы уже есть в репозитории: создавать Cargo.toml и src/main.rs для этого способа запуска не нужно. Первый запуск Rust компилирует зависимости и занимает больше времени.

Если репозиторий уже открыт локально в VS Code, выбери команду **Dev Containers: Reopen in Container**. Она использует ту же конфигурацию. При ошибке установки открой журнал Dev Containers; не запускай примеры до завершения `postCreateCommand`.

Официальные инструкции: [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container), [образы Python](https://github.com/devcontainers/images/tree/main/src/python), [Rust feature](https://github.com/devcontainers/features/tree/main/src/rust).

Для NVIDIA CUDA и Apple Metal переключись на сайте на «На компьютере». Metal/MPS/MLX с GPU требуют нативного Mac; настройка Docker не открывает доступ к Apple GPU. MLX на CPU внутри Linux-контейнера запускается командой uv run --locked --extra cpu --extra mlx-cpu python examples/mlx/lora.py --device cpu.
