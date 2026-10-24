# Практика AI Atlas

Проверенные по официальным источникам стабильные версии на 24.10.2026:
PyTorch **2.14.1**, Burn **0.22.1**. Примеры используют API Burn 0.22
(`Device`, `Tensor<D>`, Flex), `torch.export` и функциональное attention PyTorch.

113 самостоятельных Python-файлов соответствуют 113 темам карты. В 25 темах
есть дополнительный вариант на Rust. Некоторые близкие темы намеренно используют
один опыт с общим принципом. В описании каждого примера указано, обучает ли он
целую маленькую модель, отдельный компонент или только показывает механизм.
Синтетические данные создаются внутри программы: скачивание датасетов и ключи API
не нужны. Код выполняется на компьютере; сайт показывает, копирует и скачивает его.

## Python

Python 3.13, отдельное виртуальное окружение:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/python/requirements.txt
python examples/python/cnn.py
python examples/python/flowmatching.py
python examples/python/run_all.py
```

В PowerShell вместо `source` используется `.venv\Scripts\Activate.ps1`.
Большинство опытов работает на CPU; `mac.py` выбирает MPS, когда он доступен.

## Rust + Burn

Нужен Rust 1.95 или новее. В CI используется 1.98.1.

```sh
cargo run --manifest-path examples/rust-burn/Cargo.toml --bin linear
cargo run --manifest-path examples/rust-burn/Cargo.toml --bin mlp
cargo run --manifest-path examples/rust-burn/Cargo.toml --bin ae
```

Встроенные проверки проверяют результат, а не только успешную компиляцию.
Библиотека зафиксирована на `=0.22.1`; выбран Flex для запуска на CPU.

## Источники и проверка

- [PyTorch: текущий релиз](https://github.com/pytorch/pytorch/releases/latest)
- [Burn: текущий релиз](https://docs.rs/crate/burn/latest)
- [Миграция Burn 0.22](https://burn.dev/books/burn/migrating-to-0.22.html)
- [Автоматические прогоны](https://github.com/kobzarvs/ai-learning/actions/workflows/check-examples.yml)

Все **113 Python-примеров** выполнены на **PyTorch 2.14.1**, все **25 Rust-примеров**
собраны и выполнены на **Burn 0.22.1** в [успешном прогоне от 24.10.2026](https://github.com/kobzarvs/ai-learning/actions/runs/40081392239).
Проверка также включает 34 теста сайта и совпадение генерируемых файлов с исходниками.
Rust собирается с запретом устаревших API (`-D deprecated`). Прогоны выполнялись на CPU.
Успех на искусственных данных не доказывает качество для пользовательской задачи.

## Редактирование

Исходники уроков: `lessons/catalog.py` и `lessons/burn_catalog.py`.
После изменения:

```sh
python3 lessons/generate.py
node --test ai-atlas/test/*.test.cjs
node ai-atlas/build.mjs
```

Генератор одновременно обновляет запускаемые файлы и `ai-atlas/lesson-data.js`.
Не редактируй сгенерированные копии по отдельности.
