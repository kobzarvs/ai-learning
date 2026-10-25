# Каталог примеров AI Atlas

Сначала [склонируй репозиторий и установи зависимости](README.md).
Python/MLX: перейди в папку файла и запусти `uv run` с профилем из инструкции.
Rust: перейди в `rust-burn` и укажи имя файла без `.rs` в `--bin`.

| Тема | PyTorch | Rust / Burn | MLX |
|---|---|---|---|
| Классификация и регрессия (`classification`) | [CPU](python/cpu/classification.py) · [CUDA](python/cuda/classification.py) · [MPS](python/mps/classification.py) | [classification](rust-burn/src/bin/classification.rs) | [CPU](mlx/cpu/classification.py) · [METAL](mlx/metal/classification.py) |
| Распознавание изображений (`vision`) | [CPU](python/cpu/vision.py) · [CUDA](python/cuda/vision.py) · [MPS](python/mps/vision.py) | [vision](rust-burn/src/bin/vision.rs) | [CPU](mlx/cpu/vision.py) · [METAL](mlx/metal/vision.py) |
| Работа с текстом и языком (`text`) | [CPU](python/cpu/text.py) · [CUDA](python/cuda/text.py) · [MPS](python/mps/text.py) | [text](rust-burn/src/bin/text.rs) | — |
| Распознавание звуков и речи (`audio`) | [CPU](python/cpu/audio.py) · [CUDA](python/cuda/audio.py) · [MPS](python/mps/audio.py) | [audio](rust-burn/src/bin/audio.rs) | — |
| Прогноз и анализ измерений во времени (`timeseries`) | [CPU](python/cpu/timeseries.py) · [CUDA](python/cuda/timeseries.py) · [MPS](python/mps/timeseries.py) | [timeseries](rust-burn/src/bin/timeseries.rs) | — |
| Объекты, связи и объёмная форма (`graphsdata`) | [CPU](python/cpu/graphsdata.py) · [CUDA](python/cuda/graphsdata.py) · [MPS](python/mps/graphsdata.py) | [graphsdata](rust-burn/src/bin/graphsdata.rs) | [CPU](mlx/cpu/graphsdata.py) · [METAL](mlx/metal/graphsdata.py) |
| Поиск и рекомендации (`recommend`) | [CPU](python/cpu/recommend.py) · [CUDA](python/cuda/recommend.py) · [MPS](python/mps/recommend.py) | [recommend](rust-burn/src/bin/recommend.rs) | — |
| Поиск необычных случаев (`anomaly`) | [CPU](python/cpu/anomaly.py) · [CUDA](python/cuda/anomaly.py) · [MPS](python/mps/anomaly.py) | [anomaly](rust-burn/src/bin/anomaly.rs) | — |
| Несколько видов данных вместе (`multimodal`) | [CPU](python/cpu/multimodal.py) · [CUDA](python/cuda/multimodal.py) · [MPS](python/mps/multimodal.py) | [multimodal](rust-burn/src/bin/multimodal.rs) | — |
| Выбор действий и планирование (`controltask`) | [CPU](python/cpu/controltask.py) · [CUDA](python/cuda/controltask.py) · [MPS](python/mps/controltask.py) | [controltask](rust-burn/src/bin/controltask.rs) | — |
| Линейные модели (`linear`) | [CPU](python/cpu/linear.py) · [CUDA](python/cuda/linear.py) · [MPS](python/mps/linear.py) | [linear](rust-burn/src/bin/linear.rs) | [CPU](mlx/cpu/linear.py) · [METAL](mlx/metal/linear.py) |
| Деревья решений и бустинг (`trees`) | [CPU](python/cpu/trees.py) · [CUDA](python/cuda/trees.py) · [MPS](python/mps/trees.py) | [trees](rust-burn/src/bin/trees.rs) | — |
| SVM — разделение групп границей (`svm`) | [CPU](python/cpu/svm.py) · [CUDA](python/cuda/svm.py) · [MPS](python/mps/svm.py) | [svm](rust-burn/src/bin/svm.rs) | — |
| kNN — ответ по похожим примерам (`knn`) | [CPU](python/cpu/knn.py) · [CUDA](python/cuda/knn.py) · [MPS](python/mps/knn.py) | [knn](rust-burn/src/bin/knn.rs) | [CPU](mlx/cpu/knn.py) · [METAL](mlx/metal/knn.py) |
| Наивный байесовский классификатор (`naivebayes`) | [CPU](python/cpu/naivebayes.py) · [CUDA](python/cuda/naivebayes.py) · [MPS](python/mps/naivebayes.py) | [naivebayes](rust-burn/src/bin/naivebayes.rs) | — |
| Кластеризация — поиск групп (`cluster`) | [CPU](python/cpu/cluster.py) · [CUDA](python/cuda/cluster.py) · [MPS](python/mps/cluster.py) | [cluster](rust-burn/src/bin/cluster.rs) | — |
| PCA и NMF — компактное описание данных (`decomposition`) | [CPU](python/cpu/decomposition.py) · [CUDA](python/cuda/decomposition.py) | [decomposition](rust-burn/src/bin/decomposition.rs) | — |
| Поиск выбросов: Isolation Forest и LOF (`isolation`) | [CPU](python/cpu/isolation.py) · [CUDA](python/cuda/isolation.py) · [MPS](python/mps/isolation.py) | [isolation](rust-burn/src/bin/isolation.rs) | — |
| ARIMA и ETS — прогноз временных рядов (`classicaltime`) | [CPU](python/cpu/classicaltime.py) · [CUDA](python/cuda/classicaltime.py) · [MPS](python/mps/classicaltime.py) | [classicaltime](rust-burn/src/bin/classicaltime.rs) | — |
| Байесовские модели и сети зависимостей (`bayes`) | [CPU](python/cpu/bayes.py) · [CUDA](python/cuda/bayes.py) | [bayes](rust-burn/src/bin/bayes.rs) | — |
| Гауссовские процессы (`gp`) | [CPU](python/cpu/gp.py) · [CUDA](python/cuda/gp.py) | [gp](rust-burn/src/bin/gp.rs) | — |
| Скрытые состояния: HMM и фильтр Калмана (`states`) | [CPU](python/cpu/states.py) · [CUDA](python/cuda/states.py) · [MPS](python/mps/states.py) | [states](rust-burn/src/bin/states.rs) | — |
| Смеси распределений и GMM (`mixtures`) | [CPU](python/cpu/mixtures.py) · [CUDA](python/cuda/mixtures.py) · [MPS](python/mps/mixtures.py) | [mixtures](rust-burn/src/bin/mixtures.rs) | — |
| Причинные модели (`causal`) | [CPU](python/cpu/causal.py) · [CUDA](python/cuda/causal.py) · [MPS](python/mps/causal.py) | [causal](rust-burn/src/bin/causal.rs) | — |
| MLP — обычная многослойная нейросеть (`mlp`) | [CPU](python/cpu/mlp.py) · [CUDA](python/cuda/mlp.py) · [MPS](python/mps/mlp.py) | [mlp](rust-burn/src/bin/mlp.rs) | [CPU](mlx/cpu/mlp.py) · [METAL](mlx/metal/mlp.py) |
| CNN — свёрточная нейросеть (`cnn`) | [CPU](python/cpu/cnn.py) · [CUDA](python/cuda/cnn.py) · [MPS](python/mps/cnn.py) | [cnn](rust-burn/src/bin/cnn.rs) | [CPU](mlx/cpu/cnn.py) · [METAL](mlx/metal/cnn.py) |
| RNN — нейросеть с памятью о последовательности (`rnn`) | [CPU](python/cpu/rnn.py) · [CUDA](python/cuda/rnn.py) · [MPS](python/mps/rnn.py) | [rnn](rust-burn/src/bin/rnn.rs) | — |
| LSTM и GRU — управление памятью сети (`lstm`) | [CPU](python/cpu/lstm.py) · [CUDA](python/cuda/lstm.py) · [MPS](python/mps/lstm.py) | [lstm](rust-burn/src/bin/lstm.rs) | — |
| TCN — свёртки по времени (`tcn`) | [CPU](python/cpu/tcn.py) · [CUDA](python/cuda/tcn.py) · [MPS](python/mps/tcn.py) | [tcn](rust-burn/src/bin/tcn.rs) | — |
| Transformer — сеть с механизмом внимания (`transformer`) | [CPU](python/cpu/transformer.py) · [CUDA](python/cuda/transformer.py) · [MPS](python/mps/transformer.py) | [transformer](rust-burn/src/bin/transformer.rs) | [CPU](mlx/cpu/transformer.py) · [METAL](mlx/metal/transformer.py) |
| GNN — нейросеть для связанных объектов (`gnn`) | [CPU](python/cpu/gnn.py) · [CUDA](python/cuda/gnn.py) · [MPS](python/mps/gnn.py) | [gnn](rust-burn/src/bin/gnn.rs) | [CPU](mlx/cpu/gnn.py) · [METAL](mlx/metal/gnn.py) |
| Сиамские сети и числовые представления (`siamese`) | [CPU](python/cpu/siamese.py) · [CUDA](python/cuda/siamese.py) · [MPS](python/mps/siamese.py) | [siamese](rust-burn/src/bin/siamese.rs) | — |
| ResNet, U-Net и ViT — устройства сетей для зрения (`specialblocks`) | [CPU](python/cpu/specialblocks.py) · [CUDA](python/cuda/specialblocks.py) · [MPS](python/mps/specialblocks.py) | [specialblocks](rust-burn/src/bin/specialblocks.rs) | [CPU](mlx/cpu/specialblocks.py) · [METAL](mlx/metal/specialblocks.py) |
| SSM и Mamba — компактная память последовательности (`ssm`) | [CPU](python/cpu/ssm.py) · [CUDA](python/cuda/ssm.py) · [MPS](python/mps/ssm.py) | [ssm](rust-burn/src/bin/ssm.rs) | [CPU](mlx/cpu/ssm.py) · [METAL](mlx/metal/ssm.py) |
| Линейное внимание, RWKV и гибридные сети (`linearattention`) | [CPU](python/cpu/linearattention.py) · [CUDA](python/cuda/linearattention.py) · [MPS](python/mps/linearattention.py) | [linearattention](rust-burn/src/bin/linearattention.rs) | — |
| MoE — сеть с выбором специалистов (`moe`) | [CPU](python/cpu/moe.py) · [CUDA](python/cuda/moe.py) · [MPS](python/mps/moe.py) | [moe](rust-burn/src/bin/moe.rs) | — |
| KAN — сеть с обучаемыми функциями связей (`kan`) | [CPU](python/cpu/kan.py) · [CUDA](python/cuda/kan.py) · [MPS](python/mps/kan.py) | [kan](rust-burn/src/bin/kan.rs) | — |
| Непрерывная динамика: Liquid, LTC и Neural ODE (`liquid`) | [CPU](python/cpu/liquid.py) · [CUDA](python/cuda/liquid.py) · [MPS](python/mps/liquid.py) | [liquid](rust-burn/src/bin/liquid.rs) | — |
| Резервуарные сети и ESN (`reservoir`) | [CPU](python/cpu/reservoir.py) · [CUDA](python/cuda/reservoir.py) | [reservoir](rust-burn/src/bin/reservoir.rs) | [CPU](mlx/cpu/reservoir.py) |
| AE — автоэнкодер (`ae`) | [CPU](python/cpu/ae.py) · [CUDA](python/cuda/ae.py) · [MPS](python/mps/ae.py) | [ae](rust-burn/src/bin/ae.rs) | [CPU](mlx/cpu/ae.py) · [METAL](mlx/metal/ae.py) |
| VAE — вероятностный автоэнкодер (`vae`) | [CPU](python/cpu/vae.py) · [CUDA](python/cuda/vae.py) · [MPS](python/mps/vae.py) | [vae](rust-burn/src/bin/vae.rs) | — |
| GAN — генератор и различающая сеть (`gan`) | [CPU](python/cpu/gan.py) · [CUDA](python/cuda/gan.py) · [MPS](python/mps/gan.py) | [gan](rust-burn/src/bin/gan.rs) | — |
| Диффузионные модели (`diffusion`) | [CPU](python/cpu/diffusion.py) · [CUDA](python/cuda/diffusion.py) · [MPS](python/mps/diffusion.py) | [diffusion](rust-burn/src/bin/diffusion.rs) | — |
| Авторегрессия — создание по одному элементу (`autoregression`) | [CPU](python/cpu/autoregression.py) · [CUDA](python/cuda/autoregression.py) · [MPS](python/mps/autoregression.py) | [autoregression](rust-burn/src/bin/autoregression.rs) | — |
| Нормализующие потоки (`flows`) | [CPU](python/cpu/flows.py) · [CUDA](python/cuda/flows.py) · [MPS](python/mps/flows.py) | [flows](rust-burn/src/bin/flows.rs) | — |
| Flow Matching — обучение пути от шума к данным (`flowmatching`) | [CPU](python/cpu/flowmatching.py) · [CUDA](python/cuda/flowmatching.py) · [MPS](python/mps/flowmatching.py) | [flowmatching](rust-burn/src/bin/flowmatching.rs) | — |
| EBM — модель оценки совместимости (`ebm`) | [CPU](python/cpu/ebm.py) · [CUDA](python/cuda/ebm.py) · [MPS](python/mps/ebm.py) | [ebm](rust-burn/src/bin/ebm.rs) | — |
| JEPA — предсказание представления скрытой части (`jepa`) | [CPU](python/cpu/jepa.py) · [CUDA](python/cuda/jepa.py) · [MPS](python/mps/jepa.py) | [jepa](rust-burn/src/bin/jepa.rs) | — |
| PINN — обучение с физическими ограничениями (`pinn`) | [CPU](python/cpu/pinn.py) · [CUDA](python/cuda/pinn.py) · [MPS](python/mps/pinn.py) | [pinn](rust-burn/src/bin/pinn.rs) | — |
| Нейронные операторы: FNO и DeepONet (`neuraloperator`) | [CPU](python/cpu/neuraloperator.py) · [CUDA](python/cuda/neuraloperator.py) | [neuraloperator](rust-burn/src/bin/neuraloperator.rs) | — |
| Правила, логика и поиск программ (`symbolic`) | [CPU](python/cpu/symbolic.py) · [CUDA](python/cuda/symbolic.py) · [MPS](python/mps/symbolic.py) | [symbolic](rust-burn/src/bin/symbolic.rs) | — |
| Символьная регрессия — поиск формулы (`symbolicreg`) | [CPU](python/cpu/symbolicreg.py) · [CUDA](python/cuda/symbolicreg.py) | [symbolicreg](rust-burn/src/bin/symbolicreg.rs) | — |
| Эволюционный поиск (`evolution`) | [CPU](python/cpu/evolution.py) · [CUDA](python/cuda/evolution.py) · [MPS](python/mps/evolution.py) | [evolution](rust-burn/src/bin/evolution.rs) | — |
| SNN — импульсные нейросети (`snn`) | [CPU](python/cpu/snn.py) · [CUDA](python/cuda/snn.py) · [MPS](python/mps/snn.py) | [snn](rust-burn/src/bin/snn.rs) | — |
| QNN — квантовые нейронные модели (`quantum`) | [CPU](python/cpu/quantum.py) · [CUDA](python/cuda/quantum.py) · [MPS](python/mps/quantum.py) | [quantum](rust-burn/src/bin/quantum.rs) | — |
| Гиперразмерные представления и VSA (`hdc`) | [CPU](python/cpu/hdc.py) · [CUDA](python/cuda/hdc.py) · [MPS](python/mps/hdc.py) | [hdc](rust-burn/src/bin/hdc.rs) | — |
| Обучение по правильным ответам (`supervised`) | [CPU](python/cpu/supervised.py) · [CUDA](python/cuda/supervised.py) · [MPS](python/mps/supervised.py) | [supervised](rust-burn/src/bin/supervised.rs) | [CPU](mlx/cpu/supervised.py) · [METAL](mlx/metal/supervised.py) |
| Поиск структуры без готовых ответов (`unsupervised`) | [CPU](python/cpu/unsupervised.py) · [CUDA](python/cuda/unsupervised.py) · [MPS](python/mps/unsupervised.py) | [unsupervised](rust-burn/src/bin/unsupervised.rs) | — |
| Самообучение по заданиям из данных (`selfsupervised`) | [CPU](python/cpu/selfsupervised.py) · [CUDA](python/cuda/selfsupervised.py) · [MPS](python/mps/selfsupervised.py) | [selfsupervised](rust-burn/src/bin/selfsupervised.rs) | — |
| Немного разметки и много неразмеченных данных (`semisupervised`) | [CPU](python/cpu/semisupervised.py) · [CUDA](python/cuda/semisupervised.py) · [MPS](python/mps/semisupervised.py) | [semisupervised](rust-burn/src/bin/semisupervised.rs) | — |
| Контрастивное обучение — понять похожесть (`contrastive`) | [CPU](python/cpu/contrastive.py) · [CUDA](python/cuda/contrastive.py) · [MPS](python/mps/contrastive.py) | [contrastive](rust-burn/src/bin/contrastive.rs) | — |
| Скрытые части и обучение у собственной копии (`masked`) | [CPU](python/cpu/masked.py) · [CUDA](python/cuda/masked.py) · [MPS](python/mps/masked.py) | [masked](rust-burn/src/bin/masked.rs) | — |
| Обучение модели с нуля (`scratch`) | [CPU](python/cpu/scratch.py) · [CUDA](python/cuda/scratch.py) · [MPS](python/mps/scratch.py) | [scratch](rust-burn/src/bin/scratch.rs) | [CPU](mlx/cpu/scratch.py) · [METAL](mlx/metal/scratch.py) |
| Перенос навыков и дообучение (`transfer`) | [CPU](python/cpu/transfer.py) · [CUDA](python/cuda/transfer.py) · [MPS](python/mps/transfer.py) | [transfer](rust-burn/src/bin/transfer.rs) | — |
| LoRA и QLoRA — небольшие добавки к готовой сети (`lora`) | [CPU](python/cpu/lora.py) · [CUDA](python/cuda/lora.py) · [MPS](python/mps/lora.py) | [lora](rust-burn/src/bin/lora.rs) | [CPU](mlx/cpu/lora.py) · [METAL](mlx/metal/lora.py) |
| Новые условия и несколько задач (`domain`) | [CPU](python/cpu/domain.py) · [CUDA](python/cuda/domain.py) · [MPS](python/mps/domain.py) | [domain](rust-burn/src/bin/domain.rs) | — |
| Обучение быстрому освоению новых задач (`meta`) | [CPU](python/cpu/meta.py) · [CUDA](python/cuda/meta.py) · [MPS](python/mps/meta.py) | [meta](rust-burn/src/bin/meta.rs) | — |
| Обучение на примерах внутри запроса (`icl`) | [CPU](python/cpu/icl.py) · [CUDA](python/cuda/icl.py) · [MPS](python/mps/icl.py) | [icl](rust-burn/src/bin/icl.rs) | — |
| Хорошие ответы и предпочтения: SFT, DPO, RLHF (`preferences`) | [CPU](python/cpu/preferences.py) · [CUDA](python/cuda/preferences.py) · [MPS](python/mps/preferences.py) | [preferences](rust-burn/src/bin/preferences.rs) | — |
| Бандиты — выбор варианта с изучением результата (`bandits`) | [CPU](python/cpu/bandits.py) · [CUDA](python/cuda/bandits.py) | [bandits](rust-burn/src/bin/bandits.rs) | — |
| RL — обучение действиям по награде (`rl`) | [CPU](python/cpu/rl.py) · [CUDA](python/cuda/rl.py) · [MPS](python/mps/rl.py) | [rl](rust-burn/src/bin/rl.rs) | — |
| PPO и SAC — обучение правилу выбора действий (`policy`) | [CPU](python/cpu/policy.py) · [CUDA](python/cuda/policy.py) · [MPS](python/mps/policy.py) | [policy](rust-burn/src/bin/policy.rs) | — |
| Подражание действиям и обучение по записям (`imitation`) | [CPU](python/cpu/imitation.py) · [CUDA](python/cuda/imitation.py) · [MPS](python/mps/imitation.py) | [imitation](rust-burn/src/bin/imitation.rs) | — |
| Модель мира и планирование с её помощью (`world`) | [CPU](python/cpu/world.py) · [CUDA](python/cuda/world.py) · [MPS](python/mps/world.py) | [world](rust-burn/src/bin/world.rs) | — |
| Обучение в игре с собой и другими агентами (`selfplay`) | [CPU](python/cpu/selfplay.py) · [CUDA](python/cuda/selfplay.py) · [MPS](python/mps/selfplay.py) | [selfplay](rust-burn/src/bin/selfplay.rs) | — |
| Активное обучение — выбрать, что разметить (`active`) | [CPU](python/cpu/active.py) · [CUDA](python/cuda/active.py) · [MPS](python/mps/active.py) | [active](rust-burn/src/bin/active.rs) | — |
| Обучение по мере поступления данных (`continual`) | [CPU](python/cpu/continual.py) · [CUDA](python/cuda/continual.py) · [MPS](python/mps/continual.py) | [continual](rust-burn/src/bin/continual.rs) | — |
| Совместное обучение без сбора всех исходных данных (`federated`) | [CPU](python/cpu/federated.py) · [CUDA](python/cuda/federated.py) · [MPS](python/mps/federated.py) | [federated](rust-burn/src/bin/federated.rs) | — |
| Порядок обучения и расширение примеров (`curriculum`) | [CPU](python/cpu/curriculum.py) · [CUDA](python/cuda/curriculum.py) · [MPS](python/mps/curriculum.py) | [curriculum](rust-burn/src/bin/curriculum.rs) | — |
| AutoML — автоматический подбор модели (`automl`) | [CPU](python/cpu/automl.py) · [CUDA](python/cuda/automl.py) · [MPS](python/mps/automl.py) | [automl](rust-burn/src/bin/automl.rs) | — |
| Маленькая модель по устройству (`compactarch`) | [CPU](python/cpu/compactarch.py) · [CUDA](python/cuda/compactarch.py) · [MPS](python/mps/compactarch.py) | [compactarch](rust-burn/src/bin/compactarch.rs) | [CPU](mlx/cpu/compactarch.py) · [METAL](mlx/metal/compactarch.py) |
| Дистилляция — обучение у другой модели (`distill`) | [CPU](python/cpu/distill.py) · [CUDA](python/cuda/distill.py) · [MPS](python/mps/distill.py) | [distill](rust-burn/src/bin/distill.rs) | — |
| Квантование — хранить числа компактнее (`quant`) | [CPU](python/cpu/quant.py) · [CUDA](python/cuda/quant.py) · [MPS](python/mps/quant.py) | [quant](rust-burn/src/bin/quant.rs) | — |
| Прореживание — удаление частей сети (`pruning`) | [CPU](python/cpu/pruning.py) · [CUDA](python/cuda/pruning.py) · [MPS](python/mps/pruning.py) | [pruning](rust-burn/src/bin/pruning.rs) | — |
| Совместное использование и разложение параметров (`factorweights`) | [CPU](python/cpu/factorweights.py) · [CUDA](python/cuda/factorweights.py) | [factorweights](rust-burn/src/bin/factorweights.rs) | [CPU](mlx/cpu/factorweights.py) |
| Ускорение выполнения без переобучения (`cache`) | [CPU](python/cpu/cache.py) · [CUDA](python/cuda/cache.py) · [MPS](python/mps/cache.py) | [cache](rust-burn/src/bin/cache.rs) | [CPU](mlx/cpu/cache.py) · [METAL](mlx/metal/cache.py) |
| Бинарные и тернарные сети (`binary`) | [CPU](python/cpu/binary.py) · [CUDA](python/cuda/binary.py) · [MPS](python/mps/binary.py) | [binary](rust-burn/src/bin/binary.rs) | — |
| TinyML — модель на маленьком устройстве (`tinyml`) | [CPU](python/cpu/tinyml.py) · [CUDA](python/cuda/tinyml.py) · [MPS](python/mps/tinyml.py) | [tinyml](rust-burn/src/bin/tinyml.rs) | [CPU](mlx/cpu/tinyml.py) · [METAL](mlx/metal/tinyml.py) |
| Обучение и запуск на Mac (`mac`) | [CPU](python/cpu/mac.py) · [CUDA](python/cuda/mac.py) · [MPS](python/mps/mac.py) | [mac](rust-burn/src/bin/mac.rs) | [CPU](mlx/cpu/mac.py) · [METAL](mlx/metal/mac.py) |
| Перенос модели в приложение (`export`) | [CPU](python/cpu/export.py) · [CUDA](python/cuda/export.py) · [MPS](python/mps/export.py) | [export](rust-burn/src/bin/export.rs) | — |
| Версии, повторяемость и сопровождение (`repro`) | [CPU](python/cpu/repro.py) · [CUDA](python/cuda/repro.py) · [MPS](python/mps/repro.py) | [repro](rust-burn/src/bin/repro.rs) | — |
| RAG — ответ с поиском по документам (`rag`) | [CPU](python/cpu/rag.py) · [CUDA](python/cuda/rag.py) · [MPS](python/mps/rag.py) | [rag](rust-burn/src/bin/rag.rs) | — |
| Агенты — модель с доступными действиями (`agents`) | [CPU](python/cpu/agents.py) · [CUDA](python/cuda/agents.py) · [MPS](python/mps/agents.py) | [agents](rust-burn/src/bin/agents.rs) | — |
| Правила, модель и расчёт вместе (`hybrids`) | [CPU](python/cpu/hybrids.py) · [CUDA](python/cuda/hybrids.py) · [MPS](python/mps/hybrids.py) | [hybrids](rust-burn/src/bin/hybrids.rs) | — |
| Граф знаний и структурированная память (`kg`) | [CPU](python/cpu/kg.py) · [CUDA](python/cuda/kg.py) · [MPS](python/mps/kg.py) | [kg](rust-burn/src/bin/kg.rs) | — |
| Векторы, матрицы и тензоры — наборы чисел (`arrays`) | [CPU](python/cpu/arrays.py) · [CUDA](python/cuda/arrays.py) · [MPS](python/mps/arrays.py) | [arrays](rust-burn/src/bin/arrays.rs) | [CPU](mlx/cpu/arrays.py) · [METAL](mlx/metal/arrays.py) |
| Градиент — куда менять параметры (`derivatives`) | [CPU](python/cpu/derivatives.py) · [CUDA](python/cuda/derivatives.py) · [MPS](python/mps/derivatives.py) | [derivatives](rust-burn/src/bin/derivatives.rs) | [CPU](mlx/cpu/derivatives.py) · [METAL](mlx/metal/derivatives.py) |
| Вероятность и статистика — работа с неопределённостью (`probability`) | [CPU](python/cpu/probability.py) · [CUDA](python/cuda/probability.py) | [probability](rust-burn/src/bin/probability.rs) | [CPU](mlx/cpu/probability.py) · [METAL](mlx/metal/probability.py) |
| Ошибка обучения и её уменьшение (`optimization`) | [CPU](python/cpu/optimization.py) · [CUDA](python/cuda/optimization.py) · [MPS](python/mps/optimization.py) | [optimization](rust-burn/src/bin/optimization.rs) | [CPU](mlx/cpu/optimization.py) · [METAL](mlx/metal/optimization.py) |
| Подготовка данных и правильных ответов (`data`) | [CPU](python/cpu/data.py) · [CUDA](python/cuda/data.py) · [MPS](python/mps/data.py) | [data](rust-burn/src/bin/data.rs) | — |
| Обучение, настройка и независимая проверка (`split`) | [CPU](python/cpu/split.py) · [CUDA](python/cuda/split.py) · [MPS](python/mps/split.py) | [split](rust-burn/src/bin/split.rs) | — |
| Как измерять качество и с чем сравнивать (`metrics`) | [CPU](python/cpu/metrics.py) · [CUDA](python/cuda/metrics.py) · [MPS](python/mps/metrics.py) | [metrics](rust-burn/src/bin/metrics.rs) | — |
| Новые данные и изменение условий (`generalization`) | [CPU](python/cpu/generalization.py) · [CUDA](python/cuda/generalization.py) · [MPS](python/mps/generalization.py) | [generalization](rust-burn/src/bin/generalization.rs) | — |
| Уверенность, диапазон ответа и отказ (`uncertainty`) | [CPU](python/cpu/uncertainty.py) · [CUDA](python/cuda/uncertainty.py) · [MPS](python/mps/uncertainty.py) | [uncertainty](rust-burn/src/bin/uncertainty.rs) | — |
| Разбор ошибок и объяснение решений (`explain`) | [CPU](python/cpu/explain.py) · [CUDA](python/cuda/explain.py) · [MPS](python/mps/explain.py) | [explain](rust-burn/src/bin/explain.rs) | — |
| Понимание команд на естественном языке (`lab-text`) | [CPU](python/cpu/lab-text.py) · [CUDA](python/cuda/lab-text.py) · [MPS](python/mps/lab-text.py) | [lab-text](rust-burn/src/bin/lab-text.rs) | — |
| Распознавание изображений и дефектов (`lab-vision`) | [CPU](python/cpu/lab-vision.py) · [CUDA](python/cuda/lab-vision.py) · [MPS](python/mps/lab-vision.py) | [lab-vision](rust-burn/src/bin/lab-vision.rs) | [CPU](mlx/cpu/lab-vision.py) · [METAL](mlx/metal/lab-vision.py) |
| Распознавание режимов по сигналам (`lab-signal`) | [CPU](python/cpu/lab-signal.py) · [CUDA](python/cuda/lab-signal.py) · [MPS](python/mps/lab-signal.py) | [lab-signal](rust-burn/src/bin/lab-signal.rs) | — |
| Прогноз по производственной таблице (`lab-table`) | [CPU](python/cpu/lab-table.py) · [CUDA](python/cuda/lab-table.py) · [MPS](python/mps/lab-table.py) | [lab-table](rust-burn/src/bin/lab-table.rs) | [CPU](mlx/cpu/lab-table.py) · [METAL](mlx/metal/lab-table.py) |
| Поиск похожих документов и объектов (`lab-search`) | [CPU](python/cpu/lab-search.py) · [CUDA](python/cuda/lab-search.py) · [MPS](python/mps/lab-search.py) | [lab-search](rust-burn/src/bin/lab-search.rs) | — |
| Обучение игрового агента (`lab-game`) | [CPU](python/cpu/lab-game.py) · [CUDA](python/cuda/lab-game.py) · [MPS](python/mps/lab-game.py) | [lab-game](rust-burn/src/bin/lab-game.rs) | — |
| Восстановление физической зависимости (`lab-science`) | [CPU](python/cpu/lab-science.py) · [CUDA](python/cuda/lab-science.py) · [MPS](python/mps/lab-science.py) | [lab-science](rust-burn/src/bin/lab-science.rs) | — |
| Генерация небольших изображений и сигналов (`lab-generative`) | [CPU](python/cpu/lab-generative.py) · [CUDA](python/cuda/lab-generative.py) · [MPS](python/mps/lab-generative.py) | [lab-generative](rust-burn/src/bin/lab-generative.rs) | — |
| Игровой агент: Поле N×M (`lab-game-grid`) | [CPU](python/cpu/lab-game-grid.py) · [CUDA](python/cuda/lab-game-grid.py) · [MPS](python/mps/lab-game-grid.py) | [lab-game-grid](rust-burn/src/bin/lab-game-grid.rs) | — |
| Игровой агент: Стены и предметы (`lab-game-items`) | [CPU](python/cpu/lab-game-items.py) · [CUDA](python/cuda/lab-game-items.py) · [MPS](python/mps/lab-game-items.py) | [lab-game-items](rust-burn/src/bin/lab-game-items.rs) | — |
