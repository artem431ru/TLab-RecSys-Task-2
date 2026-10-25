"""Results report generator; does not fabricate measurements."""



def render_report(args, coverage, reward_metrics, metrics, rollouts, traces, runtime, confidence):
    def table(d):
        out = ['| Модель | Recall@10 | NDCG@10 | Покрытие каталога@10 | RM reward@top1 |',
               '|---|---:|---:|---:|---:|']
        for name, m in d.items():
            out.append(f'| {name} | {m["recall@10"]:.5f} | {m["ndcg@10"]:.5f} | '
                       f'{m["catalog_coverage@10"]:.5f} | {m["predicted_reward_top1"]:.5f} |')
        return '\n'.join(out)
    def rmrow(m):
        return f'MAE={m.get("mae", float("nan")):.4f}, RMSE={m.get("rmse", float("nan")):.4f}, n={m.get("n", 0)}'
    baseline = metrics['test']['GPT-2 SFT']
    changes = []
    for name in ['RL: raw watched_pct', 'RL: novelty penalty']:
        met = metrics['test'][name]
        changes.append(f'- **{name}:** ΔNDCG@10 = {met["ndcg@10"]-baseline["ndcg@10"]:+.5f}; '
                       f'ΔRecall@10 = {met["recall@10"]-baseline["recall@10"]:+.5f}; '
                       f'ΔRM reward = {met["predicted_reward_top1"]-baseline["predicted_reward_top1"]:+.5f}.')
    uncertainty = ['| Модель | ΔRecall@10 [95% ДИ] | ΔNDCG@10 [95% ДИ] | ΔRM reward [95% ДИ] | ΔPopularity@10 [95% ДИ] |',
                   '|---|---:|---:|---:|---:|']
    for name, data in confidence.items():
        def fmt(f):
            v = data[f]
            return f'{v["delta"]:+.4f} [{v["ci95_lower"]:+.4f}; {v["ci95_upper"]:+.4f}]'
        uncertainty.append(f'| {name} | {fmt("hit@10")} | {fmt("ndcg@10")} | '
                           f'{fmt("predicted_reward_top1")} | {fmt("mean_train_popularity@10")} |')
    conclusions = ('Повышение модельной награды не доказывает рост удовлетворённости пользователей. '
                   'Выводы ограничены офлайн-наблюдениями и малой экспериментальной выборкой.')
    report = f'''# Отчёт: RL-дообучение GPT-2 для последовательных рекомендаций KION

## Постановка

Задача: сравнить next-item модель GPT-2 до/после RL, оценивая релевантность и прогнозируемую долю досмотра. Исследование проведено на **реальном KION** (не синтетических данных). В качестве токенов используются ID фильмов/сериалов.

## Данные и протокол

- Разбиение по `last_watch_dt`: train до **26.07.2021**, validation с 26.07 по 08.08 включительно, test с **09.08.2021**.
- Словарь из {coverage['catalog_size']} самых частых объектов **только train**. Выборка {coverage['selected_users']} пользователей с ≥3 train-событиями, случайная с фиксированным seed. Отбор пользователей не использует будущее.
- Train/validation/test примеров внутри выбранного каталога: {coverage['examples']['train']} / {coverage['examples']['val']} / {coverage['examples']['test']}.
- Истории учитывают лишь фактически наблюдавшиеся события, предшествующие целевому по дате. На test доступны известные реальные события validation как история, но обучение на них не проводится.
- Метрики рекомендаций вычислены **только на событиях из top-{coverage['catalog_size']} train-каталога**, а не на полном KION. Невидимые объекты каталога не могут быть угаданы, поэтому полученные числа нельзя напрямую сравнивать с full-catalog RecSys.
- `last_watch_dt` в KION представляет агрегированную дату последнего просмотра для пары user-item. Это приближённая хронология, а не полный журнал сессий; при равных датах реальный порядок неизвестен.
- Тестовая выборка не применялась при подборе словаря, обучении и RL. Все настройки зафиксированы заранее. Дубликаты user-item исключаются природой агрегированного датасета.

## Модели

1. **GPT-2 SFT**: GPT2LMHeadModel с 2 Transformer-блоками, 4 attention-heads, hidden=64, контекст до {args.ctx} токенов; задача предсказания следующего item ID по реальной истории. Обучение {args.epochs} эпох, AdamW, LR={args.lr}, CE.
2. **Reward model**: отдельная обучаемая нейросеть с embedding истории и объекта + MLP, sigmoid для прогноза `watched_pct/100` в [0,1]. Обучается {args.rm_epochs} эпох на **наблюдаемых** парах из train. Полезность **ненаблюдавшихся** объектов из этих данных не известна. Результаты валидации: {rmrow(reward_metrics['val'])}; теста: {rmrow(reward_metrics['test'])}. MAE константного среднего train на test: {reward_metrics['test'].get('constant_mean_baseline_mae')}.
3. **RL**: REINFORCE, действие = выбранный item; награда из frozen reward model, штраф KL(π||π_SFT) с коэффициентом {args.kl} и supervised-anchoring (CE) с коэффициентом {args.anchor}. {args.rl_steps} RL-шагов, LR={args.rl_lr}. Запрещены уже просмотренные объекты (mask seen). **Два независимых старта из одной SFT:** reward = predicted watched_pct; reward = predicted watched_pct − {args.novelty_coef} × train popularity, нормированная логарифмом.

## Релевантность и награда: Validation

{table(metrics['val'])}

## Релевантность и награда: Test

{table(metrics['test'])}

Recall@10 и NDCG@10 оцениваются на каждом следующем реально просмотренном объекте. При одном релевантном объекте на запрос Recall@10 совпадает с HitRate@10. `RM reward@top1` — средняя **предсказанная**, а не реальная оценка реакции на top-1 рекомендацию.

### Изменения по сравнению с SFT на Test

{chr(10).join(changes)}

Файлы `results/metrics.json`, `results/rl_trace.json` содержат полный набор результатов, включая coverage каталога и концентрацию популярности.

### Устойчивость различий: парный bootstrap по пользователям

{chr(10).join(uncertainty)}

95% доверительные интервалы вычислены с **2000 повторными выборками пользователей с возвращением**; все запросы выбранного пользователя попадают в одну bootstrap-группу. Это уменьшает ошибку независимости запросов одного пользователя. Если интервал по NDCG включает ноль, данные не дают убедительного свидетельства отличия от SFT. Интервалы **описательные**: они не учитывают случайность обучения, отбора каталога и смещение логирования рекомендаций. Тест остаётся единственным фиксированным holdout; никаких настроек после просмотра теста не подбирали.

`ΔPopularity@10` измеряет изменение средней нормированной log-популярности рекомендованных объектов из train. Отрицательное значение означает больший сдвиг к менее популярным объектам. Это прямая проверка того, действительно ли `novelty penalty` уменьшает популярность, а не только меняет значение surrogate reward. Подробные результаты по запросам в `results/test_query_metrics.csv`, интервалы в `results/test_bootstrap.json`.

## Генерация 5 шагов

Здесь каждую собственную рекомендацию добавляем в историю как будто пользователь её выбрал. Это **симуляция, а не наблюдение**.

| Модель | RM reward шаги 1→5 |
|---|---|
{chr(10).join('| '+k+' | '+', '.join(f'{v:.3f}' for v in vals)+' |' for k,vals in rollouts.items())}

Условия оценки после первого шага изменяются: модель обучалась на настоящих историях, а получает собственные гипотетические действия. В отсутствие пользовательской модели и реальных ответов нельзя интерпретировать пятишаговые значения как доказанное долгосрочное улучшение.

## Компромисс и ограничения

{chr(10).join(changes)}

{conclusions} `watched_pct` может давать смещение к коротким объектам и просмотрам уже вовлечённых пользователей, а learned reward можно эксплуатировать в областях без достоверной разметки. Отсутствуют impression logs и корректные propensity scores. **Контрфактическую награду и retention офлайн подтвердить нельзя.** Для проверки возвращаемости требуются последовательные реальные пользовательские сессии с датами повторных визитов и A/B-тест.

Обратим внимание на концентрацию рекомендаций: покрытие каталога и доля самого популярного item в рекомендациях указаны в `metrics.json`. Низкое покрытие или растущая концентрация после RL свидетельствует о коллапсе разнообразия даже при росте predicted reward.

## Воспроизведение

```powershell
pip install -r requirements.txt
python run_experiment.py --data "../recsys_research/data/interactions.csv" --out results
```

Для запуска с более широкой выборкой: `python run_experiment.py --data <path/to/interactions.csv> --users 15000 --items 1000 --max-train 20000 --max-eval 1000 --epochs 2 --rl-steps 120`.

Случайность фиксирована `seed={args.seed}`. Устройство: автоматический выбор CUDA/CPU, общее время запуска: {runtime:.1f} сек. Получение KION: https://github.com/irsafilo/KION_DATASET. Экспериментальная среда и версии зависимостей — в `requirements.txt`; аппаратная конфигурация влияет на скорость и незначительно на численные результаты.
'''
    return report
