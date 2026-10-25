# Задание 1. Автоматизация построения признаков в рекомендательных системах

Исследование: можно ли автоматизировать построение признаков для рекомендательной
системы и не потерять в качестве относительно сильного градиентного бустинга на
ручных признаках.

## Основные результаты

Единый временной протокол на Yambda-500M, NDCG@10:

| Подход | Признаков | val | test |
|---|---:|---:|---:|
| ручные признаки | 99 | 0.9159 | 0.9155 |
| автоматика: весь пул | 187 | 0.9142 | 0.9139 |
| ручные + автоматика (union) | 201 | **0.9176** | **0.9174** |
| quota + LLM-агент (конкуренция) | 198 | 0.9176 | 0.9174 |

- union к baseline: **+0.0019**, 95% CI **[0.0008, 0.0030]** — значимо;
- auto_all к baseline: **−0.0016**, 95% CI **[−0.0029, −0.0003]** — значимо в минус;
- top-40 по gain: 0.9102 val;
- второй сэмпл (3000 пользователей): baseline 0.9112, union 0.9126,
  +0.0014, 95% CI [−0.0006, +0.0034] — направление сохраняется, но не значимо;
- KION: перенос сетки не даёт значимого прироста (0.5194 → 0.5201, p = 0.28).

Вывод: полный автоматический набор близок к ручному, но статистически значимо
уступает ему; значимый выигрыш даёт только объединение ручных и автоматических
признаков. LLM-агент полезен как генератор гипотез: аддитивно он улучшает слабую
модель (`auto_base` 0.7729 → 0.7877), но в конкуренции за квоты не меняет
сильный union.

Итоговый отчёт: `reports/report_submission.pdf`.

## Данные

Основной датасет — Yambda-500M (Яндекс Музыка). Берутся 10 000 пользователей,
вся история сохраняется; всего 48.2M событий и 905 759 треков. KION используется
как кросс-доменная проверка: 10 000 пользователей, 381k событий, без повторов.

- задача: next-week listening — прослушает ли пользователь трек в течение 7 дней
  после cutoff;
- кандидаты: все позитивы горизонта + негативы из популярных объектов;
- split: 4 недельных train-снимка → 7 дней gap → val → 7 дней gap → test;
- признаки считаются только по `ts < cutoff`;
- seed = 42; кандидаты детерминированы (сортировка позитивов до `sample`,
  tie-break пула негативов по `item_id`).

## Воспроизведение

### 0. Окружение

```bash
cd task1
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,agent]"
```

Расчёты выполнены на CPU (32 ядра, 128 GB); GPU не требуется. Для LLM-агента
нужен `OPENROUTER_API_KEY` в `.env` (шаблон — `.env.example`). Ключ в репозиторий
не коммитится.

### 1. Данные

```bash
python scripts/01_download_data.py
python scripts/02_prepare_data.py
python scripts/03_build_dataset.py --split all
```

`03_build_dataset.py` создаёт кандидатов и ручные признаки для шести снимков.
Если менялись события или логика кандидатов, пересоберите датасеты и
материализуйте auto-признаки заново с `--force` (см. ниже).

### 2. Ручной baseline

```bash
python scripts/04_train_baseline.py \
    --train-snapshots train_w0,train_w1,train_w2,train_w3
```

Артефакты: `experiments/baseline_manual/` (модель, метрики, скоры).

### 3. Автоматический поиск и union

```bash
python scripts/06_auto_search.py --stage screen --tag quota --per-entity 20
python scripts/06_auto_search.py --stage materialize --tag quota --force
python scripts/06_auto_search.py --stage train --tag quota \
    --derived-top-k 50 --no-ablate
python scripts/06_auto_search.py --stage all-features --tag quota
python scripts/22_eval_auto_all.py --split test
```

`stage screen` считает 900 спецификаций на 1000 пользователях val; `materialize`
материализует отобранные признаки для всех снимков; `stage train` обучает
auto_base, auto_rel и union. `stage all-features` обучает auto_all на всём пуле
(187 признаков) и top-40 по gain. `22_eval_auto_all.py` считает test для auto_all.

### 4. LLM-агент

Агент предлагает спецификации DSL, а не SQL. Для нового пула:

```bash
python scripts/09_llm_agent.py --tag quota --iterations 18 --batch-size 8 \
    --sample-users 200 --novel-only --reset

python scripts/06_auto_search.py --stage screen --tag agent_quota --per-entity 20 \
    --extra-pool experiments/auto_search/agent_pool_quota.json
python scripts/06_auto_search.py --stage materialize --tag agent_quota --force
python scripts/06_auto_search.py --stage train --tag agent_quota \
    --derived-top-k 50 --no-ablate

python scripts/11_agent_additive_selection.py --top-k 20
python scripts/06_auto_search.py --stage materialize --tag agent_add --force
python scripts/06_auto_search.py --stage train --tag agent_add \
    --derived-top-k 50 --no-ablate
```

Если пул `agent_pool_quota.json` уже есть в artifacts, шаг `09_llm_agent.py`
можно пропустить. Стоимость одного прогона агента — порядка $0.02–0.05.

### 5. Расширенный DSL (опционально)

```bash
python scripts/09_llm_agent.py --tag expr_qwen --expr --novel-only --reset \
    --iterations 20 --batch-size 7 --sample-users 200 --max-propose-calls 14
python scripts/13_train_expr_pool.py --pool expr_qwen --top-k 20 \
    --tag expr_add --force
```

Расширение добавляет условия внутри агрегата (`complete`, `skip`, `non_organic`),
`decay_count` с half-life и выражения `ratio`/`delta`/`product`. В основной
результат оно не входит: на сильной модели прироста над union с базовым DSL нет.

### 6. Аудит и значимость

```bash
python scripts/15_audit_leakage.py
python scripts/16_significance.py --bootstrap 2000
```

Аудит выполняет 9 проверок на данных: зазоры сплита, окна снимков, совпадение
позитивов с событиями окна label, отсутствие label-дублей, воспроизведение
признака независимым SQL, отсутствие влияния будущего, отпечатки данных.
`16_significance.py` делает paired bootstrap по пользователям; для expr-ветки
нужна обученная `expr_add`-модель.

### 7. Второй сэмпл пользователей

```bash
python scripts/02_prepare_data.py --config configs/holdout.yaml
python scripts/03_build_dataset.py --config configs/holdout.yaml --split all
python scripts/04_train_baseline.py --config configs/holdout.yaml \
    --train-snapshots train_w0,train_w1,train_w2,train_w3
python scripts/18_holdout_eval.py --config configs/holdout.yaml
```

`18_holdout_eval.py` переносит feature-set без переподбора, обучает baseline,
union и expr-union на 3000 дизъюнктных пользователях и считает test.

### 8. Кросс-доменная проверка KION

```bash
python scripts/02_prepare_data.py --config configs/kion.yaml
python scripts/03_build_dataset.py --config configs/kion.yaml --split all
python scripts/18_holdout_eval.py --config configs/kion.yaml --prefix kion
python scripts/20_kion_report.py
```

KION — разреженный кино-датасет без повторов; здесь проверяется граница
применимости метода, а не перенос качества на плотную историю.

### 9. Отчёт, рисунки и тесты

```bash
python scripts/07_make_auto_report.py --tag quota
python scripts/05_make_report_figures.py
typst compile reports/report_submission.typ reports/report_submission.pdf
pytest -q
```

Тесты: **37** (агрегаты, кандидаты, сплиты, метрики, DSL, агент, manual-recency,
детерминизм кандидатов). Артефакты экспериментов — в `experiments/`, таблицы —
в `reports/tables/`, рисунки — в `reports/figures/`.

## Ожидаемое время

На CPU 32 ядра/128 GB: подготовка и сборка данных — ~10 минут; baseline — ~10–15
минут; quota union — ~30–40 минут; agent-ветка — ~40–60 минут; holdout — ~10
минут; KION — ~5 минут. Полный цикл с агентом и extended DSL занимает порядка
2 часов; LLM-агент добавляет сетевые запросы и небольшую стоимость.

## Структура

```
task1/
├── configs/                 # base.yaml, holdout.yaml, kion.yaml
├── src/recsys_auto/
│   ├── data/                # загрузка, подготовка, сплиты, кандидаты
│   ├── features/            # DSL, ручные и SQL-агрегаты
│   ├── auto_features/       # генератор, скрининг, отбор, выражения
│   ├── agent/               # LLM-агент и валидация спецификаций
│   ├── models/              # LightGBM
│   └── eval/                # метрики и протокол ранжирования
├── scripts/                 # шаги пайплайна 01–22
├── tests/                   # 37 unit-тестов
├── experiments/             # модели и метрики запусков
└── reports/                 # report_submission.typ/pdf, таблицы, рисунки
```
