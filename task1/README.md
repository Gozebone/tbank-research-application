# Задание 1. Автоматизация построения признаков в рекомендательных системах

## Актуальные результаты

После исправления recency/tenure-признаков и детерминированной сборки кандидатов:

| Подход | Признаков | NDCG@10 val | NDCG@10 test |
|---|---:|---:|---:|
| ручные признаки | 99 | 0.9159 | 0.9155 |
| автоматика: весь пул | 187 | 0.9142 | 0.9139 |
| ручные + автоматика (union) | 201 | **0.9176** | **0.9174** |

- union к baseline: **+0.0019**, 95% CI **[0.0008, 0.0030]** — значимо;
- auto_all к baseline: **−0.0016**, 95% CI **[−0.0029, −0.0003]** — тоже значимо, но в минус;
- top-40 по gain: 0.9102 val;
- KION: перенос сетки не даёт значимого прироста (0.5194 vs 0.5194).

Вывод: автоматика почти догоняет человека, но **не заменяет** его; выигрыш даёт
только объединение ручных и автоматических признаков.

> Таблицы ниже по агенту и расширенному DSL — результаты предыдущей версии
> сэмпла; они требуют финального пересчёта после исправления recency. Основные
> выводы и числа — в таблице выше и в `reports/report_submission.pdf`.

## Пайплайн

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,agent]"

python scripts/01_download_data.py          # скачать Yambda и KION
python scripts/02_prepare_data.py           # нормализовать события, сэмплировать пользователей
python scripts/03_build_dataset.py --split all   # кандидаты + признаки для 6 снимков
python scripts/04_train_baseline.py --train-snapshots train_w0,train_w1,train_w2,train_w3

# автоматический поиск: наивный top-k
python scripts/06_auto_search.py --stage screen --tag topk --top-k 80
python scripts/06_auto_search.py --stage materialize --tag topk
python scripts/06_auto_search.py --stage train --tag topk

# автоматический поиск: квоты по сущностям
python scripts/06_auto_search.py --stage screen --tag quota --per-entity 20
python scripts/06_auto_search.py --stage materialize --tag quota
python scripts/06_auto_search.py --stage train --tag quota --derived-top-k 50 --no-ablate
python scripts/06_auto_search.py --stage all-features --tag quota
python scripts/22_eval_auto_all.py --split test
python scripts/08_greedy_selection.py --tag quota

# LLM-агент: признаки вне фиксированной сетки (нужен OPENROUTER_API_KEY в .env)
python scripts/09_llm_agent.py --tag quota --iterations 18 --batch-size 8 \
    --sample-users 200 --novel-only --reset
python scripts/06_auto_search.py --stage screen --tag agent_quota --per-entity 20 \
    --extra-pool experiments/auto_search/agent_pool_quota.json
python scripts/06_auto_search.py --stage materialize --tag agent_quota
python scripts/06_auto_search.py 
### LLM-агент

Агент предлагает не SQL, а спецификации DSL и проверяется тем же скринингом:

| Вариант | Признаков | val NDCG@10 | test NDCG@10 |
|---|---:|---:|---:|
| quota (сетка) | 201 | **0.9176** | **0.9174** |
| auto_base без агента | 55 | 0.7729 | — |

Числа для вариантов с LLM-агентом и расширенным DSL пересчитываются на
исправленном сэмпле; в этой версии README они не приводятся, чтобы не
смешивать старый и новый протокол.

Агент находит реальный пробел сетки — короткие окна популярности объекта
(5/10/21 день): на слабой модели это даёт **+0.013** NDCG@10, на сильной
union остаётся в пределах шума. Три модели (Qwen, DeepSeek, Kimi) дают почти
непересекающиеся пулы (Jaccard 0.07–0.16) и разброс выживаемости 0–13%, но
потолок не меняют.

Расширенный DSL (условия `complete`/`skip`/`non_organic`, `decay_count` с
half-life и выражения `ratio`/`delta`/`product`) поднимает выживаемость пула
агента и даёт прирост на слабой модели; на сильном union он не двигает
потолок. Точные числа пересчитываются после исправления recency.

Прирост union к baseline **статистически значим**: paired bootstrap по 5547
пользователям даёт **+0.0019, 95% CI [0.0008, 0.0030]**. При этом auto_all
значимо хуже baseline: **−0.0016, 95% CI [−0.0029, −0.0003]**.

**Проверка на втором сэмпле.** Дизъюнктные 3000 пользователей пересчитываются
после исправления recency; старые числа этой версии не используются, чтобы не
смешивать протоколы.

**KION (другой домен).** На разреженном кино-датасете (381k событий, 10k
пользователей, без повторов) перенос набора признаков не даёт значимого
прироста: +0.0007 (p = 0.28). Это граница применимости метода: нужна плотная
история и повторные взаимодействия.

Итоговый отчёт: `reports/report_submission.pdf` (Typst,
`reports/report_submission.typ`). Автоматический поиск: 900 сгенерированных
спецификаций, скрининг по AUC внутри пользователя, несколько стратегий отбора
(top-k, квоты, gain, greedy) и шаблонные производные отношения.

## Статус

- [x] Выбор датасета и постановка задачи
- [x] Структура проекта
- [x] Скачивание данных (Yambda-500M + KION)
- [x] Подготовка данных и time-split
- [x] Базовый пайплайн с ручными признаками (99 признаков, LightGBM)
- [x] Автоматическая генерация и отбор признаков (900 спецификаций, 2 стратегии)
- [x] LLM-агент, предлагающий признаки вне фиксированной сетки
- [x] Расширенный DSL: условия, затухание и бинарные выражения; сравнение на 3 моделях
- [x] Аудит утечек, bootstrap-интервалы, сегментный анализ, подбор гиперпараметров
- [x] Проверка на втором дизъюнктном сэмпле пользователей
- [x] Кросс-датасетная проверка на KION
- [x] Typst-отчёт (elsearticle) и PDF
- [x] Сравнение и анализ (val/test, вычислительные затраты)
- [x] Итоговый отчёт (`reports/README.md`)--stage train --tag agent_quota --derived-top-k 50 --no-ablate

# аддитивный вариант: 100 сеточных + 20 агентских (без конкуренции за квоты)
python scripts/11_agent_additive_selection.py --top-k 20
python scripts/06_auto_search.py --stage materialize --tag agent_add
python scripts/06_auto_search.py --stage train --tag agent_add --derived-top-k 50 --no-ablate

# сравнение пулов разных LLM (опционально)
python scripts/09_llm_agent.py --tag ab_deepseek --model deepseek/deepseek-chat-v3-0324 \
    --price-in 0.25 --price-out 1.00 --novel-only --reset --iterations 18 --sample-users 200
python scripts/09_llm_agent.py --tag ab_kimi --model moonshotai/kimi-k2 \
    --price-in 0.14 --price-out 0.60 --novel-only --reset --iterations 18 --sample-users 200
python scripts/12_compare_agents.py --pools quota,ab_deepseek,ab_kimi

python scripts/10_make_agent_report.py --agent-tag quota --train-tag agent_quota --add-tag agent_add

# аудит утечек, значимость различий, подбор гиперпараметров
python scripts/15_audit_leakage.py
python scripts/16_significance.py --bootstrap 2000
python scripts/17_tune_hyperparams.py --tag quota

# второй дизъюнктный сэмпл (3000 других пользователей)
python scripts/02_prepare_data.py --config configs/holdout.yaml
python scripts/03_build_dataset.py --config configs/holdout.yaml --split all
python scripts/04_train_baseline.py --config configs/holdout.yaml \
    --train-snapshots train_w0,train_w1,train_w2,train_w3
python scripts/18_holdout_eval.py --config configs/holdout.yaml
python scripts/19_make_holdout_report.py

# расширенный DSL: условия, затухание и бинарные выражения
python scripts/09_llm_agent.py --tag expr_qwen --expr --novel-only --reset \
    --iterations 20 --batch-size 7 --sample-users 200 --max-propose-calls 14
python scripts/09_llm_agent.py --tag expr_deepseek --expr --novel-only --reset \
    --model deepseek/deepseek-chat-v3-0324 --price-in 0.25 --price-out 1.00 \
    --iterations 20 --batch-size 7 --sample-users 200 --max-propose-calls 14
python scripts/09_llm_agent.py --tag expr_kimi --expr --novel-only --reset \
    --model moonshotai/kimi-k2 --price-in 0.14 --price-out 0.60 \
    --iterations 20 --batch-size 7 --sample-users 200 --max-propose-calls 14
python scripts/12_compare_agents.py --pools expr_qwen,expr_deepseek,expr_kimi \
    --out agent_model_ab_expr.csv
python scripts/13_train_expr_pool.py --pool expr_qwen --top-k 20 --tag expr_add

python scripts/07_make_auto_report.py --tag quota
python scripts/05_make_report_figures.py
pytest -q
```


