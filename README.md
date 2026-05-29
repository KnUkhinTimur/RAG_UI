# RAG Page Studio

Flask-приложение для генерации HTML/CSS-страниц на основе ваших RAG-таблиц в PostgreSQL:

- `ui_page_structure`
- `ui_brand_style`
- `ui_images`
- `messages`
- `chats`

Что умеет:

- подбирать `top 3` картинок по смыслу;
- учитывать `width` и `height` изображения;
- давать выбор нужного фото перед генерацией;
- генерировать single-file HTML с CSS;
- сохранять историю генераций и preview;
- хранить `selected_image_url` в `messages`.

## 1. Подготовка

Создайте `.env` на основе `.env.example`.

Минимум нужно заполнить:

- `PG_HOST`
- `PG_PORT`
- `PG_DATABASE`
- `PG_USER`
- `PG_PASSWORD`
- `LLM_API_KEY`

Если используете Groq, можно оставить:

- `LLM_PROVIDER=groq`
- `LLM_MODEL=llama-3.3-70b-versatile`

## 2. Установка

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Запуск

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Приложение откроется на:

`http://127.0.0.1:5000`

## 4. Важно про PostgreSQL

Для векторного поиска таблицы `ui_page_structure`, `ui_brand_style`, `ui_images` должны уже существовать и содержать колонку `embedding` типа `vector`.

Проект сам создаёт и обновляет служебные таблицы:

- `chats`
- `messages`

Также он автоматически добавляет колонку `selected_image_url`, если её ещё нет.
