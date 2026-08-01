# ==============================================================================
# 🚀 MLOps Enterprise Makefile (Zero-Config Deployment)
# ==============================================================================

# 1. Експорт змінних середовища (якщо файл існує)
ifneq (,$(wildcard ./.env))
	include .env
	export $(shell awk -F= '/^[a-zA-Z_]/ {print $$1}' .env)
endif

# 2. Кросплатформна підтримка ОС та збір інформації про залізо (OS, Arch, RAM)
ifeq ($(OS),Windows_NT)
	OPEN_CMD := start ""
	SYS_OS_NAME := Windows
	SYS_CPU_ARCH := $(PROCESSOR_ARCHITECTURE)
	SYS_RAM_GB := $(shell powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)")
else
	UNAME_S := $(shell uname -s)
	SYS_CPU_ARCH := $(shell uname -m)
	ifeq ($(UNAME_S),Linux)
		OPEN_CMD := xdg-open
		SYS_OS_NAME := $(shell grep '^PRETTY_NAME=' /etc/os-release | cut -d '"' -f 2 || echo "Linux")
		SYS_RAM_GB := $(shell awk '/MemTotal/ {printf "%.0f", $$2/1024/1024}' /proc/meminfo)
	endif
	ifeq ($(UNAME_S),Darwin)
		OPEN_CMD := open
		SYS_OS_NAME := $(shell sw_vers -productName) $(shell sw_vers -productVersion)
		SYS_RAM_GB := $(shell expr $$(sysctl -n hw.memsize) / 1073741824)
	endif
endif

# 3. Кольори для термінала
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
PURPLE := \033[35m
RESET := \033[0m

# 4. Системні змінні
PY_VER := 3.12
VENV := .venv
# Динамічно шукаємо доступну команду (python3 або python)
PYTHON_CMD := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")
UV := $(PYTHON_CMD) -m uv
MIN_RAM_GB := 8

.PHONY: help check-resources setup clean deep-clean run-hw1 run-hw2 run-hw3 run-hw4 api-hw1 api-hw2 api-hw3 api-hw4

# Дефолтна ціль
help:
	@echo "$(CYAN)================================================================================$(RESET)"
	@echo "$(GREEN)          📊 Applied Machine Learning & MLOps Portfolio by IRONKAGE$(RESET)"
	@echo "$(PURPLE)          🔮 ОС: $(SYS_OS_NAME) | Архітектура ЦП: $(SYS_CPU_ARCH) | ОЗП: $(SYS_RAM_GB) Гб$(RESET)"
	@echo "$(CYAN)================================================================================$(RESET)"
	@echo "$(YELLOW)🛠  Команди інфраструктури:$(RESET)"
	@echo "  $(GREEN)make setup$(RESET)        - 📦 Встановити системні залежності, uv та Python-пакети"
	@echo "  $(GREEN)make clean$(RESET)        - 🧹 Очистити кеш (pycache, ruff) та видалити .venv"
	@echo "  $(GREEN)make deep-clean$(RESET)   - 🧨 ПОВНЕ очищення: видалити середовище та набори даних"
	@echo ""
	@echo "$(YELLOW)📡 Команди запуску проектів (Marimo Notebooks):$(RESET)"
	@echo "  $(CYAN)make run-hw1$(RESET)      - 🏡 ДЗ №1: Лінійна регресія та 3D-картографія Каліфорнії"
	@echo "  $(CYAN)make run-hw2$(RESET)      - 🌦️  ДЗ №2: Логістична регресія та прогноз погоди Австралії"
	@echo "  $(CYAN)make run-hw3$(RESET)      - 💼 ДЗ №3: Складні конвеєри KNN та прогнозування зарплат"
	@echo "  $(CYAN)make run-hw4$(RESET)      - 🚗 ДЗ №4: Дерева рішень, Ансамблі та аналіз важливих ознак"
	@echo ""
	@echo "$(YELLOW)🌐 MLOps: Розгортання Production API (FastAPI):$(RESET)"
	@echo "  $(GREEN)make api-hw1$(RESET)      - 🏡 ДЗ №1: Мікросервіс California Housing на порту 8000"
	@echo "  $(GREEN)make api-hw2$(RESET)      - 🌦️  ДЗ №2: Мікросервіс Rain Classifier на порту 8000"
	@echo "  $(GREEN)make api-hw3$(RESET)      - 💼 ДЗ №3: Мікросервіс Salary Prediction на порту 8000"
	@echo "  $(GREEN)make api-hw4$(RESET)      - 🚗 ДЗ №4: Мікросервіс CarDekho Pricing на порту 8000"
	@echo "$(CYAN)================================================================================$(RESET)"

# ------------------------------------------------------------------------------
# 1. ІНІЦІАЛІЗАЦІЯ ІНФРАСТРУКТУРИ ТА ПЕРЕВІРКА ЗАЛІЗА
# ------------------------------------------------------------------------------
check-resources:
	@echo "$(CYAN)🔍 Аналіз апаратного забезпечення (RAM Validator)...$(RESET)"
	@echo "import platform, subprocess, sys" > .hw_check.py
	@echo "gb = 0" >> .hw_check.py
	@echo "try:" >> .hw_check.py
	@echo "    s = platform.system()" >> .hw_check.py
	@echo "    if s == 'Windows':" >> .hw_check.py
	@echo "        import ctypes" >> .hw_check.py
	@echo "        class M(ctypes.Structure): _fields_=[('l',ctypes.c_ulong),('m',ctypes.c_ulong),('t',ctypes.c_ulonglong)]" >> .hw_check.py
	@echo "        mem = M(); mem.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)); gb = mem.t / (1024**3)" >> .hw_check.py
	@echo "    elif s == 'Darwin':" >> .hw_check.py
	@echo "        gb = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])) / (1024**3)" >> .hw_check.py
	@echo "    else:" >> .hw_check.py
	@echo "        gb = int([l.split()[1] for l in open('/proc/meminfo') if 'MemTotal' in l][0]) / (1024**2)" >> .hw_check.py
	@echo "except Exception as e: print(f'Помилка детектора: {e}'); sys.exit(0)" >> .hw_check.py
	@echo "req = int('$(MIN_RAM_GB)')" >> .hw_check.py
	@echo "if gb < req: print(f'FAIL:{gb:.1f}')" >> .hw_check.py
	@echo "else: print(f'OK:{gb:.1f}')" >> .hw_check.py
	@HW_RESULT=$$(python3 .hw_check.py 2>/dev/null || python .hw_check.py 2>/dev/null); \
	rm -f .hw_check.py; \
	if echo "$$HW_RESULT" | grep -q "FAIL"; then \
		ACTUAL_RAM=$$(echo "$$HW_RESULT" | cut -d':' -f2); \
		echo "$(RED)❌ Небезпечно мало ОЗП ($$ACTUAL_RAM ГБ)! Мінімум: $(MIN_RAM_GB) ГБ.$(RESET)"; \
	elif echo "$$HW_RESULT" | grep -q "OK"; then \
		echo "$(GREEN)✅ Залізо схвалено (>= $(MIN_RAM_GB) ГБ ОЗП).$(RESET)"; \
	fi

setup: check-resources
	@echo "$(CYAN)🚀 Ініціалізація інфраструктури...$(RESET)"
ifeq ($(UNAME_S),Darwin)
	@echo "$(YELLOW)🍎 Перевірка macOS залежностей (OpenMP для XGBoost/LightGBM)...$(RESET)"
	@if command -v brew >/dev/null 2>&1; then \
		brew list libomp >/dev/null 2>&1 || { echo "$(YELLOW)📦 Встановлення libomp через Homebrew...$(RESET)"; brew install libomp; }; \
	else \
		echo "$(RED)⚠️ Увага: Homebrew не знайдено! Встановіть libomp вручну: brew install libomp$(RESET)"; \
	fi
endif
ifeq ($(UNAME_S),Linux)
	@echo "$(YELLOW)🐧 Перевірка Linux залежностей (libgomp1)...$(RESET)"
	@if command -v dpkg >/dev/null 2>&1; then \
		dpkg -s libgomp1 >/dev/null 2>&1 || echo "$(RED)⚠️ Увага: Можливо, бракує libgomp1. В разі помилок виконайте: sudo apt-get install libgomp1$(RESET)"; \
	fi
endif
	@echo "$(YELLOW)📁 Створення системних директорій...$(RESET)"
	@mkdir -p data models
	@echo "$(YELLOW)📦 Встановлення пакетного менеджера uv...$(RESET)"
	@$(PYTHON_CMD) -m pip install --upgrade pip --break-system-packages > /dev/null 2>&1 || true
	@$(PYTHON_CMD) -m pip install uv --break-system-packages > /dev/null 2>&1 || $(PYTHON_CMD) -m pip install --user uv > /dev/null 2>&1 || true
	@echo "$(YELLOW)🐍 Створення ізольованого середовища (Python $(PY_VER))...$(RESET)"
	@$(UV) venv --python $(PY_VER) $(VENV)
	@echo "$(YELLOW)⚡ Встановлення залежностей з requirements.txt...$(RESET)"
	@$(UV) pip install --python $(VENV) -r requirements.txt
	@if [ ! -f .env ]; then \
		echo "$(RED)🔐 Файл .env не знайдено. Створюю безпечний конфіг з .env.example...$(RESET)"; \
		cp .env.example .env; \
	fi
	@echo "$(GREEN)✅ Setup завершено! Введіть 'make help' для перегляду списку команд.$(RESET)"

# ------------------------------------------------------------------------------
# 2. ЗАПУСК ПРОЕКТІВ (Marimo Notebooks)
# ------------------------------------------------------------------------------
run-hw1:
	@echo "$(CYAN)🏡 Запуск Marimo для California Housing...$(RESET)"
	PYTHONPATH=. $(VENV)/bin/python -m marimo edit --watch hw_01_california/hw_01_linreg.py

run-hw2:
	@echo "$(CYAN)🌦️  Запуск Marimo для Rain in Australia...$(RESET)"
	PYTHONPATH=. $(VENV)/bin/python -m marimo edit --watch hw_02_australia/hw_02_logreg.py

run-hw3:
	@echo "$(CYAN)💼 Запуск Marimo для Salaries Estimation...$(RESET)"
	PYTHONPATH=. $(VENV)/bin/python -m marimo edit --watch hw_03_salaries/hw_03_knn.py

run-hw4:
	@echo "$(CYAN)🚗 Запуск Marimo для Autos & CarDekho Pricing...$(RESET)"
	PYTHONPATH=. $(VENV)/bin/python -m marimo edit --watch hw_04_cars/hw_04_trees_ensemble.py

# ------------------------------------------------------------------------------
# 3. ЗАПУСК РОЗГОРТАННЯ MLOps (FastAPI + Scalar)
# ------------------------------------------------------------------------------
api-hw1:
	@echo "$(CYAN)🚀 Запуск мікросервісу FastAPI (California Housing)...$(RESET)"
	@if [ ! -f models/california_housing/api.py ]; then \
		echo "$(RED)❌ Помилка: api.py не знайдено! Спочатку згенеруйте артефакти моделі в Marimo.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)📡 Сервер доступний за адресою: http://127.0.0.1:8000$(RESET)"
	@echo "$(YELLOW)📚 Scalar UI (Сучасна Документація API): http://127.0.0.1:8000/docs$(RESET)"
	@cd models/california_housing && ../../$(VENV)/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

api-hw2:
	@echo "$(CYAN)🚀 Запуск мікросервісу FastAPI (Rain in Australia)...$(RESET)"
	@if [ ! -f models/rain_in_australia/api.py ]; then \
		echo "$(RED)❌ Помилка: api.py не знайдено! Спочатку згенеруйте артефакти моделі в Marimo.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)📡 Сервер доступний за адресою: http://127.0.0.1:8000$(RESET)"
	@echo "$(YELLOW)📚 Scalar UI (Сучасна Документація API): http://127.0.0.1:8000/docs$(RESET)"
	@cd models/rain_in_australia && ../../$(VENV)/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

api-hw3:
	@echo "$(CYAN)🚀 Запуск мікросервісу FastAPI (Salary Prediction)...$(RESET)"
	@if [ ! -f models/salary_prediction/api.py ]; then \
		echo "$(RED)❌ Помилка: api.py не знайдено! Спочатку згенеруйте артефакти моделі в Marimo.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)📡 Сервер доступний за адресою: http://127.0.0.1:8000$(RESET)"
	@echo "$(YELLOW)📚 Scalar UI (Сучасна Документація API): http://127.0.0.1:8000/docs$(RESET)"
	@cd models/salary_prediction && ../../$(VENV)/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

api-hw4:
	@echo "$(CYAN)🚀 Запуск мікросервісу FastAPI (CarDekho Pricing)...$(RESET)"
	@if [ ! -f models/cardekho_pricing/api.py ]; then \
		echo "$(RED)❌ Помилка: api.py не знайдено! Спочатку згенеруйте артефакти моделі в Marimo.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)📡 Сервер доступний за адресою: http://127.0.0.1:8000$(RESET)"
	@echo "$(YELLOW)📚 Scalar UI (Сучасна Документація API): http://127.0.0.1:8000/docs$(RESET)"
	@cd models/cardekho_pricing && ../../$(VENV)/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# ------------------------------------------------------------------------------
# 4. ОЧИЩЕННЯ СМІТТЯ ТА КЕШІВ
# ------------------------------------------------------------------------------
clean:
	@echo "$(YELLOW)🧹 Видалення віртуального середовища та системного кешу...$(RESET)"
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "__marimo__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✨ Системний кеш очищено.$(RESET)"

deep-clean: clean
	@echo "$(RED)🧨 ПОВНЕ ОЧИЩЕННЯ: Видалення даних та ML-артефактів...$(RESET)"
	rm -f data/*.csv data/*.parquet data/*.zip data/.tmp* data/*.json
	rm -f models/*.joblib models/*.pkl models/*.safetensors
	rm -rf mlruns/
	@echo "$(GREEN)✅ Репозиторій повернуто до первозданного вигляду (Zero-State)!$(RESET)"

# Хак для ігнорування невідомих аргументів термінала
%:
	@:
