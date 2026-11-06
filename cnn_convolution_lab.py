"""CNN Convolution Lab — учебный симулятор свёртки.

Двенадцатый проект серии образовательных симуляторов:
    1. One Artificial Neuron   2. Perceptron           3. Mini Neural Network
    4. Gradient Descent        5. Backpropagation      6. Neural Network Training Lab
    7. Activation Functions    8. Loss Functions Lab   9. Classification Lab
   10. Overfitting & Underfitting Lab                 11. Optimizers Lab

Приложение показывает одну цепочку:
    входная матрица -> ядро -> поэлементное умножение -> сумма -> смещение
    -> активация -> карта признаков -> пулинг (max или average)
    -> финальная карта.

Главный вопрос, на которое отвечает симулятор: что именно происходит в каждой
позиции свёртки и в каждом окне пулинга, и почему карта выглядит так, а не иначе.

Операция — та, что в библиотеках нейросетей называют свёрткой: ядро применяется
без переворота (математически это взаимная корреляция). Различие объяснено в
справке и нигде не скрыто.

Считает всё сам на списках Python. Ни NumPy, ни внешних пакетов, ни изображений,
ни обучения: ни градиентов, ни потерь, ни метрик качества.

Только стандартная библиотека: Python 3, tkinter, tkinter.ttk, math, dataclasses.

Автор: SteveWorks001

Куда смотреть
-------------
Раздел 1   Область значений, палитра, обозначения
Раздел 2   Пресеты ядер и входных матриц
Раздел 3   Математика свёртки и пулинга: без всякого GUI
Раздел 4   Форматирование и разбор чисел и матриц
Раздел 5   Локализация: весь текст живёт в TEXTS (русский по умолчанию)
Раздел 6   Учебные эксперименты
Раздел 7   Справка: заголовки и тексты разделов
Раздел 8   Холсты: вход, ядро, произведения, карта признаков
Раздел 9   Главное окно
Раздел 10  Точка входа
"""

import math
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk, font as tkfont

# ----------------------------------------------------------------------
# Раздел 1. Область значений, палитра, обозначения
# ----------------------------------------------------------------------

KERNEL_SIZE = 3                              # ядро всегда 3x3
MIN_DIM, MAX_DIM = 3, 9                      # допустимый размер входа
DEFAULT_DIM = 7                              # 7x7 — как в учебниках
STRIDES = (1, 2)
PADS = (0, 1)                                # 0 — без заполнения, 1 — «то же»
ACTIVATIONS = ("none", "relu")
VALUE_LIMIT = 99.0                           # предел модуля введённого числа

POOL_MODES = ("max", "avg")                  # пулинг: максимум или среднее
POOL_WINDOWS = (2, 3)                        # окна пулинга без наложения

DEFAULT_INPUT_KEY = "vedge"
DEFAULT_KERNEL_KEY = "vedge"
DEFAULT_STRIDE = 1
DEFAULT_PAD = 0
DEFAULT_BIAS = 0.0
DEFAULT_ACTIVATION = "none"
DEFAULT_POOL_MODE = "max"
DEFAULT_POOL_WINDOW = 2
POOL_DIGITS = 4                              # знаков у среднего: деление не всегда точное
# Клетка финальной карты шире обычной: среднее показывает до четырёх знаков
# после точки, и самое длинное число ("-99.9999") в моноширинном 10 занимает
# 64 пиксела. 72 — чтобы подпись не залезала на соседа и на метку позиции.
POOL_CELL = 72

MIN_WIDTH = 1400                             # минимальная ширина окна
MIN_HEIGHT = 720                             # минимальная высота окна
TEXT_WIDTH = 40                              # ширина текстовых панелей, символы
LEFT_COL = 352                               # ширина колонки параметров

CELL = 40                                    # сторона ячейки матрицы, пикселы

# Ядро применяется без переворота — так работает свёртка в CNN. Ниже это
# названо прямым текстом, чтобы обозначение не расходилось с формулой.
COL_TEXT = "#242424"
COL_BG = "#fbfbf7"
COL_PAPER = "#fdfdf9"
COL_FRAME = "#c8c8c0"
COL_DIM = "#8a8a82"
COL_HEAD = "#31566f"
COL_KEY = "#b07d10"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"
COL_WIN = "#b07d10"                          # рамка текущего окна свёртки
COL_PAD = "#dfe3e8"                          # клетки нулевого заполнения
COL_NEG = "#c8651b"                          # отрицательные значения карты
COL_POS = "#2f6f9f"                          # положительные значения карты
COL_INTENSITY_LO = "#f6f4ec"                 # нижний конец шкалы яркости
COL_INTENSITY_HI = "#2f6f9f"                 # верхний конец шкалы яркости
COL_MARK = "#5b3a86"                         # отмеченные позиции ядра

def lerp_color(first, second, t):
    """Смешать два цвета: t = 0 даёт first, t = 1 даёт second."""
    t = max(0.0, min(1.0, t))
    channel_first = [int(first[i:i + 2], 16) for i in (1, 3, 5)]
    channel_second = [int(second[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [int(round(a + (b - a) * t)) for a, b in
             zip(channel_first, channel_second)]
    return "#%02x%02x%02x" % tuple(mixed)


def text_on(background):
    """Тёмный или светлый текст поверх цвета ячейки."""
    channels = [int(background[i:i + 2], 16) for i in (1, 3, 5)]
    brightness = (0.299 * channels[0] + 0.587 * channels[1] +
                  0.114 * channels[2]) / 255.0
    return "#1a1a16" if brightness > 0.55 else "#fdfdf9"


# ----------------------------------------------------------------------
# Раздел 2. Пресеты ядер и входных матриц
# ----------------------------------------------------------------------

def matrix(rows, columns, value):
    """Прямоугольная матрица из одного числа."""
    return [[float(value)] * columns for _ in range(rows)]


def preset_identity():
    return [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def preset_vertical_edge():
    return [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]


def preset_horizontal_edge():
    return [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]


def preset_sharpen():
    return [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]


def preset_blur():
    """Взвешенное усреднение соседей. Сумма весов равна 16: на неё делят,
    когда хотят среднее, а не удвоенную сумму. Здесь ядро остаётся целым,
    чтобы каждое произведение было точным числом."""
    return [[1, 2, 1], [2, 4, 2], [1, 2, 1]]


KERNEL_PRESETS = (
    {"key": "vedge", "values": preset_vertical_edge()},
    {"key": "hedge", "values": preset_horizontal_edge()},
    {"key": "sharpen", "values": preset_sharpen()},
    {"key": "blur", "values": preset_blur()},
    {"key": "identity", "values": preset_identity()},
    {"key": "custom", "values": preset_vertical_edge()},
)

KERNEL_BY_KEY = {entry["key"]: entry for entry in KERNEL_PRESETS}
KERNEL_COMPARISON_KEYS = ("identity", "vedge", "hedge", "sharpen", "blur")


def input_vertical_edge(size=DEFAULT_DIM):
    """Столбцы слева тёмные, справа светлые: скачок яркости по горизонтали."""
    return [[2.0 if j < size // 2 else 8.0 for j in range(size)]
            for i in range(size)]


def input_horizontal_edge(size=DEFAULT_DIM):
    """Строки сверху тёмные, снизу светлые: скачок яркости по вертикали."""
    return [[2.0 if i < size // 2 else 8.0 for j in range(size)]
            for i in range(size)]


def input_bright_center(size=DEFAULT_DIM):
    """Светлое пятно в центре на тёмном фоне."""
    middle = size // 2
    result = matrix(size, size, 1.0)
    for i in range(middle - 1, middle + 2):
        for j in range(middle - 1, middle + 2):
            if 0 <= i < size and 0 <= j < size:
                result[i][j] = 9.0
    return result


def input_single_dot(size=DEFAULT_DIM):
    """Одна светлая точка в центре: про образ ядра по одному отклику."""
    middle = size // 2
    result = matrix(size, size, 0.0)
    result[middle][middle] = 9.0
    return result


def input_cross(size=DEFAULT_DIM):
    """Плюс из светлых клеток на тёмном фоне."""
    middle = size // 2
    return [[8.0 if i == middle or j == middle else 1.0
             for j in range(size)] for i in range(size)]


def input_simple_shape(size=DEFAULT_DIM):
    """Светлая рамка вокруг тёмной середины: два края сразу."""
    result = matrix(size, size, 2.0)
    for i in range(size):
        for j in range(size):
            if i in (1, size - 2) or j in (1, size - 2):
                result[i][j] = 8.0
    return result


def input_diagonal(size=DEFAULT_DIM):
    """Нарисованная вручную матрица по умолчанию: косая светлая полоса."""
    return [[7.0 if i == j else 1.0 for j in range(size)]
            for i in range(size)]


INPUT_PRESETS = (
    {"key": "vedge", "rows": DEFAULT_DIM, "build": input_vertical_edge},
    {"key": "hedge", "rows": DEFAULT_DIM, "build": input_horizontal_edge},
    {"key": "center", "rows": DEFAULT_DIM, "build": input_bright_center},
    {"key": "dot", "rows": DEFAULT_DIM, "build": input_single_dot},
    {"key": "cross", "rows": DEFAULT_DIM, "build": input_cross},
    {"key": "shape", "rows": DEFAULT_DIM, "build": input_simple_shape},
    {"key": "custom", "rows": DEFAULT_DIM, "build": input_diagonal},
)

INPUT_BY_KEY = {entry["key"]: entry for entry in INPUT_PRESETS}


def build_input(key, size=DEFAULT_DIM):
    """Матрица пресета; у «своей матрицы» размер задаёт пользователь."""
    return [list(row) for row in INPUT_BY_KEY[key]["build"](size)]


# ----------------------------------------------------------------------
# Раздел 3. Математика свёртки и пулинга
# ----------------------------------------------------------------------
# Всё, что считает, живёт здесь и не знает про tkinter. GUI только вызывает
# эти функции и рисует то, что они вернули.

def padded_matrix(values, pad):
    """Нулевая рамка вокруг матрицы. pad = 0 возвращает копию."""
    rows = len(values)
    columns = len(values[0]) if rows else 0
    total_rows = rows + 2 * pad
    total_columns = columns + 2 * pad
    result = matrix(total_rows, total_columns, 0.0)
    for i in range(rows):
        for j in range(columns):
            result[i + pad][j + pad] = float(values[i][j])
    return result


def out_size(length, pad, kernel_size, stride):
    """output = floor((input + 2P - K) / S) + 1. 0 и меньше — конфигурация
    невозможна: окно не помещается."""
    return math.floor((length + 2 * pad - kernel_size) / stride) + 1


@dataclass
class Cell:
    """Один шаг свёртки: все промежуточные числа сразу, до подписи в GUI."""

    out_i: int
    out_j: int
    index: int
    win_top: int                             # строка верхнего края в padded
    win_left: int                            # столбец левого края в padded
    src_top: int                             # то же в исходных координатах
    src_left: int
    region: list = field(default_factory=list)
    products: list = field(default_factory=list)
    terms: tuple = ()                        # 9 троек (вход, ядро, произведение)
    sum_products: float = 0.0
    bias: float = 0.0
    pre_activation: float = 0.0
    activated: float = 0.0
    output: float = 0.0


def relu(value):
    """ReLU(z) = max(0, z)."""
    return value if value > 0.0 else 0.0


def activate(value, activation):
    if activation == "relu":
        return relu(value)
    return value


def cell_at(values, kernel, stride, pad, bias, activation, out_i, out_j):
    """Разложить одну позицию свёртки на все промежуточные числа.

    Ядро не переворачивается: X(i·S + m, j·S + n) · K(m, n).
    """
    padded = padded_matrix(values, pad)
    k = len(kernel)
    top = out_i * stride
    left = out_j * stride
    region = []
    products = []
    terms = []
    total = 0.0
    for m in range(k):
        region_row = []
        product_row = []
        for n in range(k):
            input_value = padded[top + m][left + n]
            weight = float(kernel[m][n])
            product = input_value * weight
            region_row.append(input_value)
            product_row.append(product)
            terms.append((input_value, weight, product))
            total += product
        region.append(region_row)
        products.append(product_row)
    pre = total + float(bias)
    activated = activate(pre, activation)
    rows = len(values)
    columns = len(values[0])
    out_columns = out_size(columns, pad, k, stride)
    return Cell(
        out_i=out_i,
        out_j=out_j,
        index=out_i * out_columns + out_j,
        win_top=top,
        win_left=left,
        src_top=top - pad,
        src_left=left - pad,
        region=region,
        products=products,
        terms=tuple(terms),
        sum_products=total,
        bias=float(bias),
        pre_activation=pre,
        activated=activated,
        output=activated,
    )


def convolve(values, kernel, stride, pad, bias, activation):
    """Полная карта признаков: {values, pre, out_rows, out_cols, cells}.

    values — итог (после активации), pre — сумма с смещением до активации.
    """
    rows = len(values)
    columns = len(values[0])
    out_rows = out_size(rows, pad, len(kernel), stride)
    out_columns = out_size(columns, pad, len(kernel), stride)
    pre_map = matrix(out_rows, out_columns, 0.0)
    value_map = matrix(out_rows, out_columns, 0.0)
    cells = []
    for i in range(out_rows):
        for j in range(out_columns):
            step = cell_at(values, kernel, stride, pad, bias, activation, i, j)
            pre_map[i][j] = step.pre_activation
            value_map[i][j] = step.output
            cells.append(step)
    return {
        "values": value_map,
        "pre": pre_map,
        "out_rows": out_rows,
        "out_columns": out_columns,
        "cells": cells,
        "total": out_rows * out_columns,
    }


def position_count(values, kernel, stride, pad):
    return (out_size(len(values), pad, len(kernel), stride) *
            out_size(len(values[0]), pad, len(kernel), stride))


def origins_of_stride(values, kernel_size, pad, new_stride):
    """Множество (i, j) — начал окна при данном шаге, в координатах padded."""
    rows = len(values)
    columns = len(values[0])
    out_rows = out_size(rows, pad, kernel_size, new_stride)
    out_columns = out_size(columns, pad, kernel_size, new_stride)
    return {(i * new_stride, j * new_stride)
            for i in range(max(0, out_rows))
            for j in range(max(0, out_columns))
            if out_rows > 0 and out_columns > 0}


def skipped_origins(values, kernel_size, pad, stride):
    """Начала окон, которые шаг пропускает: при шаге 1 их больше.

    Возвращает пару множеств: посещённые начала и пропущенные.
    """
    visited = origins_of_stride(values, kernel_size, pad, stride)
    single = origins_of_stride(values, kernel_size, pad, 1)
    return visited, single - visited


# --- пулинг: max и average по карте признаков -------------------------

@dataclass
class PoolCell:
    """Один шаг пулинга: окно, его клетки по порядку и результат."""

    out_i: int
    out_j: int
    index: int
    top: int                                 # строка верхнего края окна в карте
    left: int                                # столбец левого края окна в карте
    window: int
    mode: str
    region: list = field(default_factory=list)   # значения окна по строкам
    coords: list = field(default_factory=list)   # (i, j) в порядке «строка, столбец»
    flat: tuple = ()                              # те же значения плоским списком
    sum_values: float = 0.0
    count: int = 0
    output: float = 0.0
    best: float = 0.0                             # максимум окна (для max)
    winner: tuple = ()                            # (i, j) первой максимальной клетки
    ties: list = field(default_factory=list)      # все клетки с этим максимумом


def pool_cell(values, window, mode, out_i, out_j):
    """Разложить одно окно пулинга на клетки и результат. None — если окна нет.

    Окна не перекрываются: начало стоит в (out_i·W, out_j·W). Max сравнивает
    сырые числа, а не подписи на экране, и помнит все клетки с максимумом:
    победителем считается первая из них по порядку «строка, затем столбец».
    """
    rows = len(values)
    columns = len(values[0]) if rows else 0
    if window < 1 or out_i < 0 or out_j < 0:
        return None
    top, left = out_i * window, out_j * window
    if top + window > rows or left + window > columns:
        return None
    region = []
    coords = []
    flat = []
    for m in range(window):
        line = []
        for n in range(window):
            i, j = top + m, left + n
            value = float(values[i][j])
            line.append(value)
            coords.append((i, j))
            flat.append(value)
        region.append(line)
    total = 0.0
    for value in flat:
        total += value
    count = len(flat)
    best = max(flat)
    ties = []
    winner = ()
    if mode == "avg":
        output = total / count
    else:
        output = best
        ties = [coord for coord, value in zip(coords, flat) if value == best]
        winner = ties[0]
    # Номер клетки «сначала строки, потом столбцы»: та же нумерация, что и
    # метка на карте Z, поэтому подпись и клетка всегда согласованы.
    out_columns = max(0, out_size(columns, 0, window, window))
    return PoolCell(
        out_i=out_i,
        out_j=out_j,
        index=out_i * out_columns + out_j,
        top=top,
        left=left,
        window=window,
        mode=mode,
        region=region,
        coords=coords,
        flat=tuple(flat),
        sum_values=total,
        count=count,
        output=output,
        best=best,
        winner=winner,
        ties=ties,
    )


def pool_map(values, window, mode):
    """Пулинг всей карты: {values, out_rows, out_columns, cells, total, unused_*}.

    Размеры считает та же out_size с шагом, равным окну, поэтому неполные окна
    справа и снизу отсекаются целиком, а не делятся на меньшее число клеток.
    """
    rows = len(values)
    columns = len(values[0]) if rows else 0
    if window < 1:
        return {"values": [], "out_rows": 0, "out_columns": 0, "cells": [],
                "total": 0, "unused_rows": rows, "unused_columns": columns,
                "unused_cells": rows * columns}
    out_rows = max(0, out_size(rows, 0, window, window))
    out_columns = max(0, out_size(columns, 0, window, window))
    value_map = matrix(out_rows, out_columns, 0.0)
    cells = []
    for i in range(out_rows):
        for j in range(out_columns):
            step = pool_cell(values, window, mode, i, j)
            if step is None:
                continue
            value_map[i][j] = step.output
            cells.append(step)
    used_rows = out_rows * window if cells else 0
    used_columns = out_columns * window if cells else 0
    return {
        "values": value_map,
        "out_rows": out_rows,
        "out_columns": out_columns,
        "cells": cells,
        "total": len(cells),
        "unused_rows": max(0, rows - used_rows),
        "unused_columns": max(0, columns - used_columns),
        "unused_cells": max(0, rows * columns - used_rows * used_columns),
    }


def pool_window_for(values, window, row, column):
    """Окно пулинга, накрывающее клетку карты (row, column). None — если не
    накрыто: клетка осталась за неполным окном справа или снизу."""
    rows = len(values)
    columns = len(values[0]) if rows else 0
    if window < 1 or not (0 <= row < rows and 0 <= column < columns):
        return None
    out_i, out_j = row // window, column // window
    if out_i * window + window > rows or out_j * window + window > columns:
        return None
    return (out_i, out_j)


def min_max(values):
    flat = [float(cell) for row in values for cell in row]
    if not flat:
        return 0.0, 0.0
    return min(flat), max(flat)


def intensity_color(value, low, high):
    """Яркость клетки входа: от COL_INTENSITY_LO к COL_INTENSITY_HI."""
    if high - low <= 1e-12:
        t = 0.0
    else:
        t = (value - low) / (high - low)
    return lerp_color(COL_INTENSITY_LO, COL_INTENSITY_HI, t)


def diverging_color(value, scale):
    """Карта признаков: минус — тёплый, плюс — холодный, ноль — бумага."""
    if scale <= 1e-12:
        return COL_PAPER
    t = max(-1.0, min(1.0, value / scale))
    if t >= 0.0:
        return lerp_color(COL_PAPER, COL_POS, t)
    return lerp_color(COL_PAPER, COL_NEG, -t)


def abs_scale(values):
    flat = [abs(float(cell)) for row in values for cell in row]
    return max(flat) if flat else 0.0


# ----------------------------------------------------------------------
# Раздел 4. Форматирование и разбор чисел и матриц
# ----------------------------------------------------------------------

def fmt(value, digits=4):
    """Число для показа: целые без точки, дробные без лишних нулей."""
    number = float(value)
    if abs(number) < 1e-12:
        number = 0.0
    rounded = round(number, digits)
    if abs(rounded - round(rounded)) < 1e-12:
        return "%d" % int(round(rounded))
    text = ("%%.%df" % digits) % rounded
    text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def parse_number(text):
    """Принять точку и запятую как десятичный разделитель. None — если нет."""
    raw = str(text).strip().replace(",", ".")
    if raw in ("", "+", "-", "."):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def term_text(input_value, weight):
    """Одно слагаемое вида a·k с аккуратной записью отрицательного ядра."""
    left = fmt(input_value)
    right = fmt(weight)
    if weight < 0:
        right = "(%s)" % right
    return "%s·%s" % (left, right)


def sum_text(terms):
    """Строка сложения девяти произведений: 2·(-1) + 8·0 + 8·1 и так далее.

    Знак берут по самому произведению, поэтому запись читается как строка на
    бумаге: отрицательное слагаемое стоит после минуса.
    """
    parts = []
    for index, (input_value, weight, product) in enumerate(terms):
        if index == 0:
            parts.append(term_text(input_value, weight))
        elif product < 0:
            parts.append(" − " + term_text(abs(input_value), abs(weight)))
        else:
            parts.append(" + " + term_text(input_value, weight))
    return "".join(parts)


def relu_text(value):
    """max(0, z) с подставленным числом."""
    return "max(0, %s)" % fmt(value)


def pool_sum_text(values, digits=POOL_DIGITS):
    """Слагаемые среднего в порядке «строка, затем столбец»: 8 + 2 − 5 ….

    Знак берут по самому числу, поэтому запись читается как строка на бумаге.
    """
    parts = []
    for index, value in enumerate(values):
        if index == 0:
            parts.append(fmt(value, digits))
        elif value < 0:
            parts.append(" − " + fmt(abs(value), digits))
        else:
            parts.append(" + " + fmt(value, digits))
    return "".join(parts)


def pool_values_text(values, digits=POOL_DIGITS):
    """Числа окна через запятую — ровно в том порядке, в каком их видит max."""
    return ", ".join(fmt(value, digits) for value in values)


def matrix_to_text(values):
    """Матрица в редактируемый вид: строки — с новой строки, числа — пробел."""
    return "\n".join(" ".join(fmt(cell) for cell in row) for row in values)


def parse_matrix(text):
    """Разобрать текст в матрицу. Возвращает (matrix, None) или (None, код).

    Строки разделяют перевод строки или точка с запятой, числа внутри строки —
    запятая, пробел или табуляция. Коды отказа: empty — пусто, number — не
    число, limit — число вне дозволенного, ragged — строки разной длины,
    size — размер вне диапазона (матрица при этом возвращается, чтобы её
    размеры попали в сообщение об отказе).
    """
    rows = []
    for raw_line in str(text).replace("\r", "\n").replace(";", "\n").split("\n"):
        line = raw_line.replace(",", " ").split()
        if not line:
            continue
        row = []
        for chunk in line:
            number = parse_number(chunk)
            if number is None:
                return None, "number"
            if abs(number) > VALUE_LIMIT:
                return None, "limit"
            row.append(number)
        rows.append(row)
    if not rows:
        return None, "empty"
    if any(len(row) != len(rows[0]) for row in rows):
        return None, "ragged"
    if not (MIN_DIM <= len(rows) <= MAX_DIM and
            MIN_DIM <= len(rows[0]) <= MAX_DIM):
        return rows, "size"
    return rows, None


def parse_scalar(text):
    """Одно число из поля ввода: (значение, None) или (None, код отказа)."""
    number = parse_number(text)
    if number is None:
        return None, "number"
    if abs(number) > VALUE_LIMIT:
        return None, "limit"
    return number, None


# ----------------------------------------------------------------------
# Раздел 5. Локализация
# ----------------------------------------------------------------------

LANG = "ru"


def set_language(code):
    global LANG
    LANG = code


def get_language():
    return LANG


def T(key, **kwargs):
    """Текст по ключу для текущего языка.

    Ключей без перевода быть не должно; если всё же нашёл — берём английский,
    а при отсутствии и его показываем сам ключ, чтобы интерфейс не падал.
    """
    table = TEXTS.get(LANG) or TEXTS["en"]
    text = table.get(key)
    if text is None:
        text = TEXTS["en"].get(key)
    if text is None:
        return key
    if kwargs:
        return text.format(**kwargs)
    return text


# Весь переводимый текст живёт здесь. Формулы и числа язык не меняют: они
# подписаны ключами ниже, а сами выражения нейтральны.

TEXTS = {
    "ru": {
        "app_title": "CNN Convolution Lab — как ядро проходит по входной матрице",
        "subtitle": ("Входная матрица → ядро → поэлементное умножение → сумма → "
                     "смещение → активация → карта признаков → пулинг (max или "
                     "average) → финальная карта. Каждое число видно в моменте: "
                     "ядро здесь применяется без переворота, как это делают "
                     "библиотеки свёрточных сетей."),
        "box_language": "Язык интерфейса",
        "lang_ru": "Русский",
        "lang_en": "English",
        # --- вкладки
        "tab_conv": "Свёртка",
        "tab_pool": "Пулинг",
        "tab_compare": "Сравнение ядер",
        "tab_experiments": "Эксперименты",
        "tab_formula": "Формулы",
        "tab_help": "Справка",
        # --- левая колонка: вход
        "box_input": "Входная матрица X",
        "lbl_input_preset": "Учебный рисунок",
        "input_vedge_name": "Вертикальный край",
        "input_vedge_desc": "Столбцы слева тёмные (2), справа светлые (8). Скачок "
                            "яркости идёт по горизонтали — ровно там, где его "
                            "ищет вертикальный край.",
        "input_hedge_name": "Горизонтальный край",
        "input_hedge_desc": "Строки сверху тёмные (2), снизу светлые (8). Скачок "
                            "яркости идёт по вертикали.",
        "input_center_name": "Светлое пятно в центре",
        "input_center_desc": "Квадрат 3×3 из девяток на фоне единиц: у пятна четыре "
                             "края, и отклик появляется там, где окно накрывает "
                             "границу.",
        "input_dot_name": "Одна светлая точка",
        "input_dot_desc": "Девятка в центре, вокруг нули. Карта признаков "
                          "повторяет само ядро с шагом свёртки: удобно смотреть, "
                          "что вес значит.",
        "input_cross_name": "Крест",
        "input_cross_desc": "Средняя строка и средний столбец светлые: вертикальный "
                            "и горизонтальный края одновременно.",
        "input_shape_name": "Простая фигура",
        "input_shape_desc": "Светлая рамка вокруг тёмной середины: два края сверху и "
                            "снизу и два слева и справа.",
        "input_custom_name": "Своя матрица",
        "input_custom_desc": "Матрицу набирают вручную в поле ниже. Размер — от "
                             "{lo}×{lo} до {hi}×{hi}, числа — от -{lim} до {lim}.",
        "box_matrix": "Своя матрица",
        "hint_matrix": "По одному ряду на строку (или через точку с запятой), числа "
                       "через пробел или запятую. Дробные пишутся с точкой: 2.5 — это "
                       "одно число, а 2,5 — два. Пустой ввод не принимается.",
        "btn_apply_matrix": "Применить матрицу",
        "matrix_size": "Размер входа: {rows}×{cols}",
        # --- левая колонка: ядро
        "box_kernel": "Ядро свёртки K",
        "lbl_kernel_preset": "Готовое ядро",
        "kernel_vedge_name": "Вертикальный край",
        "kernel_vedge_desc": "Ядро сравнивает соседние столбцы: правее — плюс, "
                             "левее — минус. Оно отзывается на смену яркости по "
                             "горизонтали и молчит на ровном месте.",
        "kernel_hedge_name": "Горизонтальный край",
        "kernel_hedge_desc": "Зеркальный набор весов: нижние строки с плюсом, "
                             "верхние с минусом. Отклик растёт там, где яркость "
                             "меняется от строки к строке.",
        "kernel_sharpen_name": "Резкость",
        "kernel_sharpen_desc": "Центр весит сильнее соседей, которые берутся со "
                              "знаком минус. Локальный контраст усиливается: край "
                              "становится резче, ровное остаётся ровным.",
        "kernel_blur_name": "Размытие",
        "kernel_blur_desc": "Соседи складываются с весами 1, 2 и 4 — усреднение по "
                            "соседству. Делить на сумму весов (16) здесь не нужно: "
                            "тогда карта просто становится глаже и светлее.",
        "kernel_identity_name": "Единичное",
        "kernel_identity_desc": "Одна единица в центре: карта повторяет сам вход, "
                                "сжатый шагом свёртки. Хорошо видно, что делает "
                                "окно без всякого фильтра.",
        "kernel_custom_name": "Своё ядро",
        "kernel_custom_desc": "Вес, набранный вручную. От него ждут ровно то, что "
                              "в нём записано: свёртка складывает произведения "
                              "окна на эти числа.",
        "hint_kernel": "Девять полей — это и есть ядро: влево-вправо и вниз. Правка "
                       "любого поля переводит пресет в «Своё ядро».",
        "lbl_bias": "Смещение b",
        "hint_bias": "К сумме произведений всегда прибавляется одно и то же число: "
                     "по умолчанию b = 0, поэтому оно ничего не меняет.",
        # --- левая колонка: шаг, заполнение, активация
        "box_geometry": "Шаг и заполнение",
        "lbl_stride": "Шаг S",
        "stride_1": "1 — окно сдвигается на клетку",
        "stride_2": "2 — окно перепрыгивает клетку",
        "lbl_padding": "Заполнение P",
        "pad_0": "0 — без заполнения (valid)",
        "pad_1": "1 — нулевая рамка (same)",
        "hint_stride": "При S = 2 начало окна каждый раз скачет на две клетки, "
                       "промежуточные положения не считаются: карта меньше.",
        "hint_padding": "P = 1 добавляет вокруг входа рамку из нулей. Окно начинает "
                        "вставать у самого края картинки, а карта растёт.",
        "box_activation": "Активация после свёртки",
        "act_none": "None — без активации",
        "act_relu": "ReLU — max(0, z)",
        "hint_activation": "Здесь активация только одна и только затем, чтобы "
                           "показать путь «сумма → ReLU → клетка карты». Функциями "
                           "активации занимается отдельный лабораторный проект.",
        # --- левая колонка: пулинг
        "box_pool": "Пулинг после активации",
        "lbl_pool_mode": "Тип пулинга",
        "pool_max": "Max — максимум в окне",
        "pool_avg": "Average — среднее в окне",
        "lbl_pool_window": "Окно пулинга",
        "pool_w2": "2×2, без наложения",
        "pool_w3": "3×3, без наложения",
        "hint_pool": "Пулинг — последнее звено цепочки: он берёт карту признаков "
                     "уже после активации и уменьшает её, ничего не пересчитывая "
                     "окном свёртки.",
        "hint_pool_window": "Окно сдвигается на свою ширину, поэтому клетка карты "
                            "не переходит из окна в окно. Неполное окно справа или "
                            "снизу отбрасывается целиком: такие клетки видны "
                            "серыми.",
        "btn_reset_all": "Сбросить всё",
        # --- центр: вход и окно
        "cap_input": "Вход X",
        "cap_input_padded": "Вход X с нулевой рамкой P = 1",
        "cap_kernel": "Ядро K",
        "cap_products": "Произведения X·K",
        "cap_map": "Карта признаков Y",
        "legend_window": "Золотая рамка — окно свёртки",
        "legend_visited": "Точка — начало окна при текущем шаге",
        "legend_skipped": "✕ — начало, которое шаг пропускает",
        "legend_negative": "Тёплые клетки — отрицательные значения карты",
        "legend_positive": "Холодные клетки — положительные значения",
        "window_coords": "Окно: строки {r0}…{r1}, столбцы {c0}…{c1}",
        "window_coords_pad": "Окно: padded-строки {r0}…{r1}, padded-столбцы "
                             "{c0}…{c1} (за краями входа стоят нули)",
        # --- шаг и сканирование
        "box_step": "Шаг за шагом",
        "btn_prev": "← Назад",
        "btn_next": "Вперёд →",
        "btn_step_reset": "Сброс шага",
        "position_label": "Позиция {index} из {total}",
        "box_scan": "Автоматический обход",
        "btn_scan_start": "Старт",
        "btn_scan_pause": "Пауза",
        "btn_scan_reset": "Сброс обхода",
        "lbl_speed": "Пауза между позициями, мс",
        # --- правая колонка: расчёт
        "calc_title": "Расчёт в этой позиции",
        "calc_region": "Окно входа X({i},{j}) … X({i2},{j2}):",
        "calc_kernel_row": "Строки ядра K — без переворота:",
        "calc_products": "Девять произведений, по строке на ряд окна:",
        "calc_sum_head": "Σ — сумма произведений:",
        "calc_sum_line": "{expr} = {total}",
        "calc_bias_head": "Сумма со смещением:",
        "calc_bias_line": "z = {total} + {bias} = {pre}",
        "calc_relu_head": "Активация ReLU:",
        "calc_relu_line": "Y({i},{j}) = ReLU(z) = {relu} = {out}",
        "calc_none_head": "Активация:",
        "calc_none_line": "Y({i},{j}) = z = {out} (None ничего не меняет)",
        "calc_cell_head": "Клетка карты:",
        "calc_cell_line": "Y({i},{j}) = {out}",
        "calc_map_missing": "Здесь окно не влезает: позиция не считается.",
        # --- размер выхода
        "box_outsize": "Размер карты признаков",
        "outsize_formula": "выход = floor((вход + 2P − K) / S) + 1",
        "outsize_rows": "Строки: floor(({n} + 2·{p} − {k}) / {s}) + 1 = {o}",
        "outsize_cols": "Столбцы: floor(({n} + 2·{p} − {k}) / {s}) + 1 = {o}",
        "outsize_square": "floor(({n} + 2·{p} − {k}) / {s}) + 1 = {o}",
        "outsize_map": "Карта получается {rows}×{cols}, всего {total} позиций.",
        "outsize_impossible": "При S = {s} и P = {p} окно {k}×{k} не влезает в "
                              "вход {rows}×{cols}: позиций нет вовсе.",
        # --- вкладка пулинга
        "pool_intro": ("Пулинг идёт после активации и до финальной карты: окно "
                       "накрывает карту признаков без наложения, и каждая клетка "
                       "Z — это одно число из своего окна: максимум или среднее."),
        "cap_pool_src": "Карта признаков Y — вход пулинга",
        "cap_pool_out": "Финальная карта Z после пулинга",
        "legend_pool_window": "Золотая рамка — выбранное окно пулинга",
        "legend_pool_star": "★ — клетка с максимумом; жирным — та, откуда "
                            "взято число",
        "legend_pool_unused": "Серые клетки — не попали ни в одно окно",
        "legend_pool_selected": "Золотая рамка — выбранная клетка Z",
        "pool_summary": "Y {rows}×{cols} → окно {w}×{w} → Z {prows}×{pcols}, "
                        "клеток: {ptotal}.",
        "pool_no_map": "Пулингу нечего считать: карта Y {rows}×{cols}, а целое "
                       "окно {w}×{w} не встаёт никуда.",
        "pool_outsize_rows": "Строки: floor(({n} − {w}) / {w}) + 1 = {o}",
        "pool_outsize_cols": "Столбцы: floor(({n} − {w}) / {w}) + 1 = {o}",
        "pool_outsize_square": "floor(({n} − {w}) / {w}) + 1 = {o}",
        "pool_calc_title": "Расчёт этой клетки пулинга",
        "pool_window_coords": "Окно {w}×{w}: строки {r0}…{r1}, столбцы {c0}…{c1}",
        "pool_values_head": "Числа окна по порядку (сначала строки, потом столбцы):",
        "pool_max_head": "Max — самое большое число окна:",
        "pool_max_line": "Z({i},{j}) = max({values}) = {out}",
        "pool_winner_line": "Максимум {best} стоит в клетке Y({r},{c}) — первой "
                            "по порядку среди равных.",
        "pool_tie_line": "Равных максимумов в окне: {n} — {cells}. Число берут из "
                         "первого по порядку, клетки Y({r},{c}).",
        "pool_avg_head": "Average — среднее по окну:",
        "pool_avg_line": "Z({i},{j}) = ({expr}) / {count} = {out}",
        "pool_avg_note": "Показано с округлением до {d} знаков: точное частное "
                         "бывает длиннее. Особенно часто так выходит в окне "
                         "{w}×{w}, где делитель не кратен десяти.",
        "pool_cell_head": "Клетка финальной карты:",
        "pool_cell_line": "Z({i},{j}) = {out}",
        "pool_map_missing": "Ни одного целого окна {w}×{w}: пулинг не считает ни "
                            "одной клетки.",
        "pool_unused": "Снизу и справа без окна — строки: {rows}, столбцы: "
                       "{cols}, всего клеток: {cells}.",
        "pool_all_used": "Все клетки карты попали в окна: {rows}×{cols} делится "
                         "на {w}×{w} без остатка.",
        # --- вкладка сравнения
        "cmp_intro": "Один и тот же вход, пять разных ядер: меняется только ядро, "
                     "поэтому разница на картах — это работа фильтра, а не данных.",
        "cmp_input_line": "Вход: {name} ({rows}×{cols}), шаг S = {s}, заполнение "
                          "P = {p}, смещение b = {b}, активация: {a}.",
        "cmp_map_title": "Ядро: {name}",
        "cmp_map_size": "карта {rows}×{cols}",
        "cmp_max": "максимум |значение| = {v}",
        "cmp_minmax": "значения от {lo} до {hi}",
        "cmp_note": "Кликните по карте, чтобы перенести эту позицию на вкладку "
                    "«Свёртка» и разобрать её по числам.",
        # --- вкладка экспериментов
        "exp_intro": "Каждый эксперимент собирает готовый учебный набор: вход, "
                     "ядро, шаг, заполнение, смещение и активация. Числа "
                     "детерминированы: одно и то же нажатие даёт то же самое.",
        "exp_apply": "Открыть в разделе «Свёртка»",
        "exp_result": "Карта {rows}×{cols}; выбранная позиция {index} из {total}; "
                      "значение Y = {value}.",
        "exp_edge_v_name": "Э1 — вертикальные края",
        "exp_edge_v_goal": "Вертикальный край во входа и ядро того же типа. Окно, "
                           "поймавшее границу яркости, даёт 18; окна, вставшие "
                           "целиком в тёмное или целиком в светлое, дают 0. Так "
                           "выглядит «фильтр отзывается на край».",
        "exp_edge_h_name": "Э2 — горизонтальные края",
        "exp_edge_h_goal": "Тот же приём поперёк: горизонтальный край и ядро, "
                           "сравнивающее соседние строки. Пустая средняя строка "
                           "ядра не участвует в сумме.",
        "exp_smooth_name": "Э3 — сглаживание",
        "exp_smooth_goal": "Ядро размытия на кресте: карта становится глаже, а "
                           "значения крупнее входа, потому что веса складывают "
                           "девятерых соседей и нигде не делят на 16.",
        "exp_sharpen_name": "Э4 — усиление контраста",
        "exp_sharpen_goal": "Ядро резкости на светлом пятне: внутри пятна карта "
                            "равна самому входу (9), на краю вырастает до 17, а "
                            "сразу за краем уходит в минус (-7). Выбрана клетка с "
                            "числом 17.",
        "exp_stride_name": "Э5 — шаг 2",
        "exp_stride_goal": "Тот же вход и то же ядро, но шаг 2. Окно прыгает через "
                           "клетку, половины положений нет, карта 3×3 вместо 5×5: "
                           "проверьте это по отметкам ✕ на входе.",
        "exp_padding_name": "Э6 — заполнение нулями",
        "exp_padding_goal": "Один и тот же набор с P = 1 и без него: карта растёт с 5×5 "
                            "до 7×7, потому что окно встаёт у самого края картинки. "
                            "Угловая позиция даёт 4: два веса отработали по "
                            "нулям рамки.",
        "exp_relu_name": "Э7 — ReLU после свёртки",
        "exp_relu_goal": "Вертикальный край на светлом пятне: слева от границы "
                         "сумма равна 24, справа — минус 24. ReLU обнуляет "
                         "отрицательную половину, и на карте остаётся только одна "
                         "сторона края. Выбрана клетка со значением z = -24 до "
                         "активации.",
        # --- вкладка формул
        "formula_core_head": "Что считает приложение",
        "formula_core": "z(i,j) = ΣΣ X(i+m, j+n) · K(m,n) + b",
        "formula_core_note": "Двойная сумма идёт по m = 0…K−1 и n = 0…K−1, то есть "
                             "по всем девяти клеткам окна. Ядро не перевёрнуто: "
                             "это взаимная корреляция, именно так свёртку делают "
                             "библиотеки свёрточных сетей.",
        "formula_symbols_head": "Обозначения",
        "formula_out_head": "Размер карты",
        "formula_out": "output = floor((input + 2P − K) / S) + 1",
        "formula_out_note": "Здесь substitution показана живыми числами текущей "
                            "конфигурации: смена шага или заполнения меняет строку "
                            "сразу.",
        "formula_current_head": "Текущая конфигурация в числах",
        "symbol_x": "X — входная матрица, яркость клетки (rows, columns)",
        "symbol_k": "K — ядро свёртки, {k}×{k} весов",
        "symbol_ij": "i, j — координаты клетки карты признаков Y",
        "symbol_mn": "m, n — координаты клетки внутри окна и ядра",
        "symbol_sum": "ΣΣ — сумма девяти произведений окна на ядро",
        "symbol_b": "b — смещение, одно число для всей карты",
        "symbol_z": "z — сумма с смещением, значение до активации",
        "symbol_y": "Y — карта признаков: Y = z или Y = ReLU(z)",
        "symbol_s": "S — шаг: на сколько клеток сдвигается окно",
        "symbol_p": "P — заполнение: ширина нулевой рамки вокруг входа",
        "formula_relu": "ReLU(z) = max(0, z)",
        "formula_relu_note": "Активация здесь одна и только как продолжение "
                             "цепочки: карта признаков после неё.",
        # --- справка
        "help_terms_title": "Свёртка или взаимная корреляция",
        "help_terms_text": ("В математике свёртка переворачивает ядро на 180°, а "
                            "взаимная корреляция — нет. Библиотеки свёрточных "
                            "сетей называют «свёрткой» операцию без переворота, и "
                            "это приложение делает так же: клетка карты — сумма "
                            "произведений окна X(i+m, j+n) на K(m,n) в тех же "
                            "координатах.\n"
                            "Разница для учебного вывода одна: если бы ядро "
                            "переворачивали, симметричные ядра дали бы ту же карту, "
                            "а несимметричное ядро показало бы край в другую "
                            "сторону. Приложение переворот не делает и нигде не "
                            "притворяется, что делает."),
        "help_chain_title": "Что видно на экране",
        "help_chain_text": ("Пять вещей стоят рядом именно в порядке цепочки: вход "
                            "X, ядро K, девять поэлементных произведений, карта "
                            "признаков Y и числа посередине.\n"
                            "Золотая рамка на входе — окно, которое сейчас в "
                            "работе. Та же клетка подсвечена на карте справа, и "
                            "панель расчёта показывает все девять произведений, "
                            "сумму, смещение и активацию для этой позиции.\n"
                            "Нажмите любую клетку карты — окно переставится туда. "
                            "Нажмите любую клетку входа — окно встанет так, чтобы "
                            "эту клетку накрыть.\n"
                            "Цепочка продолжается на вкладке «Пулинг»: там карта Y "
                            "уменьшается окном 2×2 или 3×3 до финальной карты Z, и "
                            "каждое число снова видно."),
        "help_step_title": "Режим «шаг за шагом»",
        "help_step_text": ("«Вперёд» и «Назад» двигают окно по строкам слева "
                           "направо, затем — следующая строка. Порядок позиций "
                           "совпадает с обходом карты признаков, поэтому номер "
                           "позиции и координаты клетки всегда согласованы.\n"
                           "Режим шага не зависит от анимации: кнопка пересчитывает "
                           "позицию сразу, можно идти назад и останавливаться где "
                           "угодно."),
        "help_stride_title": "Шаг и заполнение на картинке",
        "help_stride_text": ("Шаг: начало окна сдвигается на S клеток. При S = 2 на "
                             "входе остаются точки там, где окно вставало, и "
                             "отметки ✕ там, где оно могло бы встать, но не "
                             "встало. Половинa положений пропадает, карта "
                             "уменьшается.\n"
                             "Заполнение: вокруг входа появляется рамка из нулей. "
                             "Окно начинает вставать у самого края картинки, часть "
                             "его весов умножается на нули — и карта растёт. Клетки "
                             "рамки нарисованы отдельно, чтобы их нельзя было "
                             "спутать с яркостью входа."),
        "help_bias_title": "Смещение и ReLU",
        "help_bias_text": ("Свёртка — это не только сумма произведений: к ней "
                           "прибавляют одно и то же число b. Панель расчёта "
                           "показывает «Σ + b» отдельной строкой, чтобы видно было "
                           "два разных шага.\n"
                           "ReLU оставлена одна: она демонстрирует только путь "
                           "«z → max(0, z) → клетка карты». Katalog функций "
                           "активации — задача отдельного проекта серии."),
        "help_pool_title": "Пулинг: максимум и среднее",
        "help_pool_text": ("Пулинг уменьшает карту признаков. Окно шагает по ней на "
                           "свою ширину, поэтому клетки не перекрываются: Max берёт "
                           "самое большое число окна, Average складывает все числа "
                           "окна и делит на их количество.\n"
                           "Неполное окно справа или снизу отбрасывается целиком. "
                           "Размеры считает та же формула, что и у свёртки, только с "
                           "шагом, равным окну: floor((размер − окно) / окно) + 1. "
                           "Клетки, которым не хватило окна, нарисованы серыми, а их "
                           "число подписано под картой.\n"
                           "У max есть нюанс: если максимум стоит в нескольких "
                           "клетках, показаны все они, но число берут из первой по "
                           "порядку «строка, затем столбец».\n"
                           "У average показанное число округлено до четырёх знаков: "
                           "7/9 на экране выглядит как 0.7778, поэтому в окне 3×3 "
                           "(делитель 9) округление заметно сильнее всего."),
        "help_kernels_title": "Что ядро «замечает»",
        "help_kernels_text": ("Ни одно из этих ядер ничего не ищет само по себе: "
                              "это набор весов, и то, что получается на карте, "
                              "видно по числам.\n"
                              "• Вертикальный край — усиливает разницу соседних "
                              "столбцов; на ровном участке даёт около нуля.\n"
                              "• Горизонтальный край — то же по строкам.\n"
                              "• Размытие — складывает соседей, карта становится "
                              "глаже.\n"
                              "• Резкость — центральный вес больше остальных, локальный "
                              "контраст усиливается.\n"
                              "• Единичное — передаёт вход как есть."),
        "help_scope_title": "Чего здесь нет — и почему",
        "help_scope_text": ("Нет обучения, градиентов, функции потерь, свёрточных "
                            "слоёв друг на друге, метрик качества и загруженных "
                            "изображений. Пулинг здесь есть, но ровно один и как "
                            "последнее звено цепочки; отдельная система пулинга с "
                            "наложением окон и многими стратегиями — задача других "
                            "проектов серии.\n"
                            "Здесь ровно один свёрточный проход и одно окно пулинга "
                            "после него: один вход, одно ядро, одна карта признаков "
                            "и одна финальная карта."),
        "help_keys_title": "Клавиатура",
        "help_keys_text": ("← и → — назад и вперёд по позициям свёртки.\n"
                           "Home — вернуться к первой позиции.\n"
                           "Пробел — пуск и пауза автоматического обхода.\n"
                           "Enter в поле матрицы — применить набранное."),
        # --- строка состояния
        "status_start": "Готово: вход «{input}», ядро «{kernel}», карта {rows}×{cols}.",
        "status_lang": "Язык интерфейса: {name}. Состояние не изменилось.",
        "status_step": "Позиция {index} из {total}: Y({i},{j}) = {value}.",
        "status_edge_first": "Это первая позиция: назад идти некуда.",
        "status_edge_last": "Это последняя позиция: вперёд идти некуда.",
        "status_reset": "Сброс: выбрано начало карты, обход остановлен.",
        "status_scan": "Обход идёт: {speed} мс на позицию.",
        "status_scan_pause": "Обход на паузе: позиция {index} из {total}.",
        "status_scan_end": "Обход дошёл до последней позиции.",
        "status_input_preset": "Вход: {name}.",
        "status_kernel_preset": "Ядро: {name}.",
        "status_kernel_custom": "Ядро набрано вручную: пресет switched на «Своё».",
        "status_stride": "Шаг S = {s}: позиций теперь {total}.",
        "status_pad": "Заполнение P = {p}: вход с рамкой {rows}×{cols}, позиций "
                      "{total}.",
        "status_bias": "Смещение b = {b}.",
        "status_activation": "Активация: {name}.",
        "status_matrix_applied": "Принята матрица {rows}×{cols}.",
        "status_experiment": "Эксперимент «{name}»: {goal}",
        "status_impossible": "Конфигурация невозможна: {reason}",
        "status_cell_map": "Клетка карты Y({i},{j}) выбрана: окно — строки {r0}…{r1}, "
                           "столбцы {c0}…{c1}.",
        "status_cell_input": "Окно поставлено на клетку входа ({r},{c}).",
        "status_pool_mode": "Пулинг: {name}. Финальная карта Z — {rows}×{cols}.",
        "status_pool_window": "Окно пулинга {w}×{w}: финальная карта Z "
                              "{rows}×{cols}, клеток: {total}.",
        "status_pool_cell": "Клетка Z({i},{j}) выбрана: окно — строки {r0}…{r1}, "
                            "столбцы {c0}…{c1}.",
        "status_pool_src": "Окно пулинга поставлено на клетку Y({r},{c}).",
        "status_pool_unused": "Клетка Y({r},{c}) не попала ни в одно окно: снизу и "
                              "справа целое окно {w}×{w} не влезает.",
        # --- сообщения об ошибках
        "err_matrix_empty": "Матрица пуста: наберите хотя бы {lo} строк по {lo} чисел.",
        "err_matrix_number": "В матрице не число: «{value}». Нужны целые или дробные "
                             "числа, дробная часть — через точку.",
        "err_matrix_ragged": "Строки матрицы разной длины: в каждом ряду должно быть "
                             "одинаковое число чисел.",
        "err_matrix_size": "Размер матрицы вне диапазона: нужно от {lo}×{lo} до "
                           "{hi}×{hi}, а получилось {rows}×{cols}.",
        "err_matrix_limit": "Число {value} вне дозволенного: модуль не больше {lim}.",
        "err_kernel_number": "В ядре не число: «{value}». Вес — целое или дробное "
                             "число.",
        "err_kernel_limit": "Вес {value} вне дозволенного: модуль не больше {lim}.",
        "err_bias_number": "Смещение должно быть числом: «{value}» не подходит.",
        "err_bias_limit": "Смещение {value} слишком велико: модуль не больше {lim}.",
        "err_speed_number": "Пауза между позициями должна быть числом.",
        "err_impossible": "окно {k}×{k} не влезает в вход {rows}×{cols} при шаге "
                          "{s} и заполнении {p}",
        "bad_hint": "Исправьте поле: предыдущее значение осталось в расчёте.",
        "kernel_no_flip": "Ядро применяется без переворота: это взаимная "
                          "корреляция — именно так свёртку делают библиотеки CNN.",
        "act_short_none": "без активации",
        "act_short_relu": "ReLU",
        "exp_preview_title": "Что получится",
        "pad_short_0": "P = 0",
        "pad_short_1": "P = 1",
        "stride_short_1": "S = 1",
        "stride_short_2": "S = 2",
    },
    "en": {
        "app_title": "CNN Convolution Lab — how a kernel moves over an input",
        "subtitle": ("Input matrix → kernel → element-wise products → sum → bias → "
                     "activation → feature map → pooling (max or average) → final "
                     "pooled map. Every number is on screen at the moment it "
                     "appears: the kernel is applied without flipping, exactly as "
                     "convolution libraries do it."),
        "box_language": "Interface language",
        "lang_ru": "Русский",
        "lang_en": "English",
        "tab_conv": "Convolution",
        "tab_pool": "Pooling",
        "tab_compare": "Kernel comparison",
        "tab_experiments": "Experiments",
        "tab_formula": "Formulas",
        "tab_help": "Help",
        "box_input": "Input matrix X",
        "lbl_input_preset": "Educational picture",
        "input_vedge_name": "Vertical edge",
        "input_vedge_desc": "Dark columns on the left (2), bright ones on the right "
                            "(8). The intensity jump runs horizontally — exactly "
                            "where a vertical edge looks for it.",
        "input_hedge_name": "Horizontal edge",
        "input_hedge_desc": "Dark rows on top (2), bright ones below (8). The "
                            "intensity jump runs vertically.",
        "input_center_name": "Bright center",
        "input_center_desc": "A 3×3 block of nines on a background of ones: the "
                             "patch has four borders, and the response appears "
                             "wherever the window covers one of them.",
        "input_dot_name": "Single dot",
        "input_dot_desc": "One nine in the middle, zeros around it. The feature map "
                          "reproduces the kernel itself at the convolution stride — "
                          "convenient for seeing what a weight means.",
        "input_cross_name": "Cross",
        "input_cross_desc": "Middle row and middle column are bright: a vertical and "
                            "a horizontal edge at the same time.",
        "input_shape_name": "Simple shape",
        "input_shape_desc": "A bright frame around a dark center: two borders above "
                            "and below, two on the left and right.",
        "input_custom_name": "Custom matrix",
        "input_custom_desc": "The matrix is typed by hand in the field below. Sizes "
                             "from {lo}×{lo} to {hi}×{hi}, values from -{lim} to "
                             "{lim}.",
        "box_matrix": "Custom matrix",
        "hint_matrix": "One matrix row per line (or separated by semicolons), values "
                       "separated by spaces or commas. Decimals use a dot: 2.5 is one "
                       "value, while 2,5 is two. Empty input is not accepted.",
        "btn_apply_matrix": "Apply matrix",
        "matrix_size": "Input size: {rows}×{cols}",
        "box_kernel": "Convolution kernel K",
        "lbl_kernel_preset": "Kernel preset",
        "kernel_vedge_name": "Vertical Edge",
        "kernel_vedge_desc": "This kernel compares neighbouring columns: plus on the "
                             "right, minus on the left. It tends to respond to "
                             "intensity changes across columns and stays quiet on a "
                             "flat area.",
        "kernel_hedge_name": "Horizontal Edge",
        "kernel_hedge_desc": "The mirrored set of weights: bottom rows plus, top rows "
                             "minus. The response grows where intensity changes from "
                             "row to row.",
        "kernel_sharpen_name": "Sharpen",
        "kernel_sharpen_desc": "The center weighs more than the neighbours, which are "
                              "taken with a minus sign. Local contrast is "
                              "emphasized: an edge gets sharper, a flat area stays "
                              "flat.",
        "kernel_blur_name": "Blur",
        "kernel_blur_desc": "Neighbours are summed with weights 1, 2 and 4 — a "
                            "smoothed average over the neighborhood. Dividing by the "
                            "weight sum (16) is left out here, so the map simply "
                            "becomes smoother and brighter.",
        "kernel_identity_name": "Identity",
        "kernel_identity_desc": "A single one in the center: the map repeats the input "
                                "itself, shrunk by the stride. Good for seeing what "
                                "the window does without any filter.",
        "kernel_custom_name": "Custom kernel",
        "kernel_custom_desc": "A hand-set kernel. Nothing is expected from it but what "
                              "the numbers say: convolution sums the products of the "
                              "window with these weights.",
        "hint_kernel": "The nine fields are the kernel: left to right, then down. "
                       "Editing any field moves the preset to “Custom kernel”.",
        "lbl_bias": "Bias b",
        "hint_bias": "The same number is added to every sum of products: b = 0 by "
                     "default, so at first it changes nothing.",
        "box_geometry": "Stride and padding",
        "lbl_stride": "Stride S",
        "stride_1": "1 — window moves one cell",
        "stride_2": "2 — window jumps a cell",
        "lbl_padding": "Padding P",
        "pad_0": "0 — no padding (valid)",
        "pad_1": "1 — zero border (same)",
        "hint_stride": "With S = 2 the window origin leaps two cells each time, the "
                       "positions in between are not computed: the map is smaller.",
        "hint_padding": "P = 1 adds a border of zeros around the input. The window can "
                        "now stand at the very edge of the picture, and the map grows.",
        "box_activation": "Activation after convolution",
        "act_none": "None — no activation",
        "act_relu": "ReLU — max(0, z)",
        "hint_activation": "Only one activation lives here, and only to show the road "
                           "“sum → ReLU → feature-map cell”. Activation functions as a "
                           "topic belong to a separate lab of the series.",
        "box_pool": "Pooling after activation",
        "lbl_pool_mode": "Pooling type",
        "pool_max": "Max — largest value in the window",
        "pool_avg": "Average — mean of the window",
        "lbl_pool_window": "Pooling window",
        "pool_w2": "2×2, non-overlapping",
        "pool_w3": "3×3, non-overlapping",
        "hint_pool": "Pooling is the last link of the chain: it takes the feature map "
                     "after the activation and shrinks it, without recomputing "
                     "anything with the convolution window.",
        "hint_pool_window": "The window steps by its own size, so no cell belongs to "
                            "two windows. An incomplete window on the right or bottom "
                            "is dropped as a whole: those cells are drawn in gray.",
        "btn_reset_all": "Reset everything",
        "cap_input": "Input X",
        "cap_input_padded": "Input X with a zero border, P = 1",
        "cap_kernel": "Kernel K",
        "cap_products": "Element-wise products X·K",
        "cap_map": "Feature map Y",
        "legend_window": "Gold frame — the active convolution window",
        "legend_visited": "Dot — a window origin at the current stride",
        "legend_skipped": "✕ — an origin the stride skips",
        "legend_negative": "Warm cells — negative values of the map",
        "legend_positive": "Cool cells — positive values",
        "window_coords": "Window: rows {r0}…{r1}, columns {c0}…{c1}",
        "window_coords_pad": "Window: padded rows {r0}…{r1}, padded columns {c0}…{c1} "
                             "(zeros stand beyond the input)",
        "box_step": "Step by step",
        "btn_prev": "← Previous",
        "btn_next": "Next →",
        "btn_step_reset": "Reset step",
        "position_label": "Position {index} of {total}",
        "box_scan": "Automatic scan",
        "btn_scan_start": "Start",
        "btn_scan_pause": "Pause",
        "btn_scan_reset": "Reset scan",
        "lbl_speed": "Delay per position, ms",
        "calc_title": "Calculation at this position",
        "calc_region": "Input window X({i},{j}) … X({i2},{j2}):",
        "calc_kernel_row": "Kernel rows K — not flipped:",
        "calc_products": "Nine products, one line per window row:",
        "calc_sum_head": "Σ — sum of products:",
        "calc_sum_line": "{expr} = {total}",
        "calc_bias_head": "Sum with bias:",
        "calc_bias_line": "z = {total} + {bias} = {pre}",
        "calc_relu_head": "ReLU activation:",
        "calc_relu_line": "Y({i},{j}) = ReLU(z) = {relu} = {out}",
        "calc_none_head": "Activation:",
        "calc_none_line": "Y({i},{j}) = z = {out} (None changes nothing)",
        "calc_cell_head": "Feature-map cell:",
        "calc_cell_line": "Y({i},{j}) = {out}",
        "calc_map_missing": "The window does not fit here: this position is not "
                            "computed.",
        "box_outsize": "Feature-map size",
        "outsize_formula": "output = floor((input + 2P − K) / S) + 1",
        "outsize_rows": "Rows: floor(({n} + 2·{p} − {k}) / {s}) + 1 = {o}",
        "outsize_cols": "Columns: floor(({n} + 2·{p} − {k}) / {s}) + 1 = {o}",
        "outsize_square": "floor(({n} + 2·{p} − {k}) / {s}) + 1 = {o}",
        "outsize_map": "The map comes out {rows}×{cols}: {total} positions in all.",
        "outsize_impossible": "With S = {s} and P = {p} the {k}×{k} window does not "
                              "fit into a {rows}×{cols} input: no positions at all.",
        "pool_intro": ("Pooling sits after the activation and before the final map: "
                       "the window covers the feature map without overlap, and every "
                       "cell of Z is one number out of its window — the maximum or "
                       "the mean."),
        "cap_pool_src": "Feature map Y — pooling input",
        "cap_pool_out": "Final pooled map Z",
        "legend_pool_window": "Gold frame — the selected pooling window",
        "legend_pool_star": "★ — a cell holding the maximum; bold — the cell the "
                            "value came from",
        "legend_pool_unused": "Gray cells — not covered by any window",
        "legend_pool_selected": "Gold frame — the selected cell of Z",
        "pool_summary": "Y {rows}×{cols} → window {w}×{w} → Z {prows}×{pcols}, "
                        "cells: {ptotal}.",
        "pool_no_map": "Pooling has nothing to work on: map Y is {rows}×{cols} and no "
                       "whole {w}×{w} window fits anywhere.",
        "pool_outsize_rows": "Rows: floor(({n} − {w}) / {w}) + 1 = {o}",
        "pool_outsize_cols": "Columns: floor(({n} − {w}) / {w}) + 1 = {o}",
        "pool_outsize_square": "floor(({n} − {w}) / {w}) + 1 = {o}",
        "pool_calc_title": "Calculation of this pooling cell",
        "pool_window_coords": "Window {w}×{w}: rows {r0}…{r1}, columns {c0}…{c1}",
        "pool_values_head": "Window values in order (rows first, then columns):",
        "pool_max_head": "Max — the largest value of the window:",
        "pool_max_line": "Z({i},{j}) = max({values}) = {out}",
        "pool_winner_line": "The maximum {best} stands in cell Y({r},{c}) — the first "
                            "in row-major order among equal ones.",
        "pool_tie_line": "The window holds {n} such maxima: {cells}. The value is "
                         "taken from the first one — cell Y({r},{c}).",
        "pool_avg_head": "Average — the mean over the window:",
        "pool_avg_line": "Z({i},{j}) = ({expr}) / {count} = {out}",
        "pool_avg_note": "Shown rounded to {d} digits: the exact quotient can be "
                         "longer. This shows up most often in a {w}×{w} window, where "
                         "the divisor is not a power of ten.",
        "pool_cell_head": "Cell of the final map:",
        "pool_cell_line": "Z({i},{j}) = {out}",
        "pool_map_missing": "No whole {w}×{w} window fits: pooling computes no cells "
                            "at all.",
        "pool_unused": "Left without a window at the bottom and right — rows: "
                       "{rows}, columns: {cols}, cells: {cells}.",
        "pool_all_used": "Every cell of the map is covered: {rows}×{cols} divides by "
                         "{w}×{w} with no remainder.",
        "cmp_intro": "One input, five kernels: only the kernel changes, so any "
                     "difference between the maps is the work of the filter, not of "
                     "the data.",
        "cmp_input_line": "Input: {name} ({rows}×{cols}), stride S = {s}, padding "
                          "P = {p}, bias b = {b}, activation: {a}.",
        "cmp_map_title": "Kernel: {name}",
        "cmp_map_size": "map {rows}×{cols}",
        "cmp_max": "maximum |value| = {v}",
        "cmp_minmax": "values from {lo} to {hi}",
        "cmp_note": "Click a map to move to that position on the Convolution tab and "
                    "take it apart number by number.",
        "exp_intro": "Each experiment assembles a finished educational set: input, "
                     "kernel, stride, padding, bias and activation. The numbers are "
                     "deterministic: the same click always gives the same result.",
        "exp_apply": "Open on the Convolution tab",
        "exp_result": "Map {rows}×{cols}; selected position {index} of {total}; "
                      "value Y = {value}.",
        "exp_edge_v_name": "E1 — vertical edges",
        "exp_edge_v_goal": "A vertical-edge input with the matching kernel. A window "
                           "that catches the brightness jump gives 18; windows that "
                           "sit entirely in the dark or entirely in the light give 0. "
                           "This is what “a filter responds to an edge” looks like.",
        "exp_edge_h_goal": "The same trick across: a horizontal edge and a kernel that "
                           "compares neighbouring rows. The zero middle row of the "
                           "kernel takes no part in the sum.",
        "exp_edge_h_name": "E2 — horizontal edges",
        "exp_smooth_name": "E3 — smoothing",
        "exp_smooth_goal": "The blur kernel on the cross: the map becomes smoother and "
                           "its values are larger than the input, because the weights "
                           "add up nine neighbours and never divide by 16.",
        "exp_sharpen_name": "E4 — local contrast",
        "exp_sharpen_goal": "The sharpen kernel on the bright patch: inside the patch "
                            "the map equals the input (9), on the border it rises to "
                            "17, and just outside it goes negative (-7). The selected "
                            "cell holds 17.",
        "exp_stride_name": "E5 — stride 2",
        "exp_stride_goal": "The same input and kernel, but stride 2. The window leaps "
                           "over a cell, half of the positions are gone, the map is "
                           "3×3 instead of 5×5 — check that on the ✕ marks in the "
                           "input.",
        "exp_padding_name": "E6 — zero padding",
        "exp_padding_goal": "The same set with P = 1 and without it: the map grows from "
                            "5×5 to 7×7 because the window now stands at the very edge "
                            "of the picture. The corner position gives 4: two weights "
                            "multiplied the zeros of the frame.",
        "exp_relu_name": "E7 — ReLU after convolution",
        "exp_relu_goal": "The vertical-edge kernel on the bright patch: left of the "
                         "border the sum is 24, right of it -24. ReLU zeroes the "
                         "negative half, so only one side of the edge survives on the "
                         "map. The selected cell is the one with z = -24 before "
                         "activation.",
        "formula_core_head": "What the application computes",
        "formula_core": "z(i,j) = ΣΣ X(i+m, j+n) · K(m,n) + b",
        "formula_core_note": "The double sum runs over m = 0…K−1 and n = 0…K−1, that "
                             "is over all nine cells of the window. The kernel is not "
                             "flipped: this is cross-correlation, and that is how "
                             "convolution libraries name the operation.",
        "formula_symbols_head": "Notation",
        "formula_out_head": "Size of the feature map",
        "formula_out": "output = floor((input + 2P − K) / S) + 1",
        "formula_out_note": "The substitution below uses live numbers of the current "
                            "configuration: changing stride or padding rewrites the "
                            "line at once.",
        "formula_current_head": "The current configuration in numbers",
        "symbol_x": "X — the input matrix, the brightness of a cell (rows, columns)",
        "symbol_k": "K — the convolution kernel, {k}×{k} weights",
        "symbol_ij": "i, j — coordinates of a cell of the feature map Y",
        "symbol_mn": "m, n — coordinates inside the window and inside the kernel",
        "symbol_sum": "ΣΣ — the sum of the nine products of the window and the kernel",
        "symbol_b": "b — bias, one number for the whole map",
        "symbol_z": "z — the sum with bias: the value before activation",
        "symbol_y": "Y — the feature map: Y = z or Y = ReLU(z)",
        "symbol_s": "S — stride: how many cells the window shifts",
        "symbol_p": "P — padding: the width of the zero border around the input",
        "formula_relu": "ReLU(z) = max(0, z)",
        "formula_relu_note": "The activation stays a single one, and only as the last "
                             "step of the chain: the feature map after it.",
        "help_terms_title": "Convolution or cross-correlation",
        "help_terms_text": ("Mathematics flips the kernel by 180° for a convolution; "
                            "cross-correlation does not. Convolution libraries call "
                            "the unflipped operation “convolution”, and this "
                            "application does the same: a map cell is the sum of the "
                            "products of X(i+m, j+n) and K(m,n) in the same "
                            "coordinates.\n"
                            "The practical difference is small and it is stated "
                            "openly: with a flipped kernel symmetric kernels would "
                            "give the same map, while an asymmetric one would point at "
                            "the edge in the other direction. Nothing here pretends "
                            "that a flip happens."),
        "help_chain_title": "What is on the screen",
        "help_chain_text": ("Five things stand side by side in the order of the chain: "
                            "the input X, the kernel K, the nine element-wise "
                            "products, the feature map Y, and the numbers in between.\n"
                            "The gold frame on the input is the window in work. The "
                            "same cell is highlighted on the map to the right, and the "
                            "calculation panel shows all nine products, the sum, the "
                            "bias and the activation for that position.\n"
                            "Click a map cell and the window moves there. Click an "
                            "input cell and the window is placed so that it covers the "
                            "cell you clicked.\n"
                            "The chain continues on the Pooling tab: there the map Y "
                            "is shrunk by a 2×2 or 3×3 window into the final map Z, "
                            "and every number is visible again."),
        "help_step_title": "Step-by-step mode",
        "help_step_text": ("“Next” and “Previous” move the window along a row from "
                           "left to right, then down to the next row. The order of "
                           "positions matches the order of the feature map, so the "
                           "position number and the cell coordinates always agree.\n"
                           "Stepping does not depend on animation: a button recomputes "
                           "the position at once, and you may go back and stop "
                           "anywhere."),
        "help_stride_title": "Stride and padding in the picture",
        "help_stride_text": ("Stride: the window origin shifts by S cells. With S = 2 "
                             "the input keeps dots where the window stood and ✕ marks "
                             "where it could have stood but did not. Half of the "
                             "positions disappear and the map shrinks.\n"
                             "Padding: a border of zeros appears around the input. The "
                             "window starts standing at the very edge of the picture, "
                             "some of its weights multiply zeros — and the map grows. "
                             "Border cells are drawn separately so that nobody confuses "
                             "them with input brightness."),
        "help_bias_title": "Bias and ReLU",
        "help_bias_text": ("Convolution is not only the sum of products: the same "
                           "number b is added to it. The calculation panel prints "
                           "“Σ + b” as its own line, so the two steps stay visibly "
                           "different.\n"
                           "ReLU is kept alone on purpose: it shows only “z → max(0, z) "
                           "→ feature-map cell”. A catalog of activation functions is "
                           "the task of another project of the series."),
        "help_pool_title": "Pooling: max and average",
        "help_pool_text": ("Pooling shrinks the feature map. The window steps across "
                           "it by its own size, so cells never overlap: Max takes the "
                           "largest value in the window, Average adds all the values "
                           "of the window and divides by how many of them there are.\n"
                           "An incomplete window on the right or at the bottom is "
                           "dropped as a whole. The sizes come from the same floor "
                           "formula as the convolution, with the step equal to the "
                           "window: floor((size − window) / window) + 1. Cells left "
                           "without a window are drawn in gray, and their number is "
                           "printed under the map.\n"
                           "Max has one nuance: when the maximum stands in several "
                           "cells of the window, all of them are shown, but the value "
                           "is taken from the first one in row-major order.\n"
                           "The displayed average is rounded to four digits: 7/9 "
                           "appears as 0.7778, so in a 3×3 window (divisor 9) the "
                           "rounding is the most visible."),
        "help_kernels_title": "What a kernel notices",
        "help_kernels_text": ("None of these kernels searches for anything on its own: "
                              "they are sets of weights, and what appears on the map "
                              "follows from the numbers.\n"
                              "• Vertical Edge — tends to respond to the difference "
                              "between neighbouring columns; close to zero on a flat "
                              "area.\n"
                              "• Horizontal Edge — the same across rows.\n"
                              "• Blur — sums the neighbours, so the map gets smoother.\n"
                              "• Sharpen — the center weight is larger, local contrast "
                              "is emphasized.\n"
                              "• Identity — passes the input through as it is."),
        "help_scope_title": "What is not here — and why",
        "help_scope_text": ("No training, no gradients, no loss functions, no stacked "
                            "convolution layers, no quality metrics, no downloaded "
                            "images. Pooling here is one step at the end of the chain; "
                            "a pooling system of its own, with overlapping windows and "
                            "many strategies, belongs to another project of the "
                            "series.\n"
                            "Exactly one convolution pass and one pooling window live "
                            "here: one input, one kernel, one feature map, one final "
                            "map."),
        "help_keys_title": "Keyboard",
        "help_keys_text": ("← and → — previous and next convolution position.\n"
                           "Home — back to the first position.\n"
                           "Space — start and pause the automatic scan.\n"
                           "Enter in the matrix field — apply what you typed."),
        "status_start": "Ready: input “{input}”, kernel “{kernel}”, map {rows}×{cols}.",
        "status_lang": "Interface language: {name}. Nothing in the state changed.",
        "status_step": "Position {index} of {total}: Y({i},{j}) = {value}.",
        "status_edge_first": "This is the first position: nowhere to go back.",
        "status_edge_last": "This is the last position: nowhere to go forward.",
        "status_reset": "Reset: the first cell of the map is selected, scan stopped.",
        "status_scan": "Scan running: {speed} ms per position.",
        "status_scan_pause": "Scan paused: position {index} of {total}.",
        "status_scan_end": "The scan reached the last position.",
        "status_input_preset": "Input: {name}.",
        "status_kernel_preset": "Kernel: {name}.",
        "status_kernel_custom": "The kernel is typed by hand: the preset switched to "
                                "“Custom kernel”.",
        "status_stride": "Stride S = {s}: {total} positions now.",
        "status_pad": "Padding P = {p}: input with border {rows}×{cols}, {total} "
                      "positions.",
        "status_bias": "Bias b = {b}.",
        "status_activation": "Activation: {name}.",
        "status_matrix_applied": "Matrix {rows}×{cols} accepted.",
        "status_experiment": "Experiment “{name}”: {goal}",
        "status_impossible": "Impossible configuration: {reason}",
        "status_cell_map": "Feature-map cell Y({i},{j}) selected: window rows "
                           "{r0}…{r1}, columns {c0}…{c1}.",
        "status_cell_input": "The window was moved onto input cell ({r},{c}).",
        "status_pool_mode": "Pooling: {name}. The final map Z is {rows}×{cols}.",
        "status_pool_window": "Pooling window {w}×{w}: final map Z {rows}×{cols}, "
                              "cells: {total}.",
        "status_pool_cell": "Cell Z({i},{j}) selected: window rows {r0}…{r1}, "
                            "columns {c0}…{c1}.",
        "status_pool_src": "The pooling window was moved onto map cell Y({r},{c}).",
        "status_pool_unused": "Cell Y({r},{c}) is covered by no window: a whole "
                              "{w}×{w} window does not fit at the bottom or right.",
        "err_matrix_empty": "The matrix is empty: type at least {lo} rows of {lo} "
                            "values.",
        "err_matrix_number": "Not a number in the matrix: “{value}”. Integers or "
                             "decimals, with a dot as the decimal separator.",
        "err_matrix_ragged": "Rows of the matrix have different lengths: every row "
                             "needs the same number of values.",
        "err_matrix_size": "The matrix size is out of range: {lo}×{lo} to {hi}×{hi} is "
                           "accepted, you gave {rows}×{cols}.",
        "err_matrix_limit": "{value} is out of range: the absolute value may not "
                            "exceed {lim}.",
        "err_kernel_number": "Not a number in the kernel: “{value}”. A weight is an "
                             "integer or a decimal.",
        "err_kernel_limit": "Weight {value} is out of range: the absolute value may "
                            "not exceed {lim}.",
        "err_bias_number": "The bias has to be a number: “{value}” does not work.",
        "err_bias_limit": "Bias {value} is too large: the absolute value may not "
                          "exceed {lim}.",
        "err_speed_number": "The delay between positions has to be a number.",
        "err_impossible": "the {k}×{k} window does not fit into a {rows}×{cols} input "
                          "at stride {s} and padding {p}",
        "bad_hint": "Fix the field: the previous value is still in the calculation.",
        "kernel_no_flip": "The kernel is applied without flipping: this is "
                          "cross-correlation, and that is what CNN libraries call "
                          "convolution.",
        "act_short_none": "no activation",
        "act_short_relu": "ReLU",
        "exp_preview_title": "What comes out",
        "pad_short_0": "P = 0",
        "pad_short_1": "P = 1",
        "stride_short_1": "S = 1",
        "stride_short_2": "S = 2",
    },
}


# ----------------------------------------------------------------------
# Раздел 6. Учебные эксперименты
# ----------------------------------------------------------------------

EXPERIMENTS = (
    {
        "key": "edge_v",
        "input_key": "vedge",
        "kernel_key": "vedge",
        "stride": 1,
        "pad": 0,
        "bias": 0.0,
        "activation": "none",
        "out_i": 0,
        "out_j": 1,
    },
    {
        "key": "edge_h",
        "input_key": "hedge",
        "kernel_key": "hedge",
        "stride": 1,
        "pad": 0,
        "bias": 0.0,
        "activation": "none",
        "out_i": 1,
        "out_j": 0,
    },
    {
        "key": "smooth",
        "input_key": "cross",
        "kernel_key": "blur",
        "stride": 1,
        "pad": 0,
        "bias": 0.0,
        "activation": "none",
        "out_i": 3,
        "out_j": 3,
    },
    {
        "key": "sharpen",
        "input_key": "center",
        "kernel_key": "sharpen",
        "stride": 1,
        "pad": 0,
        "bias": 0.0,
        "activation": "none",
        "out_i": 1,
        "out_j": 2,
    },
    {
        "key": "stride",
        "input_key": "vedge",
        "kernel_key": "vedge",
        "stride": 2,
        "pad": 0,
        "bias": 0.0,
        "activation": "none",
        "out_i": 0,
        "out_j": 1,
    },
    {
        "key": "padding",
        "input_key": "vedge",
        "kernel_key": "vedge",
        "stride": 1,
        "pad": 1,
        "bias": 0.0,
        "activation": "none",
        "out_i": 0,
        "out_j": 0,
    },
    {
        "key": "relu",
        "input_key": "center",
        "kernel_key": "vedge",
        "stride": 1,
        "pad": 0,
        "bias": 0.0,
        "activation": "relu",
        "out_i": 2,
        "out_j": 3,
    },
)

EXPERIMENT_BY_KEY = {entry["key"]: entry for entry in EXPERIMENTS}


# ----------------------------------------------------------------------
# Раздел 7. Справка
# ----------------------------------------------------------------------

HELP_KEYS = (
    "terms",
    "chain",
    "step",
    "stride",
    "bias",
    "pool",
    "kernels",
    "scope",
    "keys",
)


# ----------------------------------------------------------------------
# Раздел 8. Холсты: вход, ядро, произведения, карта признаков
# ----------------------------------------------------------------------

def cell_spec(text="", bg=COL_PAPER, fg=None, edge=None, edge_width=1, note="",
              mark="", bold=False):
    """Описание клетки сетки: все четыре матрицы рисуются одним механизмом."""
    return {
        "text": text,
        "bg": bg,
        "fg": fg if fg else text_on(bg),
        "edge": edge,
        "edge_width": edge_width,
        "note": note,
        "mark": mark,
        "bold": bold,
    }


class GridCanvas(tk.Canvas):
    """Сетка клеток: число, фон по шкале, рамка, подпись под числом, метка.

    Размер холст считает сам по числу строк и столбцов, поэтому матрица
    3×3 и карта 7×7 помещаются одинаково аккуратно. Клик возвращает
    координаты клетки (строка, столбец) в коллбэк.
    """

    def __init__(self, master, cell_size=CELL, on_click=None, fonts=None,
                 note_font="tiny", number_font="mono", **kwargs):
        super().__init__(master, highlightthickness=0, borderwidth=0,
                         bg=COL_BG, **kwargs)
        self.cs = cell_size
        self.on_click = on_click
        self.fonts = fonts or {}
        self.axes = None
        self.note_font = note_font
        self.number_font = number_font
        self.margin = 8
        self.grid_rows = 0
        self.grid_cols = 0
        self.bind("<Button-1>", self._on_press)

    def _font(self, name, fallback):
        return self.fonts.get(name, fallback)

    def render(self, spec):
        self.delete("all")
        self.grid_rows = len(spec)
        self.grid_cols = len(spec[0]) if self.grid_rows else 0
        # Подписи строк и столбцев рисуются слева и сверху, поэтому поле
        # слева и сверху шире, когда подписи есть.
        self.margin = 26 if self._has_axes() else 8
        width = self.grid_cols * self.cs + 2 * self.margin
        height = self.grid_rows * self.cs + 2 * self.margin
        self.configure(width=width, height=height,
                       bg=COL_BG, highlightbackground=COL_BG)
        for i, row in enumerate(spec):
            for j, entry in enumerate(row):
                self._draw_cell(i, j, entry)
        self._draw_axes()

    def _has_axes(self):
        return bool(self.axes) and bool(self.axes[0] or self.axes[1])

    def _draw_cell(self, i, j, entry):
        x0 = self.margin + j * self.cs
        y0 = self.margin + i * self.cs
        x1, y1 = x0 + self.cs, y0 + self.cs
        outline = entry["edge"] or COL_FRAME
        self.create_rectangle(x0 + 1, y0 + 1, x1 - 1, y1 - 1,
                              fill=entry["bg"], outline=outline,
                              width=entry["edge_width"])
        middle_x, middle_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        number_font = self._font(self.number_font, ("TkFixedFont", 11, "bold"))
        if entry["bold"]:
            number_font = self._font("monob", number_font)
        note = entry["note"]
        if note:
            self.create_text(middle_x, middle_y - 7, text=entry["text"],
                             fill=entry["fg"], font=number_font)
            self.create_text(middle_x, middle_y + 9, text=note,
                             fill=entry["fg"],
                             font=self._font(self.note_font, ("TkDefaultFont",
                                                              8)),
                             )
        else:
            self.create_text(middle_x, middle_y, text=entry["text"],
                             fill=entry["fg"], font=number_font)
        if entry["mark"]:
            # Метка живёт в левом нижнем углу: число по центру и номер
            # позиции в углу не должны читаться как одна степень.
            self.create_text(x0 + 4, y1 - 3, text=entry["mark"], anchor="sw",
                             fill=entry["fg"],
                             font=self._font(self.note_font,
                                             ("TkDefaultFont", 8)))

    def _draw_axes(self):
        if not self._has_axes():
            return
        row_labels, column_labels = self.axes
        tiny = self._font("tiny", ("TkDefaultFont", 9))
        for j, label in enumerate(column_labels):
            self.create_text(self.margin + j * self.cs + self.cs / 2.0,
                             self.margin - 12, text=label, fill=COL_DIM,
                             font=tiny)
        for i, label in enumerate(row_labels):
            self.create_text(self.margin - 6,
                             self.margin + i * self.cs + self.cs / 2.0,
                             text=label, fill=COL_DIM, font=tiny,
                             anchor="e")

    def _on_press(self, event):
        if not self.on_click or not self.grid_rows:
            return
        column = int((event.x - self.margin) // self.cs)
        row = int((event.y - self.margin) // self.cs)
        if 0 <= row < self.grid_rows and 0 <= column < self.grid_cols:
            self.on_click(row, column)


class ScrollFrame(ttk.Frame):
    """Прокручиваемая область: контент длиннее окна — выезжают полосы.

    Колонка параметров и широкие вкладки живут на одном механизме. Полоса
    прокрутки по горизонтали нужна вкладке свёртки: там пять колонок, и на
    маленьком экране цепочка «вход → ядро → расчёт → карта» не помещается,
    а обрезаться она не имеет права.
    """

    def __init__(self, master, horizontal=False):
        super().__init__(master)
        self.horizontal = horizontal
        self.viewport = None
        self.fitted = None
        self.relayout_job = None
        self.settling = 0
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                                bg=COL_BG)
        self.bar = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.xbar = None
        if horizontal:
            self.xbar = ttk.Scrollbar(self, orient="horizontal",
                                      command=self.canvas.xview)
        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window(
            0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self.relayout)
        self.canvas.configure(yscrollcommand=self.bar.set,
                              xscrollcommand=self.xbar.set if self.xbar else None)
        if self.xbar:
            self.xbar.pack(side="bottom", fill="x")
        self.bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_viewport)

    def _on_viewport(self, event):
        self.viewport = (event.width, event.height)
        self.settling = 0
        self.relayout()

    def relayout(self, event=None):
        """Ширина контента — не меньше окна, высота — как просит контент.

        Высоту не задаём вовсе: при width/height = 0 окно холста берёт
        запрошенный размер фрейма, поэтому контент не может оказаться сжат
        по вертикали, даже если подписи перенеслись позже. Ширину держим
        под контролем: цепочка из пяти колонок на узком экране шире окна, и
        тогда выезжает горизонтальная полоса, а не обрезается содержимое.
        """
        if not self.viewport:
            return
        width = self.viewport[0]
        if self.horizontal:
            width = max(width, self.inner.winfo_reqwidth())
        if self.fitted != width:
            self.fitted = width
            self.canvas.itemconfigure(self.inner_id, width=width)
            if self.settling < 4 and self.relayout_job is None:
                self.settling += 1
                self.relayout_job = self.after_idle(self._relayout_later)
        box = self.canvas.bbox("all")
        if box is not None:
            self.canvas.configure(scrollregion=box)

    def _relayout_later(self):
        self.relayout_job = None
        self.relayout()

    def apply_size(self):
        """Вызывается из полной перерисовки: пересчитать размер и полосы."""
        self.settling = 0
        self.relayout()

    def _on_wheel(self, event):
        """Колесо крутит эту область: сначала вниз, потом вбок."""
        step = -1 if event.delta < 0 else 1
        if event.state & 1 and self.xbar is not None:
            self.canvas.xview_scroll(step, "units")
            return
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(step, "units")
        elif self.xbar is not None:
            self.canvas.xview_scroll(step, "units")


# ----------------------------------------------------------------------
# Раздел 9. Главное окно
# ----------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title(T("app_title"))
        self.configure(bg=COL_BG)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.minsize(min(MIN_WIDTH, screen_w - 40),
                     min(MIN_HEIGHT, screen_h - 60))
        self.geometry("%dx%d" % (
            min(max(MIN_WIDTH + 120, int(screen_w * 0.86)), screen_w - 20),
            min(max(MIN_HEIGHT + 100, int(screen_h * 0.88)), screen_h - 20)))
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        base = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")

        def make_font(family, size, weight="normal"):
            return tkfont.Font(family=family, size=size, weight=weight)

        self.fonts = {
            "title": make_font(base, 16, "bold"),
            "sub": make_font(base, 13),
            "midb": make_font(base, 13, "bold"),
            "small": make_font(base, 12),
            "smallb": make_font(base, 12, "bold"),
            "tiny": make_font(base, 10),
            "tinymb": make_font(base, 9, "bold"),
            "mono": make_font(mono, 12, "bold"),
            "monob": make_font(mono, 12, "bold"),
            "monos": make_font(mono, 10),
            # Мелкий моноширинный: клетки-превью на вкладках сравнения и
            # экспериментов, где трёхзначное число не должно залезать на
            # соседа.
            "monoxs": make_font(mono, 8),
        }
        self.option_add("*Font", (base, 12))
        self._setup_styles()

        # Состояние. Язык его не трогает: переключение только переклеивает
        # подписи, числа и выбор остаются на месте.
        self.input_key = DEFAULT_INPUT_KEY
        self.values = build_input(DEFAULT_INPUT_KEY)
        self.kernel_key = DEFAULT_KERNEL_KEY
        self.kernel = [list(map(float, row))
                       for row in KERNEL_BY_KEY[self.kernel_key]["values"]]
        self.stride = DEFAULT_STRIDE
        self.pad = DEFAULT_PAD
        self.bias = DEFAULT_BIAS
        self.activation = DEFAULT_ACTIVATION
        self.out_i = 0
        self.out_j = 0
        self.result = None
        self.cell = None
        self.pool_mode = DEFAULT_POOL_MODE
        self.pool_window = DEFAULT_POOL_WINDOW
        self.pool_i = 0
        self.pool_j = 0
        self.pool_result = None
        self.pool_step = None
        self.experiment_key = EXPERIMENTS[0]["key"]
        self.help_key = HELP_KEYS[0]
        self.scan_running = False
        self.scan_job = None
        self.scan_speed = 260
        self._text_refs = []
        self._fluid_refs = []
        self._scrolls = []
        self._tabs = []
        self._entry_refs = []
        self.var_status = tk.StringVar()
        self.var_lang = tk.StringVar(value=LANG)
        self.var_position = tk.StringVar()
        self.var_coords = tk.StringVar()
        self.var_outsize = tk.StringVar()
        self.var_bias = tk.StringVar(value=fmt(DEFAULT_BIAS))
        self.var_speed = tk.StringVar(value="260")
        self.var_input_combo = tk.StringVar()
        self.var_kernel_combo = tk.StringVar()
        self.kernel_vars = [tk.StringVar() for _ in range(9)]
        self.bad_widgets = []

        self._build()
        self._fill_combos()
        self._fill_experiment_list()
        self._fill_help_list()
        self._sync_entries()
        self.refresh()
        self.show_help()
        self.show_experiment()
        self.bind("<Left>", lambda event: self.on_key_step(event, -1))
        self.bind("<Right>", lambda event: self.on_key_step(event, 1))
        self.bind("<Home>", lambda event: self.on_key_home(event))
        self.bind("<space>", lambda event: self.on_key_scan(event))
        self.bind_all("<MouseWheel>", self._route_wheel)
        self.set_status("status_start", input=self.input_name(),
                        kernel=self.kernel_name(),
                        rows=self.out_rows(), cols=self.out_columns())
        self.after_idle(self.update_fluid_widths)

    # --- оформление ---------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook.Tab", padding=(9, 5))
        style.configure("TNotebook", background=COL_BG, borderwidth=0)
        style.configure("TLabelframe", background=COL_BG)
        style.configure("TLabelframe.Label", background=COL_BG,
                        foreground=COL_HEAD, font=self.fonts["smallb"])
        style.configure("TFrame", background=COL_BG)
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["title"])
        style.configure("Sub.TLabel", background=COL_BG, foreground="#5a5a54",
                        font=self.fonts["small"])
        style.configure("Cap.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["smallb"])
        style.configure("Head.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["smallb"])
        style.configure("Key.TLabel", background=COL_BG, foreground=COL_KEY,
                        font=self.fonts["monob"])
        style.configure("Muted.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=self.fonts["tiny"])
        style.configure("Arrow.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["midb"])
        style.configure("Bad.TLabel", background=COL_BG, foreground=COL_BAD_FG,
                        font=self.fonts["smallb"])
        style.configure("Go.TButton", font=self.fonts["smallb"])
        style.map("Go.TButton", foreground=[("active", COL_HEAD)])
        style.configure("GoodEntry.TEntry", fieldbackground=COL_PAPER)
        style.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG,
                        foreground=COL_BAD_FG)

    def _reg(self, widget, key, attr="text", **kwargs):
        self._text_refs.append((widget, key, attr, kwargs))
        widget.configure(**{attr: T(key, **kwargs)})

    def _box(self, parent, title_key):
        box = ttk.LabelFrame(parent, text=T(title_key), padding=(9, 4))
        box.pack(fill="x", pady=(0, 7))
        self._text_refs.append((box, title_key, "text", {}))
        return box

    def _label(self, parent, key=None, style=None, font=None, text="",
               **kwargs):
        """Метка. key — статичный текст из TEXTS, text — динамический."""
        widget = ttk.Label(parent, style=style, font=font, **kwargs)
        if key:
            self._reg(widget, key)
        else:
            widget.configure(text=text)
        return widget

    def _caption(self, parent, key):
        return self._label(parent, key, style="Cap.TLabel")

    def _muted(self, parent, key, **kwargs):
        return self._label(parent, key, style="Muted.TLabel", **kwargs)

    def _scrollable(self, parent, horizontal=False):
        """Прокручиваемая вкладка: наружу — сама область, внутрь — контент."""
        scroll = ScrollFrame(parent, horizontal=horizontal)
        scroll.pack(fill="both", expand=True)
        self._scrolls.append(scroll)
        return scroll.inner

    def _fluid(self, widget, owner, fraction=0.94, minimum=260):
        """Метка, у которой перенос подстраивается под ширину хозяина."""
        self._fluid_refs.append((widget, owner, fraction, minimum))
        return widget

    def _wheel_text(self, widget):
        """Колесо мыши листает текст под курсором, а не прокрутку вокруг."""
        def scroll(event):
            widget.yview_scroll(int(-event.delta / 120), "units")
            return "break"
        widget.bind("<MouseWheel>", scroll)
        return widget

    # --- сборка -------------------------------------------------------
    def _build(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.columnconfigure(0, minsize=LEFT_COL, weight=0)
        self.columnconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(12, 8, 12, 2))
        header.grid(row=0, column=0, columnspan=2, sticky="we")
        self._label(header, "app_title", style="Title.TLabel").pack(anchor="w")
        subtitle = self._label(header, "subtitle", style="Sub.TLabel",
                               wraplength=1180, justify="left")
        subtitle.pack(anchor="w")
        self._fluid(subtitle, self)

        self.params = ScrollFrame(self)
        self.params.grid(row=1, column=0, sticky="ns", padx=(8, 4), pady=4)
        self._scrolls.append(self.params)
        self._build_params(self.params.inner)

        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        self._build_tab_conv()
        self._build_tab_pool()
        self._build_tab_compare()
        self._build_tab_experiments()
        self._build_tab_formula()
        self._build_tab_help()
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        for widget in (self.txt_matrix, self.txt_calc, self.txt_pool,
                       self.txt_exp, self.txt_formula, self.txt_help):
            self._wheel_text(widget)

        bar = ttk.Frame(self, padding=(12, 2, 16, 4))
        bar.grid(row=2, column=0, columnspan=2, sticky="we")
        status = ttk.Label(bar, textvariable=self.var_status, anchor="w",
                           font=self.fonts["small"], foreground="#3a3a35",
                           justify="left")
        status.pack(side="left", fill="x", expand=True)
        self._fluid(status, self, fraction=0.86)
        # Подпись автора живёт внизу справа постоянно: это подпись, а не
        # учебный текст, и язык её не касается.
        credit = ttk.Label(bar, text="SteveWorks001",
                           font=self.fonts["smallb"], foreground=COL_DIM)
        credit.pack(side="right", padx=(10, 0))
        credit.tkraise()
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        if event.widget is not self:
            return
        self.after_idle(self.update_fluid_widths)

    def _on_tab_change(self, event=None):
        """Скрытая вкладка не знает своих размеров, пока её не покажут.

        Tk не присылает <Configure> неактивной странице, поэтому полосы и
        переносы подписей пересчитываются в момент переключения.
        """
        self.after_idle(self.update_fluid_widths)

    def _route_wheel(self, event):
        """Колесо прокручивает ту область, под которой стоит указатель.

        Полосы жив у колонки параметров и у вкладок, а обработчик один на всё
        окно: иначе одно движение крутило бы две области сразу.
        """
        widget = self.winfo_containing(event.x_root, event.y_root)
        node = widget
        while node is not None and node is not self:
            if isinstance(node, ScrollFrame):
                node._on_wheel(event)
                return
            node = getattr(node, "master", None)

    def update_fluid_widths(self):
        """Длинные подписи переносят текст по фактической ширине, а не
        режутся: окно можно сузить до минимального размера."""
        for widget, owner, fraction, minimum in self._fluid_refs:
            room = owner.winfo_width()
            if room < 80:
                continue
            wrap = max(minimum, min(int(room * fraction), room - 24))
            widget.configure(wraplength=wrap)
        for scroll in self._scrolls:
            scroll.apply_size()

    # --- левая колонка ------------------------------------------------
    def _build_params(self, parent):
        box = self._box(parent, "box_language")
        row = ttk.Frame(box)
        row.pack(anchor="w")
        for code, key in (("ru", "lang_ru"), ("en", "lang_en")):
            ttk.Radiobutton(row, value=code, variable=self.var_lang,
                            command=self.switch_language).pack(side="left",
                                                                padx=(0, 10))
            rb = row.winfo_children()[-1]
            self._reg(rb, key, "text")

        box = self._box(parent, "box_input")
        line = ttk.Frame(box)
        line.pack(anchor="w", fill="x")
        self._label(line, "lbl_input_preset").pack(side="left")
        self.combo_input = ttk.Combobox(line, textvariable=self.var_input_combo,
                                        state="readonly", width=22)
        self.combo_input.pack(side="right")
        self.combo_input.bind("<<ComboboxSelected>>", self.on_input_preset)
        self.txt_matrix = self._make_text(box, height_rows=6)
        self.txt_matrix.pack(anchor="w", pady=(4, 0))
        self.txt_matrix.bind("<Control-Return>",
                             lambda event: self.on_apply_matrix())
        btn = ttk.Button(box, text=T("btn_apply_matrix"), style="Go.TButton",
                         command=self.on_apply_matrix)
        btn.pack(anchor="w", pady=(4, 2))
        self._reg(btn, "btn_apply_matrix")
        self._muted(box, "hint_matrix", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))
        self.lbl_size = self._label(box, style="Head.TLabel", text="")
        self.lbl_size.pack(anchor="w", pady=(2, 0))
        self.lbl_input_desc = self._label(box, style="Muted.TLabel", text="",
                                          wraplength=LEFT_COL - 46,
                                          justify="left")
        self.lbl_input_desc.pack(anchor="w")

        box = self._box(parent, "box_kernel")
        line = ttk.Frame(box)
        line.pack(anchor="w", fill="x")
        self._label(line, "lbl_kernel_preset").pack(side="left")
        self.combo_kernel = ttk.Combobox(line, textvariable=self.var_kernel_combo,
                                         state="readonly", width=22)
        self.combo_kernel.pack(side="right")
        self.combo_kernel.bind("<<ComboboxSelected>>", self.on_kernel_preset)
        grid = ttk.Frame(box)
        grid.pack(pady=(5, 1))
        self.kernel_entries = []
        for index in range(9):
            entry = ttk.Entry(grid, textvariable=self.kernel_vars[index],
                              width=5, style="GoodEntry.TEntry")
            entry.grid(row=index // 3, column=index % 3, padx=2, pady=2)
            entry.bind("<Return>", self.on_kernel_entries)
            entry.bind("<FocusOut>", self.on_kernel_entries)
            self._entry_refs.append(entry)
            self.kernel_entries.append(entry)
        self.lbl_kernel_desc = self._label(box, style="Muted.TLabel", text="",
                                           wraplength=LEFT_COL - 46,
                                           justify="left")
        self.lbl_kernel_desc.pack(anchor="w", pady=(1, 0))
        self._muted(box, "hint_kernel", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))
        self._muted(box, "kernel_no_flip", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))
        line = ttk.Frame(box)
        line.pack(anchor="w", pady=(4, 0))
        self._label(line, "lbl_bias").pack(side="left")
        self.ent_bias = ttk.Entry(line, textvariable=self.var_bias, width=7,
                                  style="GoodEntry.TEntry")
        self.ent_bias.pack(side="left", padx=(6, 0))
        self.ent_bias.bind("<Return>", self.on_bias)
        self.ent_bias.bind("<FocusOut>", self.on_bias)
        self._entry_refs.append(self.ent_bias)
        self._muted(box, "hint_bias", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))

        box = self._box(parent, "box_geometry")
        self._label(box, "lbl_stride").pack(anchor="w")
        self.var_stride_group = tk.IntVar(value=DEFAULT_STRIDE)
        self.stride_buttons = []
        for value in STRIDES:
            rb = ttk.Radiobutton(box, value=value,
                                 variable=self.var_stride_group,
                                 command=lambda v=value: self.on_stride(v))
            rb.pack(anchor="w")
            self.stride_buttons.append(rb)
            self._reg(rb, "stride_%d" % value, "text")
        self._muted(box, "hint_stride", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 4))
        self._label(box, "lbl_padding").pack(anchor="w")
        self.var_pad_group = tk.IntVar(value=DEFAULT_PAD)
        self.pad_buttons = []
        for value in PADS:
            rb = ttk.Radiobutton(box, value=value,
                                 variable=self.var_pad_group,
                                 command=lambda v=value: self.on_pad(v))
            rb.pack(anchor="w")
            self.pad_buttons.append(rb)
            self._reg(rb, "pad_%d" % value, "text")
        self._muted(box, "hint_padding", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(0, 0))

        box = self._box(parent, "box_activation")
        self.var_activation = tk.StringVar(value=DEFAULT_ACTIVATION)
        self.activation_buttons = []
        for value in ACTIVATIONS:
            rb = ttk.Radiobutton(box, value=value, variable=self.var_activation,
                                 command=self.on_activation)
            rb.pack(anchor="w")
            self.activation_buttons.append(rb)
            self._reg(rb, "act_%s" % value, "text")
        self._muted(box, "hint_activation", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))

        box = self._box(parent, "box_pool")
        self._label(box, "lbl_pool_mode").pack(anchor="w")
        self.var_pool_mode = tk.StringVar(value=DEFAULT_POOL_MODE)
        self.pool_mode_buttons = []
        for value in POOL_MODES:
            rb = ttk.Radiobutton(box, value=value, variable=self.var_pool_mode,
                                 command=self.on_pool_mode)
            rb.pack(anchor="w")
            self.pool_mode_buttons.append(rb)
            self._reg(rb, "pool_%s" % value, "text")
        self._label(box, "lbl_pool_window").pack(anchor="w", pady=(4, 0))
        self.var_pool_window = tk.IntVar(value=DEFAULT_POOL_WINDOW)
        self.pool_window_buttons = []
        for value in POOL_WINDOWS:
            rb = ttk.Radiobutton(box, value=value,
                                 variable=self.var_pool_window,
                                 command=lambda v=value: self.on_pool_window(v))
            rb.pack(anchor="w")
            self.pool_window_buttons.append(rb)
            self._reg(rb, "pool_w%d" % value, "text")
        self._muted(box, "hint_pool", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))
        self._muted(box, "hint_pool_window", wraplength=LEFT_COL - 46,
                    justify="left").pack(anchor="w", pady=(2, 0))

        btn = ttk.Button(parent, text=T("btn_reset_all"), style="Go.TButton",
                         command=self.reset_all)
        btn.pack(anchor="w", pady=(2, 6))
        self._reg(btn, "btn_reset_all")

    def _make_text(self, parent, width=None, height_rows=6):
        widget = tk.Text(parent, width=width or TEXT_WIDTH - 6,
                         height=height_rows, wrap="none", relief="flat",
                         background=COL_PAPER, foreground=COL_TEXT,
                         borderwidth=0, highlightthickness=1,
                         highlightbackground=COL_FRAME,
                         font=self.fonts["monos"], padx=6, pady=4)
        return widget

    # --- вкладка «Свёртка» --------------------------------------------
    def _build_tab_conv(self):
        page = ttk.Frame(self.nb, padding=(10, 8))
        self.nb.add(page, text=T("tab_conv"))
        self._tabs.append((page, "tab_conv"))
        # Цепочка из пяти колонок самая широкая в программе: на узком экране
        # она не должна обрезаться, поэтому у вкладки есть горизонтальная
        # прокрутка.
        tab = self._scrollable(page, horizontal=True)
        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=0)
        tab.columnconfigure(2, weight=0)
        tab.columnconfigure(3, weight=0)
        tab.columnconfigure(4, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab)
        left.grid(row=0, column=0, sticky="nw")
        self._caption(left, "cap_input").pack(anchor="w")
        self.input_canvas = GridCanvas(left, cell_size=CELL,
                                       on_click=self.on_input_click,
                                       fonts=self.fonts)
        self.input_canvas.pack(anchor="w", pady=(2, 0))
        legend = ttk.Frame(left)
        legend.pack(anchor="w", pady=(4, 0))
        self.input_legends = []
        for key in ("legend_window", "legend_visited", "legend_skipped"):
            item = self._muted(legend, key, justify="left")
            item.pack(anchor="w", pady=1)
            self.input_legends.append(item)

        ttk.Label(tab, text="→", style="Arrow.TLabel").grid(row=0, column=1,
                                                             sticky="ns")

        middle = ttk.Frame(tab)
        middle.grid(row=0, column=2, sticky="nw", padx=(4, 0))
        self._caption(middle, "cap_kernel").pack(anchor="w")
        self.kernel_canvas = GridCanvas(middle, cell_size=CELL,
                                        fonts=self.fonts)
        self.kernel_canvas.pack(anchor="w", pady=(2, 0))
        ttk.Label(middle, text="↓", style="Arrow.TLabel").pack(pady=(2, 0))
        self._caption(middle, "cap_products").pack(anchor="w", pady=(2, 0))
        self.products_canvas = GridCanvas(middle, cell_size=CELL + 6,
                                          fonts=self.fonts)
        self.products_canvas.pack(anchor="w", pady=(2, 0))

        ttk.Label(tab, text="→", style="Arrow.TLabel").grid(row=0, column=3,
                                                            sticky="ns")

        right = ttk.Frame(tab)
        right.grid(row=0, column=4, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        holder = ttk.Frame(right)
        holder.grid(row=0, column=0, sticky="nsew")
        self._caption(holder, "calc_title").pack(anchor="w")
        self.txt_calc = tk.Text(holder, width=TEXT_WIDTH, height=15,
                                wrap="word", relief="flat",
                                background=COL_PAPER, foreground=COL_TEXT,
                                borderwidth=0, highlightthickness=1,
                                highlightbackground=COL_FRAME,
                                font=self.fonts["small"], padx=8, pady=6,
                                state="disabled")
        self.txt_calc.pack(fill="both", expand=True, pady=(2, 0))
        self.txt_calc.tag_configure("h", font=self.fonts["smallb"],
                                    foreground=COL_HEAD)
        self.txt_calc.tag_configure("t", font=self.fonts["small"])
        self.txt_calc.tag_configure("k", font=self.fonts["monob"],
                                    foreground=COL_KEY)
        self.txt_calc.tag_configure("m", font=self.fonts["monos"],
                                    foreground="#3a3a35")
        self.txt_calc.tag_configure("b", font=self.fonts["smallb"],
                                    foreground=COL_BAD_FG)
        map_frame = ttk.Frame(right)
        map_frame.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._caption(map_frame, "cap_map").pack(anchor="w")
        self.map_canvas = GridCanvas(map_frame, cell_size=CELL,
                                     on_click=self.on_map_click,
                                     fonts=self.fonts)
        self.map_canvas.pack(anchor="w", pady=(2, 0))
        self.map_legends = []
        for key in ("legend_positive", "legend_negative"):
            item = self._muted(map_frame, key, justify="left")
            item.pack(anchor="w")
            self.map_legends.append(item)

        controls = ttk.Frame(tab)
        controls.grid(row=1, column=0, columnspan=5, sticky="we", pady=(8, 0))
        info = ttk.Frame(controls)
        info.pack(side="left", fill="x", expand=True)
        self._label(info, "box_outsize", style="Cap.TLabel").pack(anchor="w")
        self.lbl_outsize = self._label(info, style="Key.TLabel", text="")
        self.lbl_outsize.pack(anchor="w")
        self.lbl_outmap = self._label(info, style="Sub.TLabel", text="")
        self.lbl_outmap.pack(anchor="w")
        self.lbl_coords = self._label(info, style="Muted.TLabel", text="")
        self.lbl_coords.pack(anchor="w")

        box = ttk.LabelFrame(controls, text=T("box_step"), padding=(8, 4))
        box.pack(side="left", padx=(8, 4))
        self._reg(box, "box_step")
        row = ttk.Frame(box)
        row.pack(anchor="w")
        self.btn_prev = ttk.Button(row, text=T("btn_prev"),
                                   command=lambda: self.step_position(-1))
        self.btn_prev.pack(side="left")
        self._reg(self.btn_prev, "btn_prev")
        self.btn_next = ttk.Button(row, text=T("btn_next"),
                                   command=lambda: self.step_position(1))
        self.btn_next.pack(side="left", padx=(4, 0))
        self._reg(self.btn_next, "btn_next")
        btn = ttk.Button(row, text=T("btn_step_reset"),
                         command=self.reset_step)
        btn.pack(side="left", padx=(4, 0))
        self._reg(btn, "btn_step_reset")
        self._label(row, textvariable=self.var_position,
                    font=self.fonts["smallb"],
                    foreground=COL_HEAD).pack(side="left", padx=(10, 0))

        box = ttk.LabelFrame(controls, text=T("box_scan"), padding=(8, 4))
        box.pack(side="left", padx=4)
        self._reg(box, "box_scan")
        row = ttk.Frame(box)
        row.pack(anchor="w")
        self.btn_scan_start = ttk.Button(row, text=T("btn_scan_start"),
                                         command=self.start_scan)
        self.btn_scan_start.pack(side="left")
        self._reg(self.btn_scan_start, "btn_scan_start")
        self.btn_scan_pause = ttk.Button(row, text=T("btn_scan_pause"),
                                         command=self.pause_scan)
        self.btn_scan_pause.pack(side="left", padx=(4, 0))
        self._reg(self.btn_scan_pause, "btn_scan_pause")
        btn = ttk.Button(row, text=T("btn_scan_reset"), command=self.reset_scan)
        btn.pack(side="left", padx=(4, 0))
        self._reg(btn, "btn_scan_reset")
        line = ttk.Frame(box)
        line.pack(anchor="w", pady=(3, 0))
        self._label(line, "lbl_speed").pack(side="left")
        self.ent_speed = ttk.Entry(line, textvariable=self.var_speed, width=6,
                                   style="GoodEntry.TEntry")
        self.ent_speed.pack(side="left", padx=(5, 0))
        self.ent_speed.bind("<Return>", self.on_speed)
        self.ent_speed.bind("<FocusOut>", self.on_speed)
        self._entry_refs.append(self.ent_speed)

    # --- вкладка «Пулинг» -----------------------------------------------
    def _build_tab_pool(self):
        page = ttk.Frame(self.nb, padding=(10, 8))
        self.nb.add(page, text=T("tab_pool"))
        self._tabs.append((page, "tab_pool"))
        # Карта Y, карта Z и панель расчёта шире любой вкладки: прокрутка лучше
        # обрезка.
        tab = self._scrollable(page, horizontal=True)
        intro = self._label(tab, "pool_intro", style="Sub.TLabel",
                            wraplength=1000, justify="left")
        intro.pack(anchor="w")
        self._fluid(intro, self.nb)
        self.lbl_pool_summary = self._label(tab, style="Head.TLabel", text="")
        self.lbl_pool_summary.pack(anchor="w", pady=(4, 0))
        self.lbl_pool_outsize = self._label(tab, style="Key.TLabel", text="")
        self.lbl_pool_outsize.pack(anchor="w")

        board = ttk.Frame(tab)
        board.pack(anchor="nw", pady=(6, 0))
        board.columnconfigure(3, weight=1)
        board.rowconfigure(0, weight=1)

        source = ttk.Frame(board)
        source.grid(row=0, column=0, sticky="nw")
        self._caption(source, "cap_pool_src").pack(anchor="w")
        self.pool_src_canvas = GridCanvas(source, cell_size=32,
                                         number_font="monos",
                                         on_click=self.on_pool_source_click,
                                         fonts=self.fonts)
        self.pool_src_canvas.pack(anchor="w", pady=(2, 0))
        self.lbl_pool_unused = self._label(source, style="Head.TLabel", text="",
                                          justify="left")
        self.lbl_pool_unused.pack(anchor="w")
        self.pool_src_legends = []
        self.lbl_pool_star_legend = None
        self.lbl_pool_unused_legend = None
        for key in ("legend_pool_window", "legend_pool_star",
                    "legend_pool_unused"):
            item = self._muted(source, key, justify="left")
            item.pack(anchor="w", pady=1)
            self.pool_src_legends.append(item)
            if key == "legend_pool_star":
                self.lbl_pool_star_legend = item
            elif key == "legend_pool_unused":
                self.lbl_pool_unused_legend = item

        ttk.Label(board, text="→", style="Arrow.TLabel").grid(row=0, column=1,
                                                              sticky="ns",
                                                              padx=(6, 6))

        pooled = ttk.Frame(board)
        pooled.grid(row=0, column=2, sticky="nw")
        self._caption(pooled, "cap_pool_out").pack(anchor="w")
        self.pool_out_canvas = GridCanvas(pooled, cell_size=POOL_CELL,
                                         number_font="monos",
                                         on_click=self.on_pool_click,
                                         fonts=self.fonts)
        self.pool_out_canvas.pack(anchor="w", pady=(2, 0))
        self.pool_out_legends = []
        for key in ("legend_pool_selected",):
            item = self._muted(pooled, key, justify="left")
            item.pack(anchor="w", pady=1)
            self.pool_out_legends.append(item)

        right = ttk.Frame(board)
        right.grid(row=0, column=3, sticky="nsew", padx=(8, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._caption(right, "pool_calc_title").pack(anchor="w")
        self.txt_pool = tk.Text(right, width=TEXT_WIDTH, height=15,
                                wrap="word", relief="flat",
                                background=COL_PAPER, foreground=COL_TEXT,
                                borderwidth=0, highlightthickness=1,
                                highlightbackground=COL_FRAME,
                                font=self.fonts["small"], padx=8, pady=6,
                                state="disabled")
        self.txt_pool.pack(fill="both", expand=True, pady=(2, 0))
        self.txt_pool.tag_configure("h", font=self.fonts["smallb"],
                                    foreground=COL_HEAD)
        self.txt_pool.tag_configure("t", font=self.fonts["small"])
        self.txt_pool.tag_configure("k", font=self.fonts["monob"],
                                    foreground=COL_KEY)
        self.txt_pool.tag_configure("m", font=self.fonts["monos"],
                                    foreground="#3a3a35")
        self.txt_pool.tag_configure("b", font=self.fonts["smallb"],
                                    foreground=COL_BAD_FG)

    # --- вкладка сравнения --------------------------------------------
    def _build_tab_compare(self):
        page = ttk.Frame(self.nb, padding=(10, 8))
        self.nb.add(page, text=T("tab_compare"))
        self._tabs.append((page, "tab_compare"))
        # Пять плиток шире любой вкладки: прокрутка лучше обрезка.
        tab = self._scrollable(page, horizontal=True)
        intro = self._label(tab, "cmp_intro", style="Sub.TLabel",
                            wraplength=1000, justify="left")
        intro.pack(anchor="w")
        self._fluid(intro, self.nb)
        self.lbl_cmp_input = self._label(tab, style="Head.TLabel", text="")
        self.lbl_cmp_input.pack(anchor="w", pady=(4, 0))

        strip = ttk.Frame(tab)
        strip.pack(anchor="w", pady=(6, 0))
        self.cmp_tiles = []
        for key in KERNEL_COMPARISON_KEYS:
            tile = ttk.Frame(strip, padding=(7, 4))
            tile.pack(side="left", anchor="nw")
            title = self._label(tile, style="Cap.TLabel", text="")
            title.pack(anchor="w")
            size = self._label(tile, style="Muted.TLabel", text="")
            size.pack(anchor="w")
            kernel_view = GridCanvas(tile, cell_size=20, fonts=self.fonts,
                                     number_font="monoxs")
            kernel_view.pack(anchor="w", pady=(3, 0))
            map_view = GridCanvas(tile, cell_size=24, fonts=self.fonts,
                                  number_font="monoxs")
            map_view.pack(anchor="w", pady=(3, 2))
            map_view.on_click = (
                lambda row, column, kernel_key=key:
                self.on_cmp_click(kernel_key, row, column))
            desc = self._label(tile, style="Muted.TLabel", text="",
                               wraplength=190, justify="left")
            desc.pack(anchor="w")
            self.cmp_tiles.append({"key": key, "title": title, "size": size,
                                   "kernel": kernel_view, "map": map_view,
                                   "desc": desc})
        note = self._muted(tab, "cmp_note", wraplength=1000, justify="left")
        note.pack(anchor="w", pady=(6, 0))
        self._fluid(note, self.nb)

    # --- вкладка экспериментов ----------------------------------------
    def _build_tab_experiments(self):
        page = ttk.Frame(self.nb, padding=(10, 8))
        self.nb.add(page, text=T("tab_experiments"))
        self._tabs.append((page, "tab_experiments"))
        tab = self._scrollable(page, horizontal=True)
        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        intro = self._label(tab, "exp_intro", style="Sub.TLabel",
                            wraplength=760, justify="left")
        intro.grid(row=0, column=1, sticky="w")
        self._fluid(intro, self.nb, fraction=0.68)

        holder = ttk.Frame(tab)
        holder.grid(row=1, column=0, rowspan=2, sticky="ns", padx=(0, 8))
        self.exp_list = tk.Listbox(holder, width=30, height=9,
                                   activestyle="dotbox", relief="flat",
                                   background=COL_PAPER, foreground=COL_TEXT,
                                   highlightthickness=1,
                                   highlightbackground=COL_FRAME,
                                   font=self.fonts["small"], exportselection=0)
        self.exp_list.pack(fill="y")
        self.exp_list.bind("<<ListboxSelect>>", self.on_experiment_select)

        body = ttk.Frame(tab)
        body.grid(row=1, column=1, sticky="nsew")
        body.columnconfigure(0, weight=1)
        self.txt_exp = tk.Text(body, width=TEXT_WIDTH + 8, height=7,
                               wrap="word", relief="flat",
                               background=COL_PAPER, foreground=COL_TEXT,
                               borderwidth=0, highlightthickness=1,
                               highlightbackground=COL_FRAME,
                               font=self.fonts["small"], padx=8, pady=6,
                               state="disabled")
        self.txt_exp.grid(row=0, column=0, sticky="nsew")
        self.txt_exp.tag_configure("h", font=self.fonts["smallb"],
                                   foreground=COL_HEAD)
        self.txt_exp.tag_configure("t", font=self.fonts["small"])
        self.lbl_exp_result = self._label(body, style="Key.TLabel", text="")
        self.lbl_exp_result.grid(row=1, column=0, sticky="w", pady=(4, 0))
        btn = ttk.Button(body, text=T("exp_apply"), style="Go.TButton",
                         command=self.on_experiment_apply)
        btn.grid(row=2, column=0, sticky="w", pady=(4, 6))
        self._reg(btn, "exp_apply")

        preview = ttk.Frame(body)
        preview.grid(row=3, column=0, sticky="w")
        self._caption(preview, "exp_preview_title").pack(anchor="w")
        strip = ttk.Frame(preview)
        strip.pack(anchor="w", pady=(2, 0))
        self.exp_tiles = []
        for index in range(2):
            tile = ttk.Frame(strip, padding=(7, 3))
            title = self._label(tile, style="Cap.TLabel", text="")
            title.pack(anchor="w")
            view = GridCanvas(tile, cell_size=24, fonts=self.fonts,
                              number_font="monoxs")
            view.pack(anchor="w", pady=(2, 0))
            self.exp_tiles.append({"title": title, "view": view, "frame": tile})

    # --- вкладка формул -----------------------------------------------
    def _build_tab_formula(self):
        page = ttk.Frame(self.nb, padding=(10, 8))
        self.nb.add(page, text=T("tab_formula"))
        self._tabs.append((page, "tab_formula"))
        tab = self._scrollable(page)
        self.txt_formula = tk.Text(tab, width=TEXT_WIDTH + 26, height=26,
                                   wrap="word", relief="flat",
                                   background=COL_PAPER, foreground=COL_TEXT,
                                   borderwidth=0, highlightthickness=1,
                                   highlightbackground=COL_FRAME,
                                   font=self.fonts["small"], padx=10, pady=8,
                                   state="disabled")
        self.txt_formula.pack(fill="both", expand=True)
        self.txt_formula.tag_configure("h", font=self.fonts["smallb"],
                                       foreground=COL_HEAD)
        self.txt_formula.tag_configure("t", font=self.fonts["small"])
        self.txt_formula.tag_configure("k", font=self.fonts["monob"],
                                       foreground=COL_KEY)
        self.txt_formula.tag_configure("m", font=self.fonts["mono"],
                                       foreground=COL_KEY)

    # --- вкладка справки ----------------------------------------------
    def _build_tab_help(self):
        page = ttk.Frame(self.nb, padding=(10, 8))
        self.nb.add(page, text=T("tab_help"))
        self._tabs.append((page, "tab_help"))
        tab = self._scrollable(page)
        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        holder = ttk.Frame(tab)
        holder.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.help_list = tk.Listbox(holder, width=28, height=10,
                                    activestyle="dotbox", relief="flat",
                                    background=COL_PAPER, foreground=COL_TEXT,
                                    highlightthickness=1,
                                    highlightbackground=COL_FRAME,
                                    font=self.fonts["small"], exportselection=0)
        self.help_list.pack(fill="y")
        self.help_list.bind("<<ListboxSelect>>", self.on_help_select)
        nav = ttk.Frame(holder)
        nav.pack(anchor="w", pady=(5, 0))
        prev = ttk.Button(nav, text="<", width=3,
                          command=lambda: self.help_move(-1))
        prev.pack(side="left")
        next_button = ttk.Button(nav, text=">", width=3,
                                 command=lambda: self.help_move(1))
        next_button.pack(side="left", padx=(4, 0))
        self.txt_help = tk.Text(tab, width=TEXT_WIDTH + 26, height=22,
                                wrap="word", relief="flat",
                                background=COL_PAPER, foreground=COL_TEXT,
                                borderwidth=0, highlightthickness=1,
                                highlightbackground=COL_FRAME,
                                font=self.fonts["small"], padx=10, pady=8,
                                state="disabled")
        self.txt_help.grid(row=0, column=1, sticky="nsew")
        self.txt_help.tag_configure("h", font=self.fonts["smallb"],
                                    foreground=COL_HEAD)
        self.txt_help.tag_configure("t", font=self.fonts["small"])

    # --- пересчёт ------------------------------------------------------
    def recompute(self):
        self.result = convolve(self.values, self.kernel, self.stride, self.pad,
                               self.bias, self.activation)
        self.clamp_selection()
        self.pool_result = pool_map(self.pooling_source(), self.pool_window,
                                    self.pool_mode)
        self.clamp_pool_selection()

    def out_rows(self):
        return self.result["out_rows"] if self.result else 0

    def out_columns(self):
        return self.result["out_columns"] if self.result else 0

    def total_positions(self):
        return max(0, self.result["total"]) if self.result else 0

    def pooling_source(self):
        """Карта признаков после активации: её и берёт пулинг."""
        return self.result["values"] if self.result else []

    def pool_rows(self):
        return self.pool_result["out_rows"] if self.pool_result else 0

    def pool_columns(self):
        return self.pool_result["out_columns"] if self.pool_result else 0

    def pool_total(self):
        return max(0, self.pool_result["total"]) if self.pool_result else 0

    def clamp_pool_selection(self):
        """Выбранная клетка пулинга обязана существовать при любой карте."""
        rows, columns = self.pool_rows(), self.pool_columns()
        if rows <= 0 or columns <= 0:
            self.pool_step = None
            return
        self.pool_i = max(0, min(self.pool_i, rows - 1))
        self.pool_j = max(0, min(self.pool_j, columns - 1))
        self.pool_step = pool_cell(self.pooling_source(), self.pool_window,
                                   self.pool_mode, self.pool_i, self.pool_j)

    def clamp_selection(self):
        """Выбранная клетка обязана существовать при любой конфигурации."""
        rows, columns = self.out_rows(), self.out_columns()
        if rows <= 0 or columns <= 0:
            self.cell = None
            return
        self.out_i = max(0, min(self.out_i, rows - 1))
        self.out_j = max(0, min(self.out_j, columns - 1))
        self.cell = cell_at(self.values, self.kernel, self.stride, self.pad,
                            self.bias, self.activation, self.out_i, self.out_j)

    def current_index(self):
        return self.cell.index + 1 if self.cell else 0

    def refresh(self, status_key=None, **kwargs):
        """Пересчёт и полная перерисовка. Обход при этом не останавливается:
        шаги и обход идут через refresh, а останавливает обход пользователь."""
        self.recompute()
        self.draw_all()
        if status_key:
            self.set_status(status_key, **kwargs)

    def draw_all(self):
        self.draw_labels()
        self.draw_input()
        self.draw_kernel()
        self.draw_products()
        self.draw_map()
        self.draw_calc()
        self.draw_pool()
        self.draw_geometry()
        self.draw_step_buttons()
        self.draw_compare()
        self.draw_formula()
        for scroll in self._scrolls:
            scroll.apply_size()

    def draw_labels(self):
        """Подписи, которые зависят и от состояния, и от языка."""
        rows, columns = len(self.values), len(self.values[0])
        self.lbl_size.configure(
            text=T("matrix_size", rows=rows, cols=columns))
        self.lbl_input_desc.configure(
            text=T("input_%s_desc" % self.input_key, lo=MIN_DIM, hi=MAX_DIM,
                   lim=fmt(VALUE_LIMIT)))
        self.lbl_kernel_desc.configure(text=T("kernel_%s_desc" % self.kernel_key))

    # --- отрисовка четырёх матриц -------------------------------------
    def draw_input(self):
        padded = padded_matrix(self.values, self.pad)
        rows = len(padded)
        columns = len(padded[0])
        low, high = min_max(self.values)
        visited, skipped = skipped_origins(self.values, KERNEL_SIZE, self.pad,
                                           self.stride)
        spec = []
        for i in range(rows):
            row = []
            for j in range(columns):
                border = self.pad and (i < self.pad or i >= rows - self.pad or
                                       j < self.pad or j >= columns - self.pad)
                value = padded[i][j]
                inside = False
                if self.cell:
                    inside = (self.cell.win_top <= i <
                              self.cell.win_top + KERNEL_SIZE and
                              self.cell.win_left <= j <
                              self.cell.win_left + KERNEL_SIZE)
                background = COL_PAD if border else intensity_color(value, low,
                                                                   high)
                mark = ""
                if self.stride > 1:
                    if (i, j) in visited:
                        mark = "•"
                    elif (i, j) in skipped:
                        mark = "✕"
                row.append(cell_spec(
                    text=fmt(value),
                    bg=background,
                    edge=COL_WIN if inside else None,
                    edge_width=3 if inside else 1,
                    note="P" if border else "",
                    mark=mark,
                ))
            spec.append(row)
        labels = [str(index) for index in range(columns)]
        self.input_canvas.axes = ([str(index) for index in range(rows)], labels)
        self.input_canvas.render(spec)
        self._wrap_legends(self.input_legends, self.input_canvas)

    def draw_kernel(self):
        scale = abs_scale(self.kernel)
        spec = []
        for row in self.kernel:
            spec.append([cell_spec(text=fmt(value),
                                   bg=diverging_color(value, scale))
                         for value in row])
        self.kernel_canvas.render(spec)

    def draw_products(self):
        spec = []
        if self.cell:
            for i, product_row in enumerate(self.cell.products):
                line = []
                for j, product in enumerate(product_row):
                    input_value, weight, _ = self.cell.terms[
                        i * KERNEL_SIZE + j]
                    line.append(cell_spec(
                        text=fmt(product),
                        bg=diverging_color(product,
                                           abs_scale(
                                               [row for row in
                                                self.cell.products])),
                        note=term_text(input_value, weight),
                    ))
                spec.append(line)
        self.products_canvas.render(spec or [[cell_spec(text="—",
                                                        bg=COL_PAPER)
                                              for _ in range(3)]
                                             for _ in range(3)])

    def draw_map(self):
        spec = []
        if self.total_positions():
            values = self.result["values"]
            scale = abs_scale(values)
            for i, row in enumerate(values):
                line = []
                for j, value in enumerate(row):
                    current = bool(self.cell) and self.cell.out_i == i
                    current = current and self.cell.out_j == j
                    line.append(cell_spec(
                        text=fmt(value),
                        bg=diverging_color(value, scale),
                        edge=COL_WIN if current else None,
                        edge_width=3 if current else 1,
                        mark=str(i * len(values[0]) + j + 1),
                        bold=current,
                    ))
                spec.append(line)
        labels = [str(index) for index in
                  range(self.out_columns() or 1)]
        self.map_canvas.axes = ([str(index) for index in
                                 range(self.out_rows() or 1)], labels)
        self.map_canvas.render(spec)
        self._wrap_legends(self.map_legends, self.map_canvas)

    def _wrap_legends(self, legends, canvas):
        """Подсказки под сеткой переносятся по ширине самой сетки."""
        room = max(150, canvas.winfo_reqwidth() - 6)
        for widget in legends:
            widget.configure(wraplength=room)

    # --- пулинг: две карты и расчёт ------------------------------------
    def draw_pool(self):
        """Карта Y с выбранным окном, карта Z и все числа одного окна."""
        source = self.pooling_source()
        result = self.pool_result or pool_map(source, self.pool_window,
                                              self.pool_mode)
        window = self.pool_window
        step = self.pool_step
        rows = len(source)
        columns = len(source[0]) if rows else 0
        used_rows = result["out_rows"] * window if result["total"] else 0
        used_columns = result["out_columns"] * window if result["total"] else 0
        scale = abs_scale(source)
        spec = []
        for i in range(rows):
            line = []
            for j in range(columns):
                value = source[i][j]
                inside = bool(step) and (step.top <= i < step.top + window and
                                         step.left <= j < step.left + window)
                unused = (i >= used_rows or j >= used_columns)
                ties = bool(step) and step.mode == "max" and (i, j) in step.ties
                line.append(cell_spec(
                    text=fmt(value),
                    bg=COL_PAD if unused else diverging_color(value, scale),
                    edge=COL_WIN if inside else None,
                    edge_width=3 if inside else 1,
                    mark="★" if ties else "",
                    bold=inside and ties and (i, j) == step.winner,
                ))
            spec.append(line)
        self.pool_src_canvas.axes = ([str(index) for index in range(rows)],
                                     [str(index) for index in range(columns)])
        self.pool_src_canvas.render(spec)
        # ★ рисуется только в max: в average подсказка про максимум была бы
        # описанием знака, которого на экране нет.
        if self.pool_mode == "max":
            if not self.lbl_pool_star_legend.winfo_ismapped():
                self.lbl_pool_star_legend.pack(
                    anchor="w", pady=1,
                    before=self.lbl_pool_unused_legend)
        else:
            self.lbl_pool_star_legend.pack_forget()
        self._wrap_legends(self.pool_src_legends, self.pool_src_canvas)

        pooled = result["values"]
        pooled_scale = abs_scale(pooled)
        pspec = []
        for i, row in enumerate(pooled):
            line = []
            for j, value in enumerate(row):
                current = bool(step) and step.out_i == i and step.out_j == j
                line.append(cell_spec(
                    text=fmt(value, POOL_DIGITS),
                    bg=diverging_color(value, pooled_scale),
                    edge=COL_WIN if current else None,
                    edge_width=3 if current else 1,
                    mark=str(i * len(pooled[0]) + j + 1),
                    bold=current,
                ))
            pspec.append(line)
        self.pool_out_canvas.axes = (
            [str(index) for index in range(result["out_rows"] or 1)],
            [str(index) for index in range(result["out_columns"] or 1)])
        self.pool_out_canvas.render(pspec)
        self._wrap_legends(self.pool_out_legends, self.pool_out_canvas)

        if result["total"]:
            self.lbl_pool_summary.configure(
                text=T("pool_summary", rows=rows, cols=columns, w=window,
                       prows=result["out_rows"],
                       pcols=result["out_columns"],
                       ptotal=result["total"]))
        else:
            self.lbl_pool_summary.configure(
                text=T("pool_no_map", rows=rows, cols=columns, w=window))
        if rows == columns:
            self.lbl_pool_outsize.configure(
                text=T("pool_outsize_square", n=rows, w=window,
                       o=result["out_rows"]))
        else:
            self.lbl_pool_outsize.configure(
                text=T("pool_outsize_rows", n=rows, w=window,
                       o=result["out_rows"]) + "   ·   " +
                T("pool_outsize_cols", n=columns, w=window,
                  o=result["out_columns"]))
        if result["unused_cells"] > 0:
            self.lbl_pool_unused.configure(
                text=T("pool_unused", rows=result["unused_rows"],
                       cols=result["unused_columns"],
                       cells=result["unused_cells"], w=window))
        else:
            self.lbl_pool_unused.configure(
                text=T("pool_all_used", rows=rows, cols=columns, w=window))
        self.lbl_pool_unused.configure(
            wraplength=max(150, self.pool_src_canvas.winfo_reqwidth() - 6))
        self.draw_pool_calc()

    def draw_pool_calc(self):
        widget = self.txt_pool
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if not self.pool_step:
            widget.insert("end",
                          T("pool_map_missing", w=self.pool_window) + "\n", "b")
            widget.configure(state="disabled")
            return
        step = self.pool_step
        widget.insert("end", T("position_label", index=step.index + 1,
                               total=self.pool_total()) + "\n", "h")
        widget.insert("end",
                      T("pool_window_coords", w=step.window, r0=step.top,
                        r1=step.top + step.window - 1, c0=step.left,
                        c1=step.left + step.window - 1) + "\n", "t")
        widget.insert("end", T("pool_values_head") + "\n", "h")
        for values_row in step.region:
            widget.insert("end", "   " + "   ".join(
                fmt(value, POOL_DIGITS) for value in values_row) + "\n", "m")
        i, j = step.out_i, step.out_j
        if step.mode == "max":
            widget.insert("end", T("pool_max_head") + "\n", "h")
            widget.insert("end", "   " + T("pool_max_line", i=i, j=j,
                                           values=pool_values_text(
                                               step.flat, POOL_DIGITS),
                                           out=fmt(step.output,
                                                   POOL_DIGITS)) + "\n", "k")
            if len(step.ties) > 1:
                widget.insert("end", "   " + T(
                    "pool_tie_line", n=len(step.ties),
                    cells=", ".join("Y(%d,%d)" % (a, b) for a, b in step.ties),
                    r=step.winner[0], c=step.winner[1]) + "\n", "t")
            else:
                widget.insert("end", "   " + T("pool_winner_line",
                                               best=fmt(step.best, POOL_DIGITS),
                                               r=step.winner[0],
                                               c=step.winner[1]) + "\n", "t")
        else:
            widget.insert("end", T("pool_avg_head") + "\n", "h")
            widget.insert("end", "   " + T("pool_avg_line", i=i, j=j,
                                           expr=pool_sum_text(step.flat,
                                                              POOL_DIGITS),
                                           count=step.count,
                                           out=fmt(step.output,
                                                   POOL_DIGITS)) + "\n", "k")
            widget.insert("end", "   " + T("pool_avg_note", d=POOL_DIGITS,
                                           w=step.window) + "\n", "t")
        widget.insert("end", T("pool_cell_head") + "\n", "h")
        widget.insert("end", "   " + T("pool_cell_line", i=i, j=j,
                                       out=fmt(step.output,
                                               POOL_DIGITS)) + "\n", "k")
        widget.configure(state="disabled")

    # --- панель расчёта ------------------------------------------------
    def draw_calc(self):
        widget = self.txt_calc
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if not self.cell:
            widget.insert("end", T("calc_map_missing") + "\n", "b")
            widget.configure(state="disabled")
            return
        step = self.cell
        rows, columns = len(self.values), len(self.values[0])
        widget.insert("end",
                      T("position_label", index=step.index + 1,
                        total=self.total_positions()) + "\n", "h")
        if self.pad:
            note = T("window_coords_pad", r0=step.win_top,
                     r1=step.win_top + KERNEL_SIZE - 1, c0=step.win_left,
                     c1=step.win_left + KERNEL_SIZE - 1)
        else:
            note = T("window_coords", r0=step.src_top,
                     r1=step.src_top + KERNEL_SIZE - 1,
                     c0=step.src_left, c1=step.src_left + KERNEL_SIZE - 1)
        widget.insert("end", note + "\n", "t")
        widget.insert("end", T("calc_products") + "\n", "h")
        for i, product_row in enumerate(step.products):
            pieces = []
            for j in range(KERNEL_SIZE):
                input_value, weight, _ = step.terms[i * KERNEL_SIZE + j]
                pieces.append(term_text(input_value, weight))
            widget.insert("end", "   " + "   ".join(pieces) + "   = " +
                          fmt(sum(product_row)) + "\n", "m")
        widget.insert("end", T("calc_sum_head") + "\n", "h")
        widget.insert("end", "   " + T("calc_sum_line",
                                       expr=sum_text(step.terms),
                                       total=fmt(step.sum_products)) + "\n", "k")
        widget.insert("end", T("calc_bias_head") + "\n", "h")
        widget.insert("end", "   " + T("calc_bias_line",
                                       total=fmt(step.sum_products),
                                       bias=fmt(step.bias),
                                       pre=fmt(step.pre_activation)) + "\n", "k")
        widget.insert("end",
                      T("calc_relu_head" if self.activation == "relu"
                        else "calc_none_head") + "\n", "h")
        if self.activation == "relu":
            line = T("calc_relu_line", i=step.out_i, j=step.out_j,
                     relu=relu_text(step.pre_activation),
                     out=fmt(step.output))
        else:
            line = T("calc_none_line", i=step.out_i, j=step.out_j,
                     out=fmt(step.output))
        widget.insert("end", "   " + line + "\n", "k")
        widget.insert("end", T("calc_cell_head") + "\n", "h")
        widget.insert("end", "   " + T("calc_cell_line", i=step.out_i,
                                       j=step.out_j,
                                       out=fmt(step.output)) + "\n", "k")
        widget.insert("end", T("kernel_no_flip") + "\n", "t")
        widget.configure(state="disabled")

    # --- размер карты ---------------------------------------------------
    def draw_geometry(self):
        k = KERNEL_SIZE
        rows, columns = len(self.values), len(self.values[0])
        out_rows, out_columns = self.out_rows(), self.out_columns()
        if rows == columns:
            self.lbl_outsize.configure(
                text=T("outsize_square", n=rows, p=self.pad, k=k,
                       s=self.stride, o=out_rows))
        else:
            self.lbl_outsize.configure(
                text=T("outsize_rows", n=rows, p=self.pad, k=k, s=self.stride,
                       o=out_rows) + "   ·   " +
                T("outsize_cols", n=columns, p=self.pad, k=k, s=self.stride,
                  o=out_columns))
        if out_rows > 0 and out_columns > 0:
            self.lbl_outmap.configure(
                text=T("outsize_map", rows=out_rows, cols=out_columns,
                       total=self.total_positions()))
        else:
            self.lbl_outmap.configure(
                text=T("outsize_impossible", s=self.stride, p=self.pad, k=k,
                       rows=rows, cols=columns))
        if self.cell:
            if self.pad:
                self.lbl_coords.configure(
                    text=T("window_coords_pad", r0=self.cell.win_top,
                           r1=self.cell.win_top + KERNEL_SIZE - 1,
                           c0=self.cell.win_left,
                           c1=self.cell.win_left + KERNEL_SIZE - 1))
            else:
                self.lbl_coords.configure(
                    text=T("window_coords", r0=self.cell.src_top,
                           r1=self.cell.src_top + KERNEL_SIZE - 1,
                           c0=self.cell.src_left,
                           c1=self.cell.src_left + KERNEL_SIZE - 1))
        else:
            self.lbl_coords.configure(text=T("calc_map_missing"))

    def draw_step_buttons(self):
        total = self.total_positions()
        index = self.current_index()
        self.var_position.set(T("position_label", index=index, total=total)
                              if total else "")
        self.btn_prev.configure(state="normal" if total and index > 1
                                else "disabled")
        self.btn_next.configure(state="normal" if total and index < total
                                else "disabled")

    # --- вкладка сравнения ---------------------------------------------
    def draw_compare(self):
        values = self.values
        self.lbl_cmp_input.configure(
            text=T("cmp_input_line", name=self.input_name(),
                   rows=len(values), cols=len(values[0]), s=self.stride,
                   p=self.pad, b=fmt(self.bias),
                   a=T("act_short_%s" % self.activation)))
        maps = {}
        for tile in self.cmp_tiles:
            kernel = [list(map(float, row))
                      for row in KERNEL_BY_KEY[tile["key"]]["values"]]
            maps[tile["key"]] = convolve(values, kernel, self.stride, self.pad,
                                         self.bias, self.activation)
        widest = max([max(entry["out_rows"], entry["out_columns"])
                      for entry in maps.values()] + [3])
        size = max(14, min(24, 620 // widest))
        for tile in self.cmp_tiles:
            result = maps[tile["key"]]
            key = tile["key"]
            kernel = KERNEL_BY_KEY[key]["values"]
            tile["title"].configure(text=T("cmp_map_title",
                                           name=T("kernel_%s_name" % key)))
            tile["size"].configure(
                text=T("cmp_map_size", rows=result["out_rows"],
                       cols=result["out_columns"]) + " · " +
                T("cmp_minmax",
                  lo=fmt(min([c for row in result["values"] for c in row] or [0])),
                  hi=fmt(max([c for row in result["values"] for c in row] or [0]))))
            scale = abs_scale(kernel)
            tile["kernel"].cs = max(12, min(20, size // 2 + 6))
            tile["kernel"].render([[cell_spec(text=fmt(value),
                                              bg=diverging_color(value, scale))
                                    for value in row] for row in kernel])
            tile["map"].cs = size
            color_scale = abs_scale(result["values"])
            tile["map"].render([[cell_spec(
                text=fmt(value), bg=diverging_color(value, color_scale),
                edge=COL_WIN if (self.kernel_key == key and self.cell and
                                 self.cell.out_i == i and self.cell.out_j == j)
                else None,
                edge_width=2) for j, value in enumerate(row)]
                for i, row in enumerate(result["values"])])
            tile["desc"].configure(text=T("kernel_%s_desc" % key))

    def on_cmp_click(self, kernel_key, row, column):
        self.kernel_key = kernel_key
        self.kernel = [list(map(float, r))
                       for r in KERNEL_BY_KEY[kernel_key]["values"]]
        self.out_i, self.out_j = row, column
        self._sync_entries()
        self._fill_combos()
        self.nb.select(0)
        self.refresh("status_kernel_preset", name=self.kernel_name())

    # --- формулы --------------------------------------------------------
    def draw_formula(self):
        widget = self.txt_formula
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", T("formula_core_head") + "\n", "h")
        widget.insert("end", "   " + T("formula_core") + "\n", "k")
        widget.insert("end", T("formula_core_note") + "\n\n", "t")
        widget.insert("end", T("formula_symbols_head") + "\n", "h")
        for key in ("symbol_x", "symbol_k", "symbol_ij", "symbol_mn",
                    "symbol_sum", "symbol_b", "symbol_z", "symbol_y",
                    "symbol_s", "symbol_p"):
            widget.insert("end", "   • " + T(key, k=KERNEL_SIZE) + "\n", "t")
        widget.insert("end", "\n", "t")
        widget.insert("end", T("formula_out_head") + "\n", "h")
        widget.insert("end", "   " + T("formula_out") + "\n", "k")
        widget.insert("end", T("formula_out_note") + "\n\n", "t")
        widget.insert("end", T("formula_current_head") + "\n", "h")
        rows, columns = len(self.values), len(self.values[0])
        out_rows, out_columns = self.out_rows(), self.out_columns()
        widget.insert("end", "   X: %d×%d · K: %d×%d · S = %d · P = %d · "
                             "b = %s\n" % (rows, columns, KERNEL_SIZE,
                                           KERNEL_SIZE, self.stride, self.pad,
                                           fmt(self.bias)), "m")
        widget.insert("end", "   " + T("outsize_formula") + "\n", "t")
        widget.insert("end", "   " + T("outsize_rows", n=rows, p=self.pad,
                                       k=KERNEL_SIZE, s=self.stride,
                                       o=out_rows) + "\n", "m")
        widget.insert("end", "   " + T("outsize_cols", n=columns, p=self.pad,
                                       k=KERNEL_SIZE, s=self.stride,
                                       o=out_columns) + "\n", "m")
        if out_rows > 0 and out_columns > 0:
            widget.insert("end", "   " + T("outsize_map", rows=out_rows,
                                           cols=out_columns,
                                           total=self.total_positions()) + "\n",
                          "t")
        else:
            widget.insert("end", "   " + T("outsize_impossible", s=self.stride,
                                           p=self.pad, k=KERNEL_SIZE, rows=rows,
                                           cols=columns) + "\n", "b")
        widget.insert("end", "\n", "t")
        widget.insert("end", T("calc_relu_head") + "\n", "h")
        widget.insert("end", "   " + T("formula_relu") + "\n", "k")
        if self.cell:
            widget.insert("end", "   Y(%d,%d) = ReLU(z) = %s = %s\n" % (
                self.cell.out_i, self.cell.out_j,
                relu_text(self.cell.pre_activation),
                fmt(self.cell.output)), "m")
        widget.insert("end", T("formula_relu_note") + "\n", "t")
        widget.configure(state="disabled")

    # --- служебное -----------------------------------------------------
    def input_name(self, key=None):
        return T("input_%s_name" % (key or self.input_key))

    def kernel_name(self, key=None):
        return T("kernel_%s_name" % (key or self.kernel_key))

    def set_status(self, key, **kwargs):
        self.var_status.set(T(key, **kwargs))

    def flag_bad(self, widget, key, **kwargs):
        message = T(key, **kwargs)
        if widget is not None:
            try:
                widget.configure(style="BadEntry.TEntry")
            except tk.TclError:
                pass
            if widget not in self.bad_widgets:
                self.bad_widgets.append(widget)
        self.var_status.set(message + " " + T("bad_hint"))

    def clear_bad(self, widget=None):
        for entry in list(self.bad_widgets):
            try:
                entry.configure(style="GoodEntry.TEntry")
            except tk.TclError:
                pass
            self.bad_widgets.remove(entry)

    def _sync_entries(self):
        for index, var in enumerate(self.kernel_vars):
            var.set(fmt(self.kernel[index // 3][index % 3]))
        self.var_bias.set(fmt(self.bias))
        self.var_speed.set(str(self.scan_speed))
        self.txt_matrix.delete("1.0", "end")
        self.txt_matrix.insert("1.0", matrix_to_text(self.values))

    def _fill_combos(self):
        names = [T("input_%s_name" % entry["key"])
                 for entry in INPUT_PRESETS]
        self.combo_input.configure(values=names)
        index = [entry["key"] for entry in INPUT_PRESETS].index(self.input_key)
        self.combo_input.current(index)
        names = [T("kernel_%s_name" % entry["key"])
                 for entry in KERNEL_PRESETS]
        self.combo_kernel.configure(values=names)
        index = [entry["key"] for entry in KERNEL_PRESETS].index(
            self.kernel_key)
        self.combo_kernel.current(index)

    def _fill_experiment_list(self):
        self.exp_list.delete(0, "end")
        for entry in EXPERIMENTS:
            self.exp_list.insert("end", T("exp_%s_name" % entry["key"]))
        keys = [entry["key"] for entry in EXPERIMENTS]
        self.exp_list.selection_clear(0, "end")
        self.exp_list.selection_set(keys.index(self.experiment_key))
        self.exp_list.activate(keys.index(self.experiment_key))

    def _fill_help_list(self):
        self.help_list.delete(0, "end")
        for key in HELP_KEYS:
            self.help_list.insert("end", T("help_%s_title" % key))
        self.help_list.selection_clear(0, "end")
        self.help_list.selection_set(HELP_KEYS.index(self.help_key))
        self.help_list.activate(HELP_KEYS.index(self.help_key))

    # --- обработчики ---------------------------------------------------
    def on_input_preset(self, event=None):
        self.stop_scan()
        index = self.combo_input.current()
        key = INPUT_PRESETS[index]["key"]
        self.input_key = key
        if key != "custom":
            self.values = build_input(key)
        self.out_i = self.out_j = 0
        self._sync_entries()
        self.refresh("status_input_preset", name=self.input_name(key))

    def first_bad_token(self, text, mode="number"):
        """Первое подозрительное слово: не число или число вне диапазона."""
        for line in str(text).replace("\r", "\n").split("\n"):
            for chunk in line.replace(";", " ").replace(",", " ").split():
                number = parse_number(chunk)
                if mode == "number" and number is None:
                    return chunk
                if mode == "limit" and number is not None:
                    if abs(number) > VALUE_LIMIT:
                        return chunk
        return ""

    def on_apply_matrix(self):
        self.stop_scan()
        raw = self.txt_matrix.get("1.0", "end")
        values, code = parse_matrix(raw)
        if code:
            if code == "empty":
                self.flag_bad(None, "err_matrix_empty", lo=MIN_DIM)
            elif code == "number":
                self.flag_bad(self.txt_matrix, "err_matrix_number",
                              value=self.first_bad_token(raw))
            elif code == "ragged":
                self.flag_bad(self.txt_matrix, "err_matrix_ragged")
            elif code == "limit":
                self.flag_bad(self.txt_matrix, "err_matrix_limit",
                              value=self.first_bad_token(raw, mode="limit"),
                              lim=fmt(VALUE_LIMIT))
            else:
                rows = values or [[0.0]]
                self.flag_bad(self.txt_matrix, "err_matrix_size", lo=MIN_DIM,
                              hi=MAX_DIM, rows=len(rows), cols=len(rows[0]))
            return
        self.clear_bad()
        self.values = values
        self.input_key = "custom"
        self.out_i = self.out_j = 0
        self._fill_combos()
        self.refresh("status_matrix_applied", rows=len(values),
                    cols=len(values[0]))

    def on_kernel_preset(self, event=None):
        self.stop_scan()
        index = self.combo_kernel.current()
        key = KERNEL_PRESETS[index]["key"]
        self.kernel_key = key
        if key != "custom":
            self.kernel = [list(map(float, row))
                           for row in KERNEL_BY_KEY[key]["values"]]
            self._sync_entries()
        self.refresh("status_kernel_preset", name=self.kernel_name(key))

    def on_kernel_entries(self, event=None):
        self.stop_scan()
        values = []
        for index, var in enumerate(self.kernel_vars):
            number, code = parse_scalar(var.get())
            if code:
                entry = self._entry_by_index(index)
                key = ("err_kernel_limit" if code == "limit"
                       else "err_kernel_number")
                self.flag_bad(entry, key, value=var.get().strip(),
                              lim=fmt(VALUE_LIMIT))
                return
            values.append(number)
        self.clear_bad()
        self.kernel = [values[row * 3:row * 3 + 3] for row in range(3)]
        self.kernel_key = "custom"
        self._fill_combos()
        self.refresh("status_kernel_custom")

    def _entry_by_index(self, index):
        return self.kernel_entries[index]

    def on_stride(self, value):
        self.stop_scan()
        self.stride = int(value)
        self.out_i = self.out_j = 0
        self.refresh("status_stride", s=self.stride,
                     total=self.total_positions_after())

    def total_positions_after(self):
        return position_count(self.values, self.kernel, self.stride, self.pad)

    def on_pad(self, value):
        self.stop_scan()
        self.pad = int(value)
        self.out_i = self.out_j = 0
        self.refresh("status_pad", p=self.pad,
                     rows=len(self.values) + 2 * self.pad,
                     cols=len(self.values[0]) + 2 * self.pad,
                     total=self.total_positions_after())

    def on_bias(self, event=None):
        self.stop_scan()
        number, code = parse_scalar(self.var_bias.get())
        if code:
            self.flag_bad(self.ent_bias,
                          "err_bias_limit" if code == "limit"
                          else "err_bias_number",
                          value=self.var_bias.get().strip(),
                          lim=fmt(VALUE_LIMIT))
            return
        self.clear_bad()
        self.bias = number
        self.refresh("status_bias", b=fmt(self.bias))

    def on_activation(self):
        self.stop_scan()
        self.activation = self.var_activation.get()
        self.refresh("status_activation",
                     name=T("act_short_%s" % self.activation))

    def on_speed(self, event=None):
        raw = self.var_speed.get().strip().replace(",", ".")
        number = parse_number(raw)
        if number is None or number < 40 or number > 4000:
            self.flag_bad(self.ent_speed, "err_speed_number")
            return
        self.clear_bad()
        self.scan_speed = int(round(number))
        self.var_speed.set(str(self.scan_speed))
        self.set_status("status_scan", speed=self.scan_speed)

    # --- шаг и обход ---------------------------------------------------
    def set_cell(self, out_i, out_j):
        self.out_i, self.out_j = out_i, out_j
        self.refresh()

    def step_position(self, delta, announce=True):
        total = self.total_positions()
        columns = self.out_columns()
        if not total:
            return False
        index = self.current_index() - 1 + delta
        if index < 0:
            self.set_status("status_edge_first")
            return False
        if index >= total:
            self.set_status("status_edge_last")
            return False
        self.out_i = index // columns
        self.out_j = index % columns
        self.refresh()
        if announce:
            self.set_status("status_step", index=self.current_index(),
                            total=total, i=self.out_i, j=self.out_j,
                            value=fmt(self.cell.output))
        return True

    def reset_step(self):
        self.stop_scan()
        self.out_i = self.out_j = 0
        self.refresh("status_reset")

    def reset_all(self):
        self.stop_scan()
        self.input_key = DEFAULT_INPUT_KEY
        self.values = build_input(DEFAULT_INPUT_KEY)
        self.kernel_key = DEFAULT_KERNEL_KEY
        self.kernel = [list(map(float, row))
                       for row in KERNEL_BY_KEY[self.kernel_key]["values"]]
        self.stride = DEFAULT_STRIDE
        self.pad = DEFAULT_PAD
        self.bias = DEFAULT_BIAS
        self.activation = DEFAULT_ACTIVATION
        self.pool_mode = DEFAULT_POOL_MODE
        self.pool_window = DEFAULT_POOL_WINDOW
        self.pool_i = self.pool_j = 0
        self.out_i = self.out_j = 0
        self.var_stride_group.set(self.stride)
        self.var_pad_group.set(self.pad)
        self.var_activation.set(self.activation)
        self.var_pool_mode.set(self.pool_mode)
        self.var_pool_window.set(self.pool_window)
        self.clear_bad()
        self._sync_entries()
        self._fill_combos()
        self.refresh("status_reset")

    def start_scan(self):
        if self.scan_running or not self.total_positions():
            if not self.total_positions():
                self.set_status("status_impossible",
                                reason=T("err_impossible", k=KERNEL_SIZE,
                                         rows=len(self.values),
                                         cols=len(self.values[0]),
                                         s=self.stride, p=self.pad))
            return
        if self.current_index() >= self.total_positions():
            self.out_i = self.out_j = 0
            self.recompute()
        self.scan_running = True
        self.btn_scan_start.configure(state="disabled")
        self.btn_scan_pause.configure(state="normal")
        self.draw_all()
        self.set_status("status_scan", speed=self.scan_speed)
        self._schedule_scan()

    def _schedule_scan(self):
        self.scan_job = self.after(self.scan_speed, self._scan_tick)

    def _scan_tick(self):
        self.scan_job = None
        if not self.scan_running:
            return
        if self.current_index() >= self.total_positions():
            self.pause_scan()
            self.set_status("status_scan_end")
            return
        self.step_position(1, announce=False)
        self._schedule_scan()

    def pause_scan(self):
        if not self.scan_running:
            return
        self.scan_running = False
        if self.scan_job is not None:
            self.after_cancel(self.scan_job)
            self.scan_job = None
        self.btn_scan_start.configure(state="normal")
        self.btn_scan_pause.configure(state="disabled")
        self.set_status("status_scan_pause", index=self.current_index(),
                        total=self.total_positions())

    def toggle_scan(self):
        if self.scan_running:
            self.pause_scan()
        else:
            self.start_scan()

    def stop_scan(self):
        self.scan_running = False
        if self.scan_job is not None:
            self.after_cancel(self.scan_job)
            self.scan_job = None
        try:
            self.btn_scan_start.configure(state="normal")
            self.btn_scan_pause.configure(state="disabled")
        except tk.TclError:
            pass

    def reset_scan(self):
        self.stop_scan()
        self.reset_step()

    # --- выбор мышью ----------------------------------------------------
    def _origin_for_input(self, row, column):
        """Окно, накрывающее кликнутую клетку; иначе — ближайшее начало."""
        best = None
        for step in self.result["cells"]:
            covers = (step.win_top <= row < step.win_top + KERNEL_SIZE and
                      step.win_left <= column < step.win_left + KERNEL_SIZE)
            distance = abs(step.win_top + (KERNEL_SIZE - 1) / 2.0 - row) + \
                abs(step.win_left + (KERNEL_SIZE - 1) / 2.0 - column)
            score = distance - (1000.0 if covers else 0.0)
            if best is None or score < best[0]:
                best = (score, step)
        return best[1] if best else None

    def on_input_click(self, row, column):
        if not self.total_positions():
            return
        step = self._origin_for_input(row, column)
        if step is None:
            return
        self.stop_scan()
        self.out_i, self.out_j = step.out_i, step.out_j
        self.refresh("status_cell_input", r=row, c=column)

    def on_map_click(self, row, column):
        if not self.total_positions():
            return
        self.stop_scan()
        step = cell_at(self.values, self.kernel, self.stride, self.pad,
                       self.bias, self.activation, row, column)
        self.out_i, self.out_j = row, column
        self.recompute()
        self.draw_all()
        self.set_status("status_cell_map", i=row, j=column,
                        r0=step.src_top, r1=step.src_top + KERNEL_SIZE - 1,
                        c0=step.src_left, c1=step.src_left + KERNEL_SIZE - 1)

    # --- пулинг: выбор и переключения -----------------------------------
    def on_pool_mode(self):
        self.stop_scan()
        self.pool_mode = self.var_pool_mode.get()
        self.refresh("status_pool_mode", name=T("pool_%s" % self.pool_mode),
                     rows=self.pool_rows(), cols=self.pool_columns())

    def on_pool_window(self, value):
        self.stop_scan()
        self.pool_window = int(value)
        self.pool_i = self.pool_j = 0
        # Числа в статусе считаются после пересчёта: до него pool_result ещё
        # относится к прежнему окну.
        self.refresh()
        self.set_status("status_pool_window", w=self.pool_window,
                        rows=self.pool_rows(), cols=self.pool_columns(),
                        total=self.pool_total())

    def on_pool_click(self, row, column):
        if not self.pool_total():
            return
        self.stop_scan()
        self.pool_i, self.pool_j = row, column
        self.recompute()
        self.draw_all()
        step = self.pool_step
        self.set_status("status_pool_cell", i=row, j=column,
                        r0=step.top, r1=step.top + step.window - 1,
                        c0=step.left, c1=step.left + step.window - 1)

    def on_pool_source_click(self, row, column):
        """Клик по карте Y: окно пулинга встаёт так, чтобы накрыть клетку."""
        origin = pool_window_for(self.pooling_source(), self.pool_window,
                                 row, column)
        if origin is None:
            self.set_status("status_pool_unused", r=row, c=column,
                            w=self.pool_window)
            return
        self.stop_scan()
        self.pool_i, self.pool_j = origin
        self.recompute()
        self.draw_all()
        self.set_status("status_pool_src", r=row, c=column)

    # --- клавиатура -----------------------------------------------------
    def _editing_field(self, event):
        """Поле ввода в фокусе: клавиши принадлежат ему, а не шагам свёртки."""
        widget = getattr(event, "widget", None)
        return isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, tk.Listbox))

    def on_key_step(self, event, delta):
        if self._editing_field(event):
            return None
        self.step_position(delta)
        return "break"

    def on_key_home(self, event):
        if self._editing_field(event):
            return None
        self.reset_step()
        return "break"

    def on_key_scan(self, event):
        if self._editing_field(event):
            return None
        self.toggle_scan()
        return "break"

    # --- эксперименты ---------------------------------------------------
    def experiment_config(self, entry):
        values = build_input(entry["input_key"])
        kernel = [list(map(float, row))
                  for row in KERNEL_BY_KEY[entry["kernel_key"]]["values"]]
        result = convolve(values, kernel, entry["stride"], entry["pad"],
                          entry["bias"], entry["activation"])
        return result

    def on_experiment_select(self, event):
        selected = self.exp_list.curselection()
        if not selected:
            return
        self.experiment_key = EXPERIMENTS[selected[0]]["key"]
        self.show_experiment()

    def show_experiment(self):
        entry = EXPERIMENT_BY_KEY[self.experiment_key]
        result = self.experiment_config(entry)
        widget = self.txt_exp
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", T("exp_%s_name" % entry["key"]) + "\n", "h")
        widget.insert("end", T("exp_%s_goal" % entry["key"]) + "\n", "t")
        widget.configure(state="disabled")
        cell = None
        if result["total"]:
            cell = result["cells"][min(entry["out_i"] *
                                       result["out_columns"] + entry["out_j"],
                                       result["total"] - 1)]
        self.lbl_exp_result.configure(
            text=T("exp_result", rows=result["out_rows"],
                   cols=result["out_columns"],
                   index=cell.index + 1 if cell else 0,
                   total=result["total"],
                   value=fmt(cell.output) if cell else "—"))
        self.draw_experiment_previews(entry, result)

    def draw_experiment_previews(self, entry, result):
        """Для шага и заполнения показываем и второй вариант — тот, с которым
        сравнивают: так видно, что именно меняет параметр."""
        variants = []
        if entry["key"] == "padding":
            values = build_input(entry["input_key"])
            kernel = self.experiment_kernel(entry)
            variants = [
                (T("pad_short_0"),
                 convolve(values, kernel, entry["stride"], 0, entry["bias"],
                          entry["activation"])),
                (T("pad_short_1"),
                 convolve(values, kernel, entry["stride"], 1, entry["bias"],
                          entry["activation"])),
            ]
        elif entry["key"] == "stride":
            values = build_input(entry["input_key"])
            kernel = self.experiment_kernel(entry)
            variants = [
                (T("stride_short_1"),
                 convolve(values, kernel, 1, entry["pad"], entry["bias"],
                          entry["activation"])),
                (T("stride_short_2"),
                 convolve(values, kernel, 2, entry["pad"], entry["bias"],
                          entry["activation"])),
            ]
        else:
            variants = [(T("cap_map"), result)]
        widest = max([max(item["out_rows"], item["out_columns"])
                      for _, item in variants] + [1])
        size = max(14, min(30, 480 // widest))
        for index, tile in enumerate(self.exp_tiles):
            if index >= len(variants):
                tile["frame"].pack_forget()
                continue
            caption, item = variants[index]
            tile["frame"].pack(side="left", anchor="nw")
            tile["title"].configure(text=caption)
            tile["view"].cs = size
            scale = abs_scale(item["values"])
            chosen = item is result and item["total"] > 0
            spec = []
            for i, row in enumerate(item["values"]):
                line = []
                for j, value in enumerate(row):
                    current = chosen and i == entry["out_i"] and \
                        j == entry["out_j"]
                    line.append(cell_spec(
                        text=fmt(value),
                        bg=diverging_color(value, scale),
                        edge=COL_WIN if current else None,
                        edge_width=2))
                spec.append(line)
            tile["view"].render(spec)

    def experiment_kernel(self, entry):
        return [list(map(float, row))
                for row in KERNEL_BY_KEY[entry["kernel_key"]]["values"]]

    def on_experiment_apply(self):
        entry = EXPERIMENT_BY_KEY[self.experiment_key]
        self.apply_experiment(entry)

    def apply_experiment(self, entry):
        self.stop_scan()
        self.input_key = entry["input_key"]
        self.values = build_input(entry["input_key"])
        self.kernel_key = entry["kernel_key"]
        self.kernel = [list(map(float, row))
                       for row in KERNEL_BY_KEY[entry["kernel_key"]]["values"]]
        self.stride = entry["stride"]
        self.pad = entry["pad"]
        self.bias = float(entry["bias"])
        self.activation = entry["activation"]
        self.out_i = entry["out_i"]
        self.out_j = entry["out_j"]
        self.var_stride_group.set(self.stride)
        self.var_pad_group.set(self.pad)
        self.var_activation.set(self.activation)
        self.clear_bad()
        self._sync_entries()
        self._fill_combos()
        self.refresh()
        self.nb.select(0)
        self.set_status("status_experiment",
                        name=T("exp_%s_name" % entry["key"]),
                        goal=T("exp_result", rows=self.out_rows(),
                               cols=self.out_columns(),
                               index=self.current_index(),
                               total=self.total_positions(),
                               value=fmt(self.cell.output) if self.cell
                               else "—"))

    # --- справка --------------------------------------------------------
    def on_help_select(self, event):
        selected = self.help_list.curselection()
        if not selected:
            return
        self.help_key = HELP_KEYS[selected[0]]
        self.show_help()

    def help_move(self, delta):
        index = HELP_KEYS.index(self.help_key) + delta
        index = max(0, min(index, len(HELP_KEYS) - 1))
        self.help_key = HELP_KEYS[index]
        self._fill_help_list()
        self.show_help()

    def show_help(self):
        widget = self.txt_help
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", T("help_%s_title" % self.help_key) + "\n", "h")
        widget.insert("end", T("help_%s_text" % self.help_key) + "\n", "t")
        widget.configure(state="disabled")

    # --- язык -----------------------------------------------------------
    def switch_language(self):
        code = self.var_lang.get()
        if code == get_language():
            return
        self.stop_scan()
        set_language(code)
        self.title(T("app_title"))
        for widget, key, attr, kwargs in self._text_refs:
            try:
                widget.configure(**{attr: T(key, **kwargs)})
            except tk.TclError:
                pass
        for frame, key in self._tabs:
            self.nb.tab(frame, text=T(key))
        self._fill_combos()
        self._fill_experiment_list()
        self._fill_help_list()
        self.refresh()
        self.show_help()
        self.show_experiment()
        name = T("lang_ru") if code == "ru" else T("lang_en")
        self.set_status("status_lang", name=name)

    def current_state(self):
        """Состояние для проверок: смена языка ничего в нём не трогает."""
        try:
            tab = self.nb.index(self.nb.tabs().index(
                self.nb.select()))
        except (tk.TclError, ValueError):
            tab = -1
        return {
            "input_key": self.input_key,
            "values": [list(row) for row in self.values],
            "kernel_key": self.kernel_key,
            "kernel": [list(row) for row in self.kernel],
            "stride": self.stride,
            "pad": self.pad,
            "bias": self.bias,
            "activation": self.activation,
            "out_i": self.out_i,
            "out_j": self.out_j,
            "index": self.current_index(),
            "total": self.total_positions(),
            "pool_mode": self.pool_mode,
            "pool_window": self.pool_window,
            "pool_i": self.pool_i,
            "pool_j": self.pool_j,
            "pool_total": self.pool_total(),
            "pool_value": self.pool_step.output if self.pool_step else None,
            "tab": tab,
            "language": get_language(),
        }


# ----------------------------------------------------------------------
# Раздел 10. Точка входа
# ----------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
