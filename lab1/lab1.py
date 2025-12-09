import re
import itertools
import numpy as np


class LPSolver:
    # Класс для реализации симплекс-метода для решения задачи линейного программирования (ЗЛП)

    def __init__(self):
        # Метод для инициализации класса

        # Список коэффициентов целевой функции
        self.objective_coefficients: list[int]

        # Направление оптимизации ('max' или 'min')
        self.objective_sense: str

        # Матрица с коэффициентами ограничений
        self.constraint_matrix: list[list[int]]

        # Список с типами ограничений ('>=', '<=', '=')
        self.constraint_senses: list[str]

        # Список с правыми частями ограничений
        self.constraint_rhs: list[int]

        # Таблица с ЗЛП в каноническом виде (в ограничениях используются только равенства)
        # Пример:
        # [20, 20, 10, 0, 0, 0, "max"]
        # [-4, -3, -2, 1, 0, 0, -33],
        # [-3, -2, -1, 0, 1, 0, -23],
        # [-1, -1, -2, 0, 0, 1, -12]
        self.canonical_problem_table: list[list]

    def load_problem(self, file_path: str) -> None:

        # Паттерн для извлечения коэффициента и индекса X из строки
        pattern = re.compile(r'^([+-]?)(\d*)x(\d+)$')

        def parse_coefficient_index(s: str) -> tuple[int, int]:
            # Функция для извлечения коэффициента и индекса X из строки
            # Примеры работы:
            # "2x1" -> (2, 1)
            # "+3x2" -> (3, 2)
            # "+x3" -> (1, 3)
            # "-x3" -> (-1, 3)
            # "-5x4" -> (-5, 4)
            # "x6" -> (1, 6)

            m = pattern.match(s)
            sign, coefficient, index = m.groups()
            coefficient = int(coefficient) if coefficient else 1
            if sign == '-':
                coefficient = -coefficient

            # В Python индексы начинаются с нуля, так что возвращаем (index - 1)
            return coefficient, int(index) - 1

        # Считываем все непустые строки из txt-файла, убирая '\n' и лишние пробелы по краям
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() != '']

        # Строка с целевой функцией
        objective_function = lines[0]

        # Разбиваем строку с целевой функцией на отдельные элементы
        objective_function = objective_function.split()

        # Удаляем элемент '->' (он не нужен для дальнейших расчётов)
        objective_function.pop(-2)

        # Направление оптимизации ('max' или 'min')
        objective_sense = objective_function.pop(-1)

        # Преобразуем элементы-строки в картежи вида (коэффициент, индекс)
        # Пример преобразования элемента: "-5x4" -> (-5, 4)
        for i in range(len(objective_function)):
            objective_function[i] = parse_coefficient_index(objective_function[i])

        # Список со строками с ограничениями
        constraints = lines[1:]

        # Разбиваем строки с ограничениями на отдельные элементы
        for i in range(len(constraints)):
            constraints[i] = constraints[i].split()

        # Список с типами ограничений ('>=', '<=', '=')
        constraint_senses = []

        # Список с правыми частями ограничений
        constraint_rhs = []

        # Заполняем списки с типами и правыми частями ограничений
        for c in constraints:
            constraint_senses.append(c.pop(-2))
            constraint_rhs.append(int(c.pop(-1)))

        # Преобразуем оставшиеся элементы в формат (коэффициент, индекс)
        for c in constraints:
            for i in range(len(c)):
                c[i] = parse_coefficient_index(c[i])

        # Получаем число значимых переменных
        # (оно равно наибольшему индексу X, встречающемуся в ЗЛП)
        number_of_variables = max(i for _, i in objective_function) + 1
        for c in constraints:
            c_i_max = max(i for _, i in c) + 1
            number_of_variables = max(number_of_variables, c_i_max)

        # Создаём список для коэффициентов целевой функции
        # (изначально заполняем его нулями)
        objective_coefficients = [0 for _ in range(number_of_variables)]

        # Заполняем список коэффициентами целевой функции
        # ci - (coefficient, index)
        for ci in objective_function:
            objective_coefficients[ci[1]] = ci[0]

        # Создаём матрицу с коэффициентами ограничений
        # (изначально заполняем её нулями)
        constraint_matrix = []
        for _ in range(len(constraints)):
            constraint_matrix.append([0 for _ in range(number_of_variables)])

        # Заполняем матрицу с коэффициентами ограничений соответствующими значениями
        for row_i in range(len(constraints)):
            for ci in constraints[row_i]:
                constraint_matrix[row_i][ci[1]] = ci[0]

        # Загружаем полученные параметры ЗЛП в глобальные переменные класса
        self.objective_coefficients = objective_coefficients
        self.objective_sense = objective_sense
        self.constraint_matrix = constraint_matrix
        self.constraint_senses = constraint_senses
        self.constraint_rhs = constraint_rhs

        # Создаём глобальную переменную для хранения базиса симплекс-таблицы
        self.basis_indexes = [None for _ in range(len(constraint_matrix))]

        # Преобразуем параметры ЗЛП к каноническому виду
        self._convert_problem_to_canonical_form()

    def get_solution(self, precision=8):
        # Решает ЗЛП методом перебора вершин (без симплекс-метода)

        c = self.objective_coefficients[:]
        sense = self.objective_sense
        A_ineq, b_ineq, A_eq, b_eq, n = self._build_standard_sets()

        k = len(A_eq)
        # Если равенств больше, чем число переменных: проверим совместность
        if k > n:
            Aeq = np.array(A_eq, dtype=float)
            beq = np.array(b_eq, dtype=float)
            x_ls, residuals, rank, s = np.linalg.lstsq(Aeq, beq, rcond=None)
            if residuals.size > 0 and residuals[0] > 1e-8:
                return
            if np.any(x_ls < -1e-9):
                return
            if len(A_ineq) > 0:
                if np.any(np.matmul(np.array(A_ineq, dtype=float), x_ls) - np.array(b_ineq, dtype=float) > 1e-9):
                    return
            val = float(np.dot(c, x_ls))
            return self._round_result([list(x_ls[:len(c)]), val], precision)

        # Подготовим полные неравенства с неотрицательностью: -I x <= 0
        A_ineq_full = [row[:] for row in A_ineq]
        b_ineq_full = b_ineq[:]
        for i in range(n):
            row = [0.0] * n
            row[i] = -1.0
            A_ineq_full.append(row)
            b_ineq_full.append(0.0)

        choose_cnt = n - k
        if choose_cnt < 0:
            return

        idx_candidates = list(range(len(A_ineq_full)))
        best_x = None
        best_val = None

        for active in itertools.combinations(idx_candidates, choose_cnt):
            M = []
            d = []
            for i in range(k):
                M.append(A_eq[i][:])
                d.append(b_eq[i])
            for ai in active:
                M.append(A_ineq_full[ai][:])
                d.append(b_ineq_full[ai])
            M = np.array(M, dtype=float)
            d = np.array(d, dtype=float)
            if M.shape[0] != n:
                continue
            try:
                x = np.linalg.solve(M, d)
            except np.linalg.LinAlgError:
                continue
            if len(A_eq) > 0:
                if np.any(np.abs(np.matmul(np.array(A_eq, dtype=float), x) - np.array(b_eq, dtype=float)) > 1e-8):
                    continue
            if len(A_ineq_full) > 0:
                if np.any(np.matmul(np.array(A_ineq_full, dtype=float), x) - np.array(b_ineq_full, dtype=float) > 1e-8):
                    continue
            val = float(np.dot(c, x))
            if best_x is None:
                best_x = x
                best_val = val
            else:
                if sense == "max":
                    if val > best_val + 1e-9:
                        best_x, best_val = x, val
                else:
                    if val < best_val - 1e-9:
                        best_x, best_val = x, val

        if best_x is None:
            return
        best_list = best_x.tolist()[:len(self.objective_coefficients)]
        return self._round_result([best_list, best_val], precision)

    def _convert_problem_to_canonical_form(self):
        # Преобразует ЗЛП к каноническому виду и записывает в единую таблицу для дальнейшей работы

        # Создаём локальные копии параметров ЗЛП
        objective_coefficients = self.objective_coefficients.copy()
        objective_sense = self.objective_sense
        constraint_matrix = [row.copy() for row in self.constraint_matrix]
        constraint_senses = self.constraint_senses.copy()
        constraint_rhs = self.constraint_rhs.copy()

        # Список индексов искусственных переменных в расширенной матрице
        self.artificial_vars = []

        # Перебираем все типы ограничений
        for i in range(len(constraint_senses)):
            # Если знак ограничения не равен "=" (т.е. равен "<=" или ">="),
            # то нужно сделать соответствующее преобразование
            if constraint_senses[i] != "=":
                # Для знака "<=" добавляем новую переменную с коэффициентом 1
                if constraint_senses[i] == "<=":
                    constraint_matrix[i].append(1)

                # Для знака ">=" домножаем всё ограничение на -1 (чтобы изменить знак на "<="),
                # а дальше также добавляем новую переменную с коэффициентом 1
                elif constraint_senses[i] == ">=":
                    for j in range(len(constraint_matrix[i])):
                        constraint_matrix[i][j] = -constraint_matrix[i][j]
                    constraint_rhs[i] = -constraint_rhs[i]
                    constraint_matrix[i].append(1)

                # Для всех остальных строк добавляем эту новую переменную с коэффициентом 0
                objective_coefficients.append(0)
                for j in range(len(constraint_senses)):
                    if j != i:
                        constraint_matrix[j].append(0)

                # Также указываем новую переменную как базисную
                # для соответствующей строки симплекс-таблицы
                self.basis_indexes[i] = len(constraint_matrix[i]) - 1
            else:
                # Для равенства: добавляем искусственную переменную (+1) и ноль в целевую функцию
                # Это позволит запустить фазу I и найти допустимый базис
                for j in range(len(constraint_matrix)):
                    if j == i:
                        constraint_matrix[j].append(1)
                    else:
                        constraint_matrix[j].append(0)
                objective_coefficients.append(0)
                artificial_col_index = len(constraint_matrix[i]) - 1
                self.artificial_vars.append(artificial_col_index)
                self.basis_indexes[i] = artificial_col_index

        # Единая таблица для канонической формы ЗЛП
        canonical_problem_table = []

        # Добавляем в таблицу коэффициенты целевой функции и направление оптимизации
        canonical_problem_table.append(objective_coefficients)
        canonical_problem_table[0].append(objective_sense)

        # Добавляем в таблицу коэффициенты и правую часть ограничений
        for i in range(len(constraint_matrix)):
            canonical_problem_table.append(constraint_matrix[i])
            canonical_problem_table[i + 1].append(constraint_rhs[i])

        # Сохраняем таблицу в глобальную переменную класса
        self.canonical_problem_table = canonical_problem_table

    def _build_standard_sets(self):
        # Возвращает (A_ineq, b_ineq, A_eq, b_eq, n) после приведения:
        # - Все >= преобразованы в <= умножением на -1
        # - '=' попадают в A_eq
        # - Неотрицательность учитывается отдельно (на этапе проверки/перебора)

        n = len(self.objective_coefficients)
        A_ineq = []
        b_ineq = []
        A_eq = []
        b_eq = []
        for row, s, rhs in zip(self.constraint_matrix, self.constraint_senses, self.constraint_rhs):
            if s == "<=":
                A_ineq.append(row[:n])
                b_ineq.append(rhs)
            elif s == ">=":
                A_ineq.append([-v for v in row[:n]])
                b_ineq.append(-rhs)
            else:
                A_eq.append(row[:n])
                b_eq.append(rhs)
        return A_ineq, b_ineq, A_eq, b_eq, n

    def _phase_one_build_feasible_basis(self, eps: float = 1e-9) -> bool:
        # Фаза I: формирование вспомогательной задачи (минимизация суммы искусственных переменных)
        # и нахождение допустимого базиса. Возвращает True, если решение допустимо; иначе False.

        # Построим вспомогательную целевую функцию: коэффициенты 1 для искусственных переменных, 0 для остальных
        c_phase1 = [0 for _ in range(len(self.canonical_problem_table[0]) - 1)]
        for j in self.artificial_vars:
            c_phase1[j] = 1

        # Сохраним состояние основной задачи
        original_c = self.canonical_problem_table[0][:-1]
        original_obj = self.canonical_problem_table[0][-1]

        # Инициализируем таблицу фазы I
        self.c = c_phase1[:]
        self.obj = "min"
        self.st = [row.copy() for row in self.canonical_problem_table[1:]]

        # Если базис не полный — попытаемся сформировать
        if None in self.basis_indexes:
            self._form_basis()

        # Запускаем симплекс для фазы I
        self.deltas = [None for _ in range(len(self.st[0]) - 1)]
        while True:
            self._calculate_deltas()
            if self._checking_all_deltas(self.obj):
                break
            # Разрешающий столбец (максимальная дельта для min)
            resolution_column_j = self.deltas.index(max(self.deltas))
            q = self._calculate_simplex_relations_of_q(resolution_column_j)
            if all(x is None for x in q):
                # Вспомогательная задача неограничена => исходная недопустима
                return False
            row_i = None
            q_min = float("inf")
            for i in range(len(q)):
                if (q[i] is not None) and (q[i] < q_min):
                    q_min = q[i]
                    row_i = i
            self._dividing_row_in_simplex_table(row_i, self.st[row_i][resolution_column_j])
            self._zero_out_other_items_in_the_column(row_i, resolution_column_j)
            self.basis_indexes[row_i] = resolution_column_j

        # Значение вспомогательной целевой функции
        # Сформируем x из базиса
        x_phase = [0 for _ in range(len(self.c))]
        for i, bi in enumerate(self.basis_indexes):
            x_phase[bi] = self.st[i][-1]
        phase_value = sum(x_phase[j] for j in self.artificial_vars)
        if phase_value > eps:
            return False

        # Удаляем искусственные столбцы из таблицы и обновляем базис
        cols_to_remove = sorted(self.artificial_vars, reverse=True)
        for col in cols_to_remove:
            # Если искусственная переменная в базисе, пытаемся заменить её обычной
            if col in self.basis_indexes:
                row_i = self.basis_indexes.index(col)
                replacement_j = None
                for j in range(len(self.st[0]) - 1):
                    if (j not in self.artificial_vars) and abs(self.st[row_i][j]) > eps:
                        replacement_j = j
                        break
                if replacement_j is not None:
                    self._dividing_row_in_simplex_table(row_i, self.st[row_i][replacement_j])
                    self._zero_out_other_items_in_the_column(row_i, replacement_j)
                    self.basis_indexes[row_i] = replacement_j
                else:
                    # Базисная строка полностью зависит от искусственной переменной — просто «выбросим» её из базиса
                    self.basis_indexes[row_i] = None
            # Удаляем столбец из всех строк
            for i in range(len(self.st)):
                del self.st[i][col]
            # Сдвигаем индексы базиса правее удалённого столбца
            for i in range(len(self.basis_indexes)):
                if self.basis_indexes[i] is not None and self.basis_indexes[i] > col:
                    self.basis_indexes[i] -= 1

        # Восстанавливаем целевую функцию основной задачи
        self.c = original_c[:len(self.st[0]) - 1]
        self.obj = original_obj

        # Если остались None в базисе — доведём до полного базиса
        if None in self.basis_indexes:
            self._form_basis()

        return True

    def _get_rid_of_negative_free_coefficients(self):
        # Избавляется от отрицательных свободных коэффициентов b

        while True:
            # Найдём строку с минимальной отрицательной b
            i_min = self._find_row_with_smallest_negative_b()
            # Если такой строки нет, значит все свободные коэффициенты положительны
            if i_min is None:
                return

            # Найдём в строке i_min минимальный отрицательный элемент
            j_min = self._find_minimum_negative_item_in_row(i_min)
            if j_min is None:
                raise Exception("В строке нет отрицательных значений!")

            # Делим строку на найденный элемент
            self._dividing_row_in_simplex_table(i_min, self.st[i_min][j_min])

            # Обнуляем все другие элементы в этом столбце
            self._zero_out_other_items_in_the_column(i_min, j_min)

            # Обновляем базис
            self.basis_indexes[i_min] = j_min

    def _find_row_with_smallest_negative_b(self):
        # Находит строку с минимальным отрицательным b

        b_min = 0
        i_min = None
        for i in range(len(self.st)):
            if self.st[i][-1] < b_min:
                b_min = self.st[i][-1]
                i_min = i
        return i_min

    def _find_minimum_negative_item_in_row(self, i):
        # Находит минимальный отрицательный элемент в строке

        row = self.st[i][:-1]
        min_element = 0
        j_min = None
        for j in range(len(row)):
            if row[j] < min_element:
                min_element = row[j]
                j_min = j
        return j_min

    def _dividing_row_in_simplex_table(self, row_i, divisor):
        # Делит строку в таблице на заданное значение

        row = self.st[row_i]
        for j in range(len(row)):
            row[j] = row[j] / divisor

    def _zero_out_other_items_in_the_column(self, i, j):
        # Приводит к нулю все элементы столбца кроме заданного

        for row_i in range(len(self.st)):
            if row_i == i:
                continue
            row = self.st[row_i]
            subtraction_coeff = row[j]
            for k in range(len(row)):
                row[k] = row[k] - (self.st[i][k] * subtraction_coeff)

    def _calculate_deltas(self):
        # Рассчитывает дельты для симплекс-таблицы

        coeffs = []
        for bi in self.basis_indexes:
            coeffs.append(self.c[bi])
        for j in range(len(self.st[0]) - 1):
            delta_j = 0
            for i in range(len(self.st)):
                delta_j += self.st[i][j] * coeffs[i]
            delta_j -= self.c[j]
            self.deltas[j] = delta_j

    def _checking_all_deltas(self, obj):
        # Проверяет оптимальность решения с помощью дельт

        for delta in self.deltas:
            if ((delta > 0) and (obj == "min")) or ((delta < 0) and (obj == "max")):
                return False
        return True

    def _calculate_simplex_relations_of_q(self, j):
        # Рассчитывает симплекс-отношения Q

        q = []
        for i in range(len(self.st)):
            if self.st[i][j] <= 0:
                q.append(None)
                continue
            q.append(self.st[i][-1] / self.st[i][j])
        return q

    def _form_basis(self):
        # Формирует базис

        for i in range(len(self.basis_indexes)):
            if self.basis_indexes[i] is None:
                for j in range(len(self.st[0]) - 1):
                    if self.st[i][j] != 0:
                        self._dividing_row_in_simplex_table(i, self.st[i][j])
                        self._zero_out_other_items_in_the_column(i, j)
                        self.basis_indexes[i] = j
                        break

    def _round_result(self, result, precision):
        # Округляет ответ до заданной точности

        if isinstance(result, list):
            return [self._round_result(x, precision) for x in result]
        elif isinstance(result, float):
            return round(result, precision)
        else:
            return result


def main():
    sm = LPSolver()
    sm.load_problem('task_lab1.txt')
    answer = sm.get_solution()
    print(answer)


if __name__ == "__main__":
    main()