# Курс Технологии компьютерного зрения

### 2 "Tracking. Re-Identification. Sort, DeepSort":
Студенту необходимо реализовать методы tracker_soft, tracker_strong и метод подсчета метрик, а также написать отчет. 
Все подробности тут https://github.com/VABer-dv/object-tracking-assignment

Критерии оценки дз по теме:

Оценка неудовлетворительно (2) ставится, если:
- студент все сделал, но не может рассказать как работает его код и что происходит в проекте
- код полностью заимствован у другой команды/из репозитория
- нарушен оговоренный срок сдачи и защиты дз

Оценка удовлетворительно (3) ставится, если:
- студент реализовал только 1 метод трекера + метод подсчета метрик, при этом презентация самой работы произведена плохо, 
студент с трудом отвечает на вопросы по проекту, отчет по работе не отвечает критериям (проговаривал на лекции)

Оценка хорошо (4) ставится, если:
- студент реализовал только 1 метод трекера + метод подсчета метрик, быстро ориентируется в коде проекта, отвечает на
поставленные вопросы на защите, отчет отвечает критериям

или

- студент реализовал 2 метода трекера + метод подсчета метрик, но имеет сложности в презентации и защите работы, 
нуждается в большом количестве наводящих вопросов, чтобы ответить на первоначальный вопрос, имеет недочеты в оформлении 
отчета

Оценка отлично (5) ставится, если:
- студент сдал и защитил работу в кратчайший срок, на ближайшей официальной практике после лекции; при этом работа может
иметь недочеты или шероховатости, но студент должен хорошо разбираться в презентуемом материале и представленном проекте

или

- студент реализовал 2 метода трекера + метод подсчета метрик, быстро ориентируется в коде проекта, отвечает на 
поставленные вопросы на защите, отчет отвечает критериям


Реализованны оба трекинга
soft реализован венгерским алгоритмом с расстоянием между центрами bbox-ов
strong - DeepSORT


В качестве метрики была выбранна кастомная: число изменеий id для каждого злосчастного колобка / число фреймов когда он был задетекчен, (меньше = лучше)
Метрики soft:
Где (x, y) это (random_range, bb_skip_percent)
|число cb| (0, 0)| (10, 0)| (0, 0.1) |  (10, 0.1) |
|---|---|---|---|---|
|5   |0.013853   |0.012917   |0.254491   | 0.332458  |
|10 |0.017848   |0.037281   | 0.615082  | 0.402863  |
|20   |0.090497   |0.102181   |0.749915   | 0.521657  |

Довольно странно что при (10, 0.1) то есть при колебающейся рамке метрика, в целом, выше чем при (0, 0.1) но легко видеть что чем хуже детекции и чем больше обьектов тем трекер справляется хуже

Метрики stong:
Где (x, y) это (random_range, bb_skip_percent)

|число cb, (random_range, bb_skip_percent)| Метрика|
|---|---|
|5, (0, 0)   |0.161905  |
|10 (10, 0.1) |0.144703   |
|20   (10, 0.1) |0.276661   |

(Нарезать картинки для всех вариантов очень тяжело, представлены только лучшие и худшие случаи)

В целом strong лучше чем soft (чего и требовалось ожидать), однако если детекций меньше то простой трекер справляется лучше.

В целом это все, спасибо за внимание.
