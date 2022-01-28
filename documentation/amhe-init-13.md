# Algorytm heurystyczny uczący sieć neuronową rozwiązywać problem ze środowiska OpenAI
#### Autorzy: Urszula Tworzydło, Wojciech Rokicki

## 1. Opis zadania

#### 1.1 Treść wybranego zadania

Zaprojektuj algorytm ewolucyjny i zbadaj jego działanie w jednym z zadań zdefiniowanych w ramach środowiska OpenAI poświęconym kontroli.

Specyfikacja zadań znajduje się pod poniższym linkiem:
gym.openai.com/envs/#classic_control
Porównaj wyniki swojego rozwiązania z dwoma wybranymi rozwiązaniami, które bazują na innych podejściach.

#### 1.2 Interpretacja polecenia
Wykorzystać heurystykę, będącą algorytmem genetycznym, w celu nauczenia sieci neuronowej sterowania obiektem w wybranym środowisku [OpenAI](https://gym.openai.com/envs/#classic_control). Otrzymane wyniki porównać z innymi podejściami takimi jak na przykład uczenie ze wzmocnieniem.

#### 1.3 Opis wybranego środowiska
Wybrane środowisko to [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/). Na wózku znajduje się stojąca pionowo tyczka. Celem zadania jest takie kontrolowanie ruchu wózka w prawo i w lewo, aby tyczka utrzymywała pozycję pionową. Zadanie kończy się niepowodzeniem w momencie, gdy kąt pomiędzy tyczką a osią pionową jest większy niż 15 stopni lub gdy wózek przesunie się o 2,4 jednostki od środka planszy.
System jest kontrolowany poprzez nadanie wózkowi siły (+1 lub -1) pod wpływem której wózek poruszy się w prawo lub w lewo. Za utrzymanie tyczki w pozycji pionowej w jednej wyświetlanej klatce zyskuje się nagrodę +1 punkt. Gra zostaje wygrana, jeśli tyczka pozostanie w pozycji pionowej przez wyznaczoną liczbę wyświetlanych klatek.


## 2. Sposób rozwiązania

### 2.1. Ogólny opis programu

Do rozwiązania problemu zostanie użyty algorytm genetyczny. Każda populacja będzie się składać z osobników będących wektorem wag jedno- lub wielowarstwowego perceptronu. 
Prosta sieć neuronowa będzie przyjmować na wejściu cztery parametry określające to, w jakiej pozycji aktualnie znajduje się wózek oraz tyczka w środowisku Cart-Pole-v1. Te parametry to:
* x: horyzontalna pozycja wózka (przy czym x=0 dla początkowej pozycji wózka)
*   v: prędkość wózka (wartości dodatnie oznaczają, że wózek porusza się w prawo)
* θ: kąt pomiędzy tyczką a osią pionową
* ω: prędkość kątowa tyczki (wartości dodatnie oznaczają, że tyczka porusza się zgodnie z kierunkiem zegara)

Jako funkcja aktywacji zostanie zastosowana funkcja sigmoidalna.
Wartość wyjściowa sieci będzie oznaczała czy wózek ma zostać przesunięty w prawo, czy też w lewo. Dla wartości większych lub równych 0,5 będzie to poruszenie wózkiem w prawo, a dla mniejszych w lewo.
Każdy osobnik populacji będzie poddawany ocenie. Będzie ona polegać na obliczeniu i zastosowaniu kolejnych dokonywanych przez sieć predykcji. Liczba dokonanych w środowisku predykcji będzie zależna od tego, czy nastąpi warunek końcowy gry. Oceną osobnika będzie suma zdobytych punktów.

### 2.2 Schemat działania programu

Pierwszym etapem działania programu jest wytworzenie populacji początkowej. Wartości wektora każdego osobnika są dobierane przy zastosowaniu inicjalizacji Xavier'a (wylosowane wartości są skalowane przez współczynnik $\frac{1}{\sqrt{n}}$, gdzie n - rozmiar wejścia sieci, czyli 4).

Dla każdej generacji algorytmu (poza pierwszą, dla której działanie zaczyna się od etapu drugiego) następują dwa etapy działania programu:
1. Działanie algorytmu genetycznego, w skład którego wchodzą następujące etapy:
    * selekcja
    * krzyżowanie
    * mutacja

2. Etap oceny poszczególnych osobników aktualnej populacji - przeprowadzenie ewaluacji poprzez zliczenie nagród z maksymalnej ilości prawidłowych obserwacji dla danego osobnika.

Na koniec wybierany jest najlepszy osobnik z ostatniej populacji, którego wartościami są wagi sieci neuronowej służącej do sterowania wózkiem.


## 3. Planowane eksperymenty numeryczne

### 3.1 Eksperymenty związane z naszą implementacją algorytmu ewolucyjnego

Eksperymenty będą polegać na dobraniu odpowiednich parametrów algorytmu, w procesie porównywania kolejnych zmian wyników przy danych parametrach. Testy będą obejmować zmianę:

- Liczby generacji (epok) algorytmu
- Wielkość populacji
- Parametrów selekcji:
  - metoda selekcji (np. proporcjonalna)
- Parametrów krzyżowania:
  - współczynnik prawdopodobieństwa zajścia mutacji
  - metoda krzyżowania (np. jednopunktowe)
- Parametrów mutacji:
  - współczynnik prawdopodobieństwa zajścia mutacji
  - metoda mutacji (np. wg rozkładu normalnego)
- Wielkości sieci sterującej obiektem

### 3.2 Eksperymenty związane z porównaniem do przykładowego rozwiązania wykorzystującego algorytm uczenia ze wzmocnieniem

Rozwiązanie otrzymane przy wykorzystaniu przedstawionego algorytmu zostanie porównane pod kątem złożoności obliczeniowej z dwoma innymi rozwiązaniami bazującymi na innych podejściach. Będzie to porównanie z algorytmem:
* [Q-Learning](https://github.com/RJBrooker/Q-learning-demo-Cartpole-V1)
* [DQN](https://github.com/gsurma/cartpole)

Eksperymenty będą polegać na porównaniu złożoności obliczeniowej dla znalezionych rozwiązań osiągających podobne wyniki (ilość poprawnych obserwacji).

## 4. Wybrane technologie

Do zrealizowania projektu wykorzystano następujące biblioteki Python'a:
- NumPy - umożliwiająca przeprowadzanie naukowych obliczeń. Zawiera m. in. narzędzia potrzebne w algebrze liniowej, statystyce oraz w generowaniu liczb losowych.
- Matplotlib - dostarcza narzędzi służących do wizualizacji danych m. in. w formie wykresów.
- Gym - zestaw narzędzi służący do tworzenia i porównywania działania algorytmów uczenia ze wzmocnieniem. Dostarcza środowisk, w których agent wchodząc w interakcję z otoczeniem może uczyć się zachowań, które będą dążyć ku zdefiniowanemu celu.
