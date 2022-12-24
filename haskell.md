Анонимная фунция -- лямбда - абстракция

Бетта-редукция - применяем аргумент к лямбда-абстракции

Связываение = 



Ленивая стратегия -- самый левый внешний

Чистое лямбда исчисление - кроме вычислительного движка больше ничего нет



Лямбда терм -- это переменная, абстракция или аппликация.

Если можем приписать тип - это терм, иначе предтерм. 



## 19. Главная пара и главный тип. Теорема Хиндли-Милнера. 

Главный тип - это тип, из которого можно вывести все другие типы подстановками типа вместо типовых переменных.

1. Когда вводим переменные нужно обеспечивать максимальное разнообразие. 
2. Если мы можем ввести переменную без стрелки, то нужно это сделать. Стрелка должна возникать из аппликации. 



## 2. Каррирование

Была функция f, которая принимала много аргументов (arg1, arg2)

`curry :: ((a, b) -> c) -> a -> b -> c`

`uncurry :: (a -> b -> c) -> (a, b) -> c`



## 3. Оператор

Оператор -- это функция, которую можно расположить между двумя аргументами. 

##### Приоритет 

В Haskell приоритет задается целым числом, чем больше число, тем выше приоритет и тем раньше выполнится оператор.

##### Ассоциативность

- ассоциативен справа, если выражение a \/ b \/ c вычисляют как a \/ (b \/ c)

- ассоциативен слева, если выражение a \/ b \/ c вычисляют как (a \/ b) \/ c

- неассоциативен, если выражение a \/ b \/ c запрещено записывать без скобок

- infixr – ассоциативность справа

  infixl – ассоциативность слева

  infix – неассоциативный оператор

##### Сечения

Сечения (sections) — синтаксический сахар для частичного применения как к левому, так и к правому аргументу.

```
*Main> ( / 3) 2
0.6666666666666666
*Main> (3 / ) 2
1.5
```

## 4.

Нестрогие функции -- игнорируют аргумент

Строгие -- не игнорируют

Энергичное исполение -- аргумент вычисляется до исполнения функции (не про хаскель)

Ленивое исполнение -- значение преобразуется внутри тела функции

## 7. Трансляция образцов в Kernel. Синонимы в образцах, ленивые и охранные образцы. Образцы в λ- и let-выражениях.

#### Трансляция в Karnel

```haskell
head (x:_) = x
head [] = error "head: empty list"
```

```haskell
head'' xs = case xs of
(x:_) -> x
[] -> error "head'': empty list"
```

#### Линивые образцы

- задаются с помощью `~`
- сапоставление всегда проходит успешно
- динамическое связывание откладывается до момента использования

```haskell
f g ~(x, y) = (f x, f y)
```

#### Охранные образцы

```haskell
firstOdd :: [Integer] -> Integer
firstOdd xs | Just x <- find odd xss, x > 1000 = x
| otherwise = 0
```

#### Образцы в λ-выражениях

```haskell
head''' = \(x:_) -> x
```

```
\p1 ... pn -> e1 ≡
    \x1 ... xn -> case (x1,...,xn) of (p1,...,pn) -> e1
```

#### Образцы в let-выражениях

```haskell
let x:xs = "ABC" in xs ++ xs
```

```
let p = e1 in e ≡ case e1 of ~p -> e
```



## 8. Объявления type и newtype. Метки полей. Строгие конструкторы данных.

- `type` -- без конструктора
- `newtype` -- 1 конструктор, во время исполнения программы не нужен. Во время исполения программы это просто Int 
- `data` -- 1 конструктор обзятельно. Какой-то дополнительный уровень косвенности.

##### `type`

в Haskell очень часто само имя или конструктор типа ялвяется очень большим, поэтому удобно для такого типа задавать более короткий, используя ключевое слово `type` (то есть *синоним типа*). 

```haskell
type String = [Char]
```

##### `newtype`

- задает новый тип c единственным однопараметрическим конструктором, упаковывающий уже существующий тип
- это обертка над сущесвтующим типом с единственным конструктором
- все представители, которые имеются у упаковываемого типа, пропадают
- тип, определенный с помощью `newtype`, гарантированно имеет один конструктор с одним параметром. Так как конструктор один, во время исполнения программы он не нужен. Если мы определяем `data`, то конструктор нужен, даже если он один, для сопоставления с образцом.

```haskell
newtype AgeNT = AgeNT Int
```

```haskell
newtype AgeNT = AgeNT { getAgeNT :: Int }
```

```
GHCi> age = AgeNT 42
GHCi> :t age
age :: AgeNT
GHCi> age
AgeNT {getAgeNT = 42}
GHCi> getAgeNT age
42
```

##### Метки полей

Пусть есть точка `data Point a = Pt a a`

В этой точке хранятся координаты. Хотим их вынуть. 

Можно им присвоить имена -- называются "метки" в хаскеле. 

```haskell
data Point a = Pt { ptX :: a, ptY :: a }
  deriving Show
```

- позволяют в таком, например, случае не указывать все поля и не ставить вместо них _ если значения не нужны

  ```haskell
  absP' Pt {ptX = x, ptY = y} = sqrt (x ^ 2 + y ^ 2)
  ```

##### Строгие конструкторы данных

Haskell -- ленивый язык, он доводит выражение до слабой головной нормальной формы. Но бывают ситуации, когда хочется довести вычисления до конца. Используем флаги строгости. 

```haskell
data Comlex a = !a :+ !a
```



## 10. Внутренняя реализация классов типов.

```haskell
data Eq' a = MkEq (a -> a -> Bool) (a -> a -> Bool)
```

- `eq` и `ne` вытаскивают нужные функции

```haskell
eq (MkEq e _) = e
ne (MkEq _ n) = n
```



## 11. Стандартные классы типов: Num и его наследники, Show и Read. 

...



## 12. Полугруппы и моноиды. Представители класса типов Monoid

Полугруппа — это множество с ассоциативной бинарной операцией над ним.

```haskell
(x <> y) <> z ≡ x <> (y <> z)
```

```haskell
class Semigroup a where
  (<>) -- бинарная операция
  sconcat -- сворачивает список
  stimes -- stimes 4 "A" == "AAAA"
```

Моноид -- полугруппа + нейтральный элемент

```haskell
class Semigroup a => Monoid a where
  mempty :: a
  
  mappend :: a -> a -> a
  mappend = (<>)
  
  mconcat :: [a] -> a
  mconcat = foldr mappend mempty
```

##### Представитель -- bool

**Bool** моноид дважды (относительно конъюнкции **(&&)** и дизъюнкции **(||)**). Чтобы реализовать разные интерфейсы для одного типа, упакуем его в обертки **newtype**.

```haskell
newtype All = All { getAll :: Bool }
  deriving (Eq, Ord, Read, Show, Bounded)

instance Semigroup All where
  All x <> All y = All (x && y)
  
newtype Any = Any { getAny :: Bool }
  deriving (Eq, Ord, Read, Show, Bounded)

instance Semigroup Any where
  Any x <> Any y = Any (x || y)
```

Для моноида нужен еще **mempty** -- **True** в одном случае и **False** -- в другом.



## 13. Свёртки списков. Правая и левая свёртки. Энергичные версии. Развертки. Правило foldr/build

- берем список и хотим получить значение

```haskell
foldr :: (a -> b -> b) -> b -> [a] -> b
foldr f ini [] = ini
foldr f ini (x:xs) = x `f` (foldr f ini xs)
```

```haskell
foldl :: (b -> a -> b) -> b -> [a] -> b
foldl f ini [] = ini
foldl f ini (x:xs) = foldl f (f ini x) xs
```

Проблема левой свертки -- thunk -- отложенные вычисления.

```haskell
foldl' :: (b -> a -> b) -> b -> [a] -> b
foldl' f ini [] = ini
foldl' f ini (x:xs) = arg `seq` foldl' f arg xs
		where arg = f ini x
		
⊥ `seq` b = ⊥
a `seq` b = b
```

##### Развертка

- берем значение и хотим получить список

`[ini, g(ini), g(g(ini)), ...]`

```haskell
unfoldr :: (b -> Maybe (a, b)) -> b -> [a] -- без Maybe нет способа остановиться
unfoldr g ini
	| Nothing <- next = []                   -- остановили процесс
	| Just (a,b) <- next = a : unfoldr g b   -- продолжаем
	where next = g ini
```

##### Правило foldr/build 

- `build`заменяет `cons` на `:` и `nil` на `[]`
- `foldr` делает обратную операцию

```haskell
foldr c n (build g) = g c n
```



## 14. Класс типов Foldable и его представители.

- есть разные контейнеры (один из них список). все контейнеры обозначаем, как `t a`

```haskell
class Foldable t where
    -- 1) правая свертка
	foldr :: (a -> b -> b) -> b -> t a -> b
	-- 2) левая свертка
	foldl :: (b -> a -> b) -> b -> t a -> b
	-- 3) сворачиваем контейнер моноидов в моноид
	fold :: Monoid m => t m -> m
	fold = foldMap id
	-- 4) берет контейнер с произвольными значениями a и функцию, которая делает из а m
	foldMap :: Monoid m => (a -> m) -> t a -> m
	-- foldMap f cont = fold (fmap f cont)
	foldMap f cont = foldr (mappend . f) mempty
```



```haskell
instance Foldable [] where 
	foldr f ini [] = ini
	foldr f ini (x:xs) = f x (foldr f ini xs)

	foldl f ini [] = ini
	foldl f ini (x:xs) = foldl f (f ini x) xs
```



## 15. Класс типов Functor и его представители.

- `fmap` берет функцию `a -> b` и поднимает её на уровень контейнера. 
- можно говорить контейнер, но аккуратнее говорить "вычислительный контекст"

```haskell
class Functor f where
    fmap :: (a -> b) ->  f a ->       f b
    ---       (+10)   [1, 2, 3]   [11, 12, 13]
```

```haskell
infixl 4 <$, <$>, $>
class Functor f where
    -- эти функции можно перегрузить
    fmap :: (a -> b) -> f a -> f b
    (<$) :: a -> f b -> f a     -- вместо функции одно занчение - 
    (<$) = fmap . const         -- заменяет все элементы контейнера на то, что передали

-- эти перегрузить нельзя
(<$>) :: Functor f => (a -> b) -> f a -> f b
(<$>) = fmap

($>) :: Functor f => f a -> b -> f b
($>) = flip (<$)
```



## 16. Applicative

```haskell
infixl 4 <*>, *>, <*, <**>
class Functor f => Applicative f where
    {-# MINIMAL pure, ((<*>) | liftA2) #-}
    pure :: a -> f a

    (<*>) :: f (a -> b) -> f a -> f b
    (<*>) = liftA2 id

    liftA2 :: (a -> b -> c) -> f a -> f b -> f c
    liftA2 g a b = g <$> a <*> b

-- дополнительные фиговины, берут только значение оттуда, куда указывают
    (*>) :: f a -> f b -> f b       
    a1 *> a2 = (id <$ a1) <*> a2

    (<*) :: f a -> f b -> f a
    (<*) = liftA2 const
```

Пример `Maybe`



## 17. Alternative

- `<|>` -- операция сложения

```haskell
class Applicative f => Alternative f where
  empty :: f a
  (<|>) :: f a -> f a -> f a
infixl 3 <|>
```

Моноид настроен на простые типы данных, а Alternative -- на контексты

## 17. MonadPlus

```haskell
class (Alternative m, Monad m) => MonadPlus m where
  mzero :: m a
  mzero = empty

  mplus :: m a -> m a -> m a
  mplus = (<|>)
```



## 18. Аппликативные парсеры

1) Простейший парсер 
   `type Parser a = String -> a`

2) `type Parser a = String -> (String, a)`
   это поможет комбинировать парсеры

3) `type Parser a = String -> Maybe (String, a)`

   `type Parser a = String -> Either String (String, a)` -- справа хранится ответ, если все хорошо, а слева -- ошибка, если все плохо. 

```haskell
newtype Parser tok a = Parser {runParser :: [tok] -> Maybe ([tok], a)}
```



Парсер -- представитель функтора 



Парсер -- аппликативный функтор

- позволит запускать 2 парсера на одной строке 

```haskell
newtype Parser tok a = Parser { runParser :: [tok] -> Maybe ([tok], a) }

instance Functor (Parser tok) where
    fmap :: (a -> b) -> Parser tok a -> Parser tok b
    fmap g (Parser p) = Parser f where
        f xs = case p xs of
            Just (cs, c) -> Just (cs, g c)
            Nothing      -> Nothing

instance Applicative (Parser tok) where
    pure :: a -> Parser tok a
    pure x = Parser $ \s -> Just (s, x)

    (<*>) :: Parser tok (a -> b) -> Parser tok a -> Parser tok b
    Parser u <*> Parser v = Parser f where
        f xs = case u xs of
            Nothing       -> Nothing
            Just (xs', g) -> case v xs' of
                Nothing        -> Notjing
                Just (xs'', x) -> Just (xs'', g x)
```



## 19. Монады

Стрелка Клейсли: `a -> m b` (это "вычисление")

`m :: * -> *` -- однопараметрический конструктор (вычислительный контекст).  

Пример `a -> Maybe b`



В монаде 3 функции:

1) `return :: a -> m a` 
   `return = pure` 

2) `(>>=) :: m a -> (a -> m b) -> m b`     ~ bind
   у каждого представителя свой. 

3) `(>>) :: m a -> m b -> m b`
   `m1 >> m2 = m1 >>= \_ -> m2 `

   

Пишем стрелку Клейсли: 

```haskell
toKl :: Monad m => (a -> b) -> (a -> m b)
toKl f = return . f
```



##### Identity

```haskell
instance Monad Indentity where
	(>>=) :: Identity a -> (a - Identity b) -> Identity b
	Identity x >>= k = k x
	
	return = Identity
```



##### Законы

```haskell
return a >>= k = k a
m >>= return = m
(m >>= k) >>= k' = m >>= (\x -> k x >>= k')
```



##### do нотация

-- синтакцический сахар

Оператор присваивания `<-`   -- монадические вычисления, из которых мы что-то вынимаем. 

## 20. MonadFail

В do нотации мы можем использовать не только переменные, но и образцы

типа: `m <- getM person` <-> `('A':xs) <- getM preson`

Сапоставление с образциом может быть удачным и нет. 

Если у нас произошло неудачное сапоставление с образцом, то в монаде Maybe мы можем просто вернуть Nothing. 

```haskell
class Monad m => MonadFail m where
	fail :: String -> m a
	
instance MonadFail Maybe where
	fail _ = Nothing
	
instance MonadFail [] where
	fail _ = []
```

Если мы не напишем MonadFail, то код неработающий код не пройдет typeChecking. 

```haskell
func x = do {
    2 <- Just 2;
    return 'Z'
}
```



## 21. Maybe и списки. 

```haskell
instance Monad [] where
	return x = [x]
	
	(>>=) :: [a] -> (a -> [b]) -> [b]
	xs >>= k = concat (map k xs)
	-- для каждого элемента из xs делаем вычиление ~ пораждаем много списком, а потом склеивам
```

```
"abc" >>= replicate 4
"aaaabbbbcccc"
```



- одно и тоже
- вычлинения представляют собой дерево

```haskell
list = do
    x <- [1, 2, 3]
    y <- [1, 2]
    return (x, y)

list2 = [(x, y) | x <- [1, 2, 3], y <- [1, 2]]
```





## 23. Монада IO

- обеспечивает ввод - вывод
- `getCharFromConsole :: RealWorld -> (RealWorld, Char)`  -- устроено, как State
  RealWorld нужно спрятать от пользователя
- `newType IO a = IO (State# RealWorld -> (# State# RealWorld, a #))`

## 24. Монада Reader

- чтение из внешнего окружения
- происходит от `instance Monad ((->), r)`. Движок - частично примененная функциональная стрелка. 
- Стандартный интерфейс
  - ask   `e <- ask`   Вытаскивает окружения
  - asks   `e <- asks $ lookup Person`   Обычно окружение это что-то большое. В этом случае мы получаем что-то большое из окружения и что-то с этим делаем. 
  - local    `e <- local (("Mike", 1):) userCount`   Подменяет окружение, в котором мы работаем. Например, добавляет туда что-то. 

```haskell
instance Monad ((->), r) where
	return :: a -> (r -> a)
	return x = \_ -> x
	
	(>>=) :: (r -> a) -> (a -> (r -> b)) -> (r -> b)
	--          m               k
	-- тип e -- r
	m >>= k = \e -> k (m e) e
```

```haskell
newtype Reader r a = Reader { runReader :: r -> a }
-- r - это тип окружения, a -- это тип, который есть в любой монаде

reader :: (r -> a) -> Reader r a
runReader :: Reader r a -> r -> a
```



## 25. Writer

- запись в лог
- движок -- это пара (a, w). w -- лог
- Стандартный интерфейс:
  - tell    `tell 1`     просто записывает что-то в лог
  - listen       какой же лог локально записался в данной записи
  - listens       аналог asks
  - censor       изменяет лог



## 26. State

- движок `s -> (a, s)`   -- композиция Reader и Writer
- Функции:
  - `runState :: State s a -> s -> (a, s)`
  - `execState :: State s a -> s -> s`
  - `evalState :: State s a -> s -> a`  
- Стандартный интефейс
  - get
  - put
  - `modify f = do  s <- get   put (f s)`
  - gets f

## 27. Except 

- движок `Either e a`
- `newtype Except e a = Except { runExcept :: Either e a }`
- Стандартный интерфейс:
  - throwE
  - catchE

## 28. Мультипарметрические классы типов

- это называет функциональная зависимость:

```haskell
class Mult a b c | a b -> c where
	(***) :: a -> b -> c
```



Рассмотрим, например, монаду Reader. В монаде Reader есть стандратный интерфейс. Одна из функций там -- это функция `ask :: Reader r r`. Предположим у нас есть другая монада m и мы хотим наделить её интерфейсом Reader. В частности, в монаде m должна быть функция `ask :: m r`. 

Как это сделать? Мы должны сделать функцию `ask` функцией - членом некоторого класса типов. У каждой монады должен быть свой r.

```
class Monad m => MonadReader r m | m -> r where
	ask :: m r
	local :: (r -> r) -> m a -> m a 
```



## 29. Трансформеры монад

Если мы захотим сделать композицию монад, то для каждой пары монад нужно будет писать отдельную композицию. n монад -> n^2 композиций. 

Оказывается, можно взять произвольную монаду m и накрутить на неё определенную монаду, например, State. Это делается с помощью трансформера. Таким образом, нужно будет написать всего n трансформеров. 

Трансформер монад -- это конструктор типа, который принимает монаду в качестве аргумента и возвращает монаду как результат. 



Есть 2 библиотеки работы со стандартными монадами. 

Отличие transformers от mtl (monad transformer library):

- mtl

  - использует некоторые дополнительные расширения, которые делают работу красивее и компактнее
  - в современно мире в основном использеют её
  - построена на основе библиотеки transformers
  - здесь используются функциональные зависимости

  - в mtl все стандартные интерфейсы монад абстрактные

- в transformers функции принадлжат конктретным монадам