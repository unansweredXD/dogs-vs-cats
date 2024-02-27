# dogs-vs-cats
# 52 team (Корешков, Потехин, Прытов)

### Быстрый запуск

Чтобы запустить проект необходимо прописать в консоли следующие (все действия выполняются в корне проекта (не src)):

```commandline
pip install -r requirements.txt
mkdir dataset/
kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset
python main.py
```
*Вторая команда выполняется в случае, если (по какой-то причине) нет папки dataset*

### *ВАЖНО! Последнюю команду придется прописать два раза, потому что в первый выпадает ошибка, с которой мы не совладали...*