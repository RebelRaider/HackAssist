import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union

def search_results(connection, table_name: str, vector: list[float], limit: int = 5):
    """
    Поиск результатов похожих векторов в базе данных.

    Parameters:
    - connection (Connection): Соединение с базой данных.
    - table_name (str): Название таблицы, содержащей вектора и другие данные.
    - vector (List[float]): Вектор для сравнения.
    - limit (int): Максимальное количество результатов.

    Returns:
    - List[dict]: Список результатов с наименованием, URL, датой, номером, текстом и расстоянием.

    Examples:
    >>> connection = Connection(...)
    >>> vector = [0.1, 0.2, 0.3]
    >>> results = search_results(connection, 'my_table', vector, limit=5)
    """
    res = []
    # Инициализируем список результатов
    vector = ",".join([str(float(i)) for i in vector])
    # Выполняем запрос к базе данных
    with connection.query(
        f"""SELECT Text, cosineDistance(({vector}), Embedding) as score FROM {table_name} ORDER BY score ASC LIMIT {limit+500}"""
    ).rows_stream as stream:
        for item in stream:
            text, score = item

            # Добавляем результат в список
            res.append(
                {
                    "text": text,
                    "distance": score,
                }
            )

    # Возвращаем первые limit результатов
    res = [item for item in res if len(item["text"]) > 100]
    return res[:limit]


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Выполняет усреднение токенов входной последовательности на основе attention mask.

    Parameters:
    - model_output (tuple): Выход модели, включающий токенов эмбеддинги и другие данные.
    - attention_mask (torch.Tensor): Маска внимания для указания значимости токенов.

    Returns:
    - torch.Tensor: Усредненный эмбеддинг.

    Examples:
    >>> embeddings = model_output[0]
    >>> mask = torch.tensor([[1, 1, 1, 0, 0]])
    >>> pooled_embedding = mean_pooling((embeddings,), mask)
    """
    # Получаем эмбеддинги токенов из выхода модели
    token_embeddings = model_output[0]

    # Расширяем маску внимания для умножения с эмбеддингами
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    # Умножаем каждый токен на его маску и суммируем
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    # Суммируем маски токенов и обрезаем значения, чтобы избежать деления на ноль
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Вычисляем усредненный эмбеддинг
    return sum_embeddings / sum_mask


def txt2embeddings(
    text: Union[str, List[str]], tokenizer, model, device: str = "cpu"
) -> torch.Tensor:
    """
    Преобразует текст в его векторное представление с использованием модели transformer.

    Parameters:
    - text (str): Текст для преобразования в векторное представление.
    - tokenizer: Токенизатор для предобработки текста.
    - model: Модель transformer для преобразования токенов в вектора.
    - device (str): Устройство для вычислений (cpu или cuda).

    Returns:
    - torch.Tensor: Векторное представление текста.

    Examples:
    >>> text = "Пример текста"
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    >>> model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    >>> embeddings = txt2embeddings(text, tokenizer, model, device="cuda")
    """
    # Кодируем входной текст с помощью токенизатора
    if isinstance(text, str):
        text = [text]
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    # Перемещаем закодированный ввод на указанное устройство
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Получаем выход модели для закодированного ввода
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Преобразуем выход модели в векторное представление текста
    return mean_pooling(model_output, encoded_input["attention_mask"])


def load_models(model: str, device: str = "cpu", torch_dtype: str = "auto") -> tuple:
    """
    Загружает токенизатор и модель для указанной предобученной модели.

    Parameters:
    - model (str): Название предобученной модели, поддерживаемой библиотекой transformers.

    Returns:
    - tuple: Кортеж из токенизатора и модели.

    Examples:
    >>> tokenizer, model = load_models("ai-forever/sbert_large_nlu_ru")
    """
    # Загружаем токенизатор для модели
    tokenizer = AutoTokenizer.from_pretrained(
        model, device_map=device, torch_dtype=torch_dtype
    )

    # Загружаем модель
    model = AutoModel.from_pretrained(model, device_map=device, torch_dtype=torch_dtype)

    return tokenizer, model


from typing import List, Dict, Any
from SETTINGS import ROLE_TOKENS, LINEBREAK_TOKEN, SYSTEM_PROMPT


def get_message_tokens(model: Any, role: str, content: str) -> List[int]:
    """
    Создает токены для сообщения с учетом роли и содержания.

    Параметры:
    - model (Any): Модель токенизатора.
    - role (str): Роль сообщения.
    - content (str): Содержание сообщения.

    Возвращает:
    List[int]: Список токенов сообщения.

    Пример использования:
    ```python
    model = SomeTokenizer()
    role = "user"
    content = "Hello, world!"
    message_tokens = get_message_tokens(model, role, content)
    ```

    Подробности:
    - Функция токенизирует содержание сообщения с учетом роли и вставляет соответствующие токены.
    - В конце сообщения добавляется токен окончания строки.
    """
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens