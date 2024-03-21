def contains_digit_filters(sentence):
    """
    判断句子中是否包含数字
    """
    for char in sentence:
        if char.isdigit():
            return True
    return False


def get_length_filter(bottom_len=8, top_len=1e3):
    """
        文本长度过滤器，返回一个过滤器，
        用于筛选出文本长度在bottom_len与top_len之间的句子
    """
    def _length_filter(text):
        if bottom_len <= len(text) <= top_len:
            return True
        return False

    return _length_filter


def catalog_filter(text):
    """
        过滤章节，识别到章节则返回False，删除掉
    :param text:
    :return:
    """
    text = text.strip()
    head = text[:5]
    if '第' == head[0]:
        if '章' in head or '节' in head or '篇' in head:
            return False
    return True


def title_filter(text):
    if len(text) <= 45:
        if '国民经济和社会发展' in text and '五年规划' in text:
            return False
    return True