import os
import re
from filters import get_length_filter, title_filter

"""
pdf -> txt
txt -> doccano
"""


def delete_page_num(text):
    """
        删除页码
    :param text:
    :return:
    """
    page_nums = [
        r'\n- \d+ -( *?)\n',
        r'\n— \d+ —( *?)+\n',
        r'\n\d+( *?)\n',
        r'\nI+( *?)\n',
    ]

    patterns = [re.compile(pattern) for pattern in page_nums]
    for pattern in patterns:
        text = pattern.sub('', text)
    return text


def trans_pdf_text(input_file, output_file, is_delete_page=True):
    """
        把pdf文件转为txt，删除页码，保存到output_file
    :param input_file:
    :param output_file:
    :param is_delete_page:
    :return:
    """
    import fitz
    pdf_file = fitz.open(input_file)  # pdf_path是PDF文件的路径

    res = []
    for i in range(len(pdf_file)):
        page = pdf_file.load_page(i)
        res.append(page.get_text())

    text = ''.join(res)
    if is_delete_page:
        text = delete_page_num(text)
    with open(output_file, 'w') as f:
        f.write(text)


def trans_folder_pdf2txt(prov, output_folder='pdf2txt'):
    """
        把某目录下pdf文件转为txt，方便预览和手动修改
    :return:
    """
    filenames = list(filter(
        lambda x: x.endswith('.pdf'),
        os.listdir(prov)
    ))
    if not os.path.exists(p := os.path.join(output_folder, prov)):
        os.mkdir(p)

    for filename in filenames:
        filename = os.path.join(prov, filename)
        output_file = os.path.join(output_folder, filename.replace('.pdf', '.txt'))
        trans_pdf_text(
            filename,
            output_file
        )


def cut_txt2sents(input_file, output_file, *args):
    """
        这部分处理由pdf转的txt文件，再将txt文本按照句号。切分
        由于pdf转的txt文件，其文件内容很乱，需要进行一些处理
        * args: 过滤器
            针对句子的过滤器
    """
    # 删除  
    delete_list = [
        '\xa0', '\t', '\u3000',
        ' ', '﻿', ' ', ' ', '​',
        '目\n录\n', '\n'
    ]

    if input_file.endswith('.txt'):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        for char in delete_list:
            text = text.replace(char, '')

        text = text.replace(';', '。')
        text = text.replace('；', '。')

        ## 本来按照\n切分最好，但是pdf转txt后，其中包含很多的\n，所以无法使用\n提前切分
        # texts = text.split('\n')

        # for text in texts:
        #     data.extend(text.split('。'))
        data = text.split('。')
        # 过滤器
        for arg in args:
            data = filter(arg, data)

        with open(output_file, 'w') as f:
            f.write('\n'.join(data))


def trans_folder_txt2doccano(input_folder, output_folder, *filter_funcs):
    """
        把某目录下的txt文件转为doccano格式
        针对一整个文件夹内的文件，批量操作)
    :return:
    """
    filenames = list(filter(
        lambda x: x.endswith('.txt'),
        os.listdir(input_folder)
    ))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for filename in filenames:
        cut_txt2sents(
            os.path.join(input_folder, filename),
            os.path.join(output_folder, filename),
            *filter_funcs
        )


if __name__ == '__main__':
    # pdf -> txt
    pdf_txt_folder = 'pdf2txt'
    prov = '湖北省'
    trans_folder_pdf2txt(prov, pdf_txt_folder)

    trans_folder_txt2doccano(
        os.path.join(pdf_txt_folder, prov),
        os.path.join('doccano', prov),
        get_length_filter(8, 200),
        title_filter
    )

    trans_folder_txt2doccano(
        prov, f'doccano/{prov}',
        get_length_filter(8, 200)
    )
