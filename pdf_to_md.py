import pdfplumber
import re
import os
from PIL import Image, ImageOps
import io

def extract_tables_with_pdfplumber(pdf_path):
    """
    PDF 파일에서 pdfplumber를 사용해 테이블을 추출하는 함수.

    :param pdf_path: PDF 파일 경로
    :return: 추출된 테이블 리스트
    """
    tables = []

    # pdfplumber로 PDF 열기
    with pdfplumber.open(pdf_path) as pdf:
        # 각 페이지에서 테이블 추출
        for page_num, page in enumerate(pdf.pages):
            print(f"페이지 {page_num + 1}에서 테이블 추출 중...")
            tables_on_page = page.extract_tables()

            if tables_on_page:
                for table in tables_on_page:
                    tables.append(table)

    return tables


def print_tables(tables):
    """
    추출된 테이블 출력 함수.
    """
    if tables:
        for i, table in enumerate(tables, start=1):
            print(f"\n테이블 {i}:")
            for row in table:
                print("\t".join(str(cell) if cell else "" for cell in row))
    else:
        print("추출된 테이블이 없습니다.")


def convert_pdf_to_markdown(pdf_path):
    """
    PDF 파일을 Markdown으로 변환하는 함수.

    :param pdf_path: PDF 파일 경로
    :return: Markdown 문자열
    """
    markdown_content = ""
    image_folder = "images"
    os.makedirs(image_folder, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"페이지 {page_num} 변환 중...")
            
            # 페이지 번호를 대제목으로 추가
            markdown_content += f"# 페이지 {page_num}\n\n"
            
            # 텍스트 추출 및 변환
            text = page.extract_text()
            markdown_content += convert_text_to_markdown(text)
            
            # 테이블 추출 및 변환
            tables = page.extract_tables()
            for table in tables:
                markdown_content += convert_table_to_markdown(table)
            
            # 이미지 추출 및 변환
            images = page.images
            for i, img in enumerate(images):
                try:
                    image = Image.open(io.BytesIO(img['stream'].get_data()))
                    if image.mode == 'CMYK':
                        image = ImageOps.invert(image.convert('RGB'))
                    else:
                        image = image.convert('RGB')
                    image_filename = f"page_{page_num}_image_{i+1}.png"
                    image_path = os.path.join(image_folder, image_filename)
                    image.save(image_path)
                    markdown_content += f"![페이지 {page_num} 이미지 {i+1}]({image_path})\n\n"
                except Exception as e:
                    print(f"페이지 {page_num}의 이미지 {i+1} 처리 중 오류 발생: {str(e)}")
                    continue
            
            markdown_content += "\n\n"  # 페이지 구분

    return markdown_content

def convert_text_to_markdown(text):
    """
    추출된 텍스트를 Markdown 형식으로 변환하는 함수.
    """
    # 제목 변환 (예: 큰 글씨나 굵은 글씨를 #으로 변환)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^[A-Z\s]+$', line.strip()):  # 대문자로만 이루어진 줄을 제목으로 간주
            lines[i] = f"# {line}\n"
    
    return '\n'.join(lines)

def convert_table_to_markdown(table):
    """
    추출된 테이블을 Markdown 테이블 형식으로 변환하는 함수.
    """
    if not table or not table[0]:
        return "\n빈 테이블\n"

    column_widths = [max(len(str(cell)) for cell in column) for column in zip(*table)]
    
    markdown_rows = []
    for i, row in enumerate(table):
        padded_cells = [str(cell).ljust(width) if cell else ' ' * width 
                        for cell, width in zip(row, column_widths)]
        markdown_rows.append(f"| {' | '.join(padded_cells)} |")
        
        if i == 0:
            separator = ['-' * width for width in column_widths]
            markdown_rows.append(f"| {' | '.join(separator)} |")

    return "\n" + "\n".join(markdown_rows) + "\n\n"

if __name__ == "__main__":
    # PDF 파일 경로
    pdf_file_path = "test_set/삼정 수소생산.pdf"

    # PDF를 Markdown으로 변환
    markdown_content = convert_pdf_to_markdown(pdf_file_path)

    # Markdown 파일로 저장
    output_file = pdf_file_path.rsplit('.', 1)[0] + '.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"변환된 Markdown 파일이 {output_file}에 저장되었습니다.")

    # 기존 테이블 추출 및 출력 코드 (선택적)
    extracted_tables = extract_tables_with_pdfplumber(pdf_file_path)
    print_tables(extracted_tables)
