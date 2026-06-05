"""
document_loader.py - 文档加载器
支持 txt, pdf(文本/OCR), docx
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional
import pdfplumber
from docx import Document

# OCR 相关
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False


class DocumentLoader:
    """统一文档加载接口，支持扫描版 PDF OCR"""

    def __init__(self):
        self.ocr = None
        if PADDLE_OCR_AVAILABLE:
            print("正在初始化 PaddleOCR（首次使用需下载模型，请等待）...")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
            )
            print("PaddleOCR 初始化完成！")

    @staticmethod
    def load_txt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_pdf(self, path: str) -> str:
        """智能 PDF 加载：先提取文本，文本太少则启用 OCR"""
        text = ""
        ocr_pages = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if len(page_text.strip()) < 50:
                    ocr_pages.append(i)
                else:
                    text += page_text + "\n"

        if ocr_pages and self.ocr:
            print(f"  检测到 {len(ocr_pages)} 页扫描内容，启用 OCR...")
            text += self._ocr_pdf_pages(path, ocr_pages)
        elif ocr_pages and not self.ocr:
            print(f"  警告: {len(ocr_pages)} 页需要 OCR，但未安装 PaddleOCR")

        return text

    def _ocr_pdf_pages(self, path: str, page_indices: List[int]) -> str:
        """对指定页面进行 OCR"""
        text = ""
        temp_dir = Path("./temp_ocr")
        temp_dir.mkdir(exist_ok=True)

        with pdfplumber.open(path) as pdf:
            for i in page_indices:
                page = pdf.pages[i]
                im = page.to_image(resolution=200)
                temp_path = temp_dir / f"page_{i}.png"
                im.save(str(temp_path))

                result = self.ocr.ocr(str(temp_path))

                if result and result[0]:
                    for line in result[0]:
                        text += line[1][0] + "\n"

                print(f"    OCR 完成第 {i + 1} 页")

        shutil.rmtree(temp_dir, ignore_errors=True)
        return text

    @staticmethod
    def load_docx(path: str) -> str:
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def load(self, path: str) -> str:
        """根据后缀自动选择加载方式"""
        ext = Path(path).suffix.lower()
        if ext == ".txt":
            return self.load_txt(path)
        elif ext == ".pdf":
            return self.load_pdf(path)
        elif ext == ".docx":
            return self.load_docx(path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")


def load_folder(
        folder_path: str,
        rag_instance,
        loader: Optional[DocumentLoader] = None
) -> List[str]:
    """
    递归加载文件夹及子目录内所有支持的文档到 RAG
    """
    if loader is None:
        loader = DocumentLoader()

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    loaded_files = []
    supported_exts = {".txt", ".pdf", ".docx"}

    for file_path in sorted(folder.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            try:
                print(f"正在加载: {file_path.relative_to(folder)}...")
                text = loader.load(str(file_path))

                if not text.strip():
                    print(f"  ⚠ 未提取到内容，跳过")
                    continue

                source = str(file_path.relative_to(folder))

                rag_instance.add_text_chunks(
                    text=text,
                    chunk_size=500,
                    chunk_overlap=50,
                    source=source
                )
                loaded_files.append(source)
                print(f"  ✓ 完成，提取 {len(text)} 字符")

            except Exception as e:
                print(f"  ✗ 失败: {e}")

    return loaded_files