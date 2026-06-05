"""
invoke.py - RAG 调用入口（优化版）
"""
import time
from pathlib import Path
from tqdm import tqdm
from rag_engine import rag
from document_loader import DocumentLoader, load_folder


def main():
    loader = DocumentLoader()
    pdf_dir = Path("./documents/pdfs/地理与旅游")

    # 1. 扫描 PDF 类型（简洁输出）
    text_pdfs = []
    scan_pdfs = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"扫描 {len(pdf_files)} 个 PDF...")

    for pdf_path in pdf_files:
        with pdf_path.open("rb") as f:
            import pdfplumber
            with pdfplumber.open(f) as pdf:
                text_count = sum(1 for p in pdf.pages[:3] if len((p.extract_text() or "").strip()) > 100)
                if text_count >= 2:
                    text_pdfs.append(pdf_path)
                else:
                    scan_pdfs.append(pdf_path)

    print(f"  文字版: {len(text_pdfs)} 个")
    print(f"  扫描版: {len(scan_pdfs)} 个")

    # 2. 加载文字版（带进度条）
    if text_pdfs:
        print(f"\n加载文字版 PDF...")
        for pdf_path in tqdm(text_pdfs, desc="处理中", unit="个"):
            text = loader.load(str(pdf_path))
            if text.strip():
                rag.add_text_chunks(
                    text=text,
                    source=pdf_path.name,
                    chunk_size=800,
                    chunk_overlap=80,
                )
        rag._save()  # 统一保存一次
        print(f"文字版加载完成")

    # 3. 扫描版提示
    if scan_pdfs:
        print(f"\n{len(scan_pdfs)} 个扫描版待 OCR：")
        for p in scan_pdfs:
            print(f"    - {p.name}")

    # 4. 最终统计
    stats = rag.get_stats()
    print(f"\n{'='*40}")
    print(f"知识库共 {stats['total_documents']} 条文档")
    print(f"来源分布: {stats['sources']}")
    print(f"{'='*40}")

    # 5. 交互查询
    while True:
        query = input("\n查询: ").strip()
        if query.lower() in ("quit", "q", ""):
            break

        results = rag.search(query, top_k=3, score_threshold=0.5)

        if not results:
            print("未找到相关知识。")
            continue

        for r in results:
            print(f"\n相似度: {r['score']:.4f} | 来源: {r['source']}")
            print(f"内容: {r['content'][:150]}...")
            print("-" * 40)


if __name__ == "__main__":
    main()