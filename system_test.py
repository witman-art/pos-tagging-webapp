import io
import time

from docx import Document

from app import app


client = app.test_client()


def run_case(name, func, expect_status=(200,)):
    start = time.perf_counter()
    response = func()
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    passed = response.status_code in expect_status
    print(
        {
            "case": name,
            "status_code": response.status_code,
            "elapsed_ms": elapsed,
            "result": "PASS" if passed else "FAIL",
        }
    )


def main():
    run_case("health_check", lambda: client.get("/api/health"))

    run_case(
        "tag_english_text",
        lambda: client.post(
            "/api/tag",
            json={"text": "The quick brown fox jumps over the lazy dog."},
        ),
    )

    run_case(
        "tag_multiple_sentences",
        lambda: client.post(
            "/api/tag",
            json={
                "text": "The system works well. It can tag English sentences."
            },
        ),
    )

    run_case(
        "empty_text",
        lambda: client.post("/api/tag", json={"text": ""}),
        expect_status=(400,),
    )

    run_case(
        "upload_no_file",
        lambda: client.post("/api/upload"),
        expect_status=(400,),
    )

    run_case(
        "upload_txt",
        lambda: client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"The system works well."), "test.txt")},
            content_type="multipart/form-data",
        ),
    )

    docx_buffer = io.BytesIO()
    doc = Document()
    doc.add_paragraph("The document import works.")
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    run_case(
        "upload_docx",
        lambda: client.post(
            "/api/upload",
            data={"file": (docx_buffer, "test.docx")},
            content_type="multipart/form-data",
        ),
    )

    run_case(
        "unsupported_file",
        lambda: client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"abc"), "test.pdf")},
            content_type="multipart/form-data",
        ),
        expect_status=(400,),
    )

    for fmt in ["html", "csv", "xlsx"]:
        run_case(
            f"export_{fmt}",
            lambda fmt=fmt: client.post(
                "/api/export",
                json={"text": "The system works well.", "format": fmt},
            ),
        )

    run_case(
        "invalid_export_format",
        lambda: client.post(
            "/api/export",
            json={"text": "The system works well.", "format": "pdf"},
        ),
        expect_status=(400,),
    )


if __name__ == "__main__":
    main()
