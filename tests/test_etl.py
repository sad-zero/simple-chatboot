import json
import os
from simple_chatbot.etl import Extractor


def test_should_extract_pdf_to_normalized_json():
    # given
    src_path = "resources/references/book.pdf"
    raw_dest_path = "resources/tests/references/book.json"
    dest_path = "resources/tests/references/normalized_book.json"
    extractor = Extractor()
    # when
    result = extractor.parse(pdf_path=src_path, raw_dest_path=raw_dest_path, dest_path=dest_path)

    # then
    assert os.path.exists(raw_dest_path)
    assert os.path.exists(dest_path)

    with open(dest_path, "r") as fd:
        actual = json.load(fd)
    assert actual == result
