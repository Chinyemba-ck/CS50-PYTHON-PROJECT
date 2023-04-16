import pytest
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from project import authentication, read_pdf, split_text

# Authentication tests
def test_authentication_correct():
    assert authentication("user1", "password1") == True

def test_authentication_incorrect_username():
    assert authentication("user4", "password1") == False

def test_authentication_incorrect_password():
    assert authentication("user1", "password4") == False

# read_pdf tests
def test_read_pdf_invalid_file():
    with pytest.raises(Exception):
        read_pdf("/workspaces/85675608/CS50/CS50_PYTHON/project/example.txt")

# split_text tests
def test_split_text():
    pdf_text = "Some random text to put.Should return a list"
    expected_result = ['Some random text to put.Should return a list']
    assert split_text(pdf_text) == expected_result
