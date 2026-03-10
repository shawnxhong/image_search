from fastapi.testclient import TestClient

from image_search_app.api.main import app


client = TestClient(app)


def test_index_route_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Agentic Image Search" in response.text


def test_index_html_route_serves_html():
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_static_css_route_serves_asset():
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
