from autocatalog.inference.catalog_generator import (
    generate_catalog_output,
)


def test_catalog_output():
    predictions = {
        "gender": "Men",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Tshirts",
        "baseColour": "Black",
        "season": "Summer",
        "usage": "Casual",
    }

    output = generate_catalog_output(predictions)

    assert output["suggested_title"] == (
        "Men Black Casual Tshirts"
    )

    assert "black tshirts" in output["search_tags"]

    assert output["json_export"]["category"] == "Apparel"
    assert output["json_export"]["color"] == "Black"