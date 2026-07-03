def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = " ".join(value.split())

    return value


def generate_title(predicted_labels):
    parts = []
    
    gender = normalize_text(predicted_labels.get("gender"))
    color = normalize_text(predicted_labels.get("baseColour"))
    usage = normalize_text(predicted_labels.get("usage"))
    article_type = normalize_text(predicted_labels.get("articleType"))

    for value in [gender, color, usage, article_type]:
        if value:
            parts.append(value)

    return " ".join(parts)


def generate_search_tags(predicted_labels):
    tags = []
    unique_tags = []
    
    gender = normalize_text(predicted_labels.get("gender")).lower()
    master_category = normalize_text(predicted_labels.get("masterCategory")).lower()
    sub_category = normalize_text(predicted_labels.get("subCategory")).lower()
    article_type = normalize_text(predicted_labels.get("articleType")).lower()
    color = normalize_text(predicted_labels.get("baseColour")).lower()
    season = normalize_text(predicted_labels.get("season")).lower()
    usage = normalize_text(predicted_labels.get("usage")).lower()

    if gender and article_type:
        tags.append(f"{gender} {article_type}")

    if color and article_type:
        tags.append(f"{color} {article_type}")

    if usage and sub_category:
        tags.append(f"{usage} {sub_category}")

    if season and master_category:
        tags.append(f"{season} {master_category}")

    if gender and usage:
        tags.append(f"{gender} {usage} wear")

    if color and usage:
        tags.append(f"{color} {usage} fashion")

    if sub_category:
        tags.append(sub_category)

    if article_type:
        tags.append(article_type)

    for tag in tags:
        tag = " ".join(tag.split())

        if tag and tag not in unique_tags:
            unique_tags.append(tag)

    return unique_tags


def generate_catalog_output(predicted_labels):
    suggested_title = generate_title(predicted_labels)
    search_tags = generate_search_tags(predicted_labels)

    return {
        "suggested_title": suggested_title,
        "search_tags": search_tags,
        "json_export": {
            "gender": predicted_labels.get("gender"),
            "category": predicted_labels.get("masterCategory"),
            "subcategory": predicted_labels.get("subCategory"),
            "article_type": predicted_labels.get("articleType"),
            "color": predicted_labels.get("baseColour"),
            "season": predicted_labels.get("season"),
            "usage": predicted_labels.get("usage"),
            "title": suggested_title,
            "tags": search_tags,
        },
    }