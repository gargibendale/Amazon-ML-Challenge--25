import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk


STOP_WORDS = set(stopwords.words('english'))
PUNCT = string.punctuation

def clean_text(text):
    """Lowercase, remove punctuation and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS and not word.isnumeric()]
    return " ".join(tokens)

def parse_catalog_content(text):
    result = {
        "item_name": None,
        "brand_name": None,
        "bullet_points": [],
        "product_description": None,
        "value": None,
        "unit": None
    }

    if pd.isna(text):
        return result

    # Split by newlines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # ITEM NAME
        if line.lower().startswith("item name:"):
            item_name = line.split(":", 1)[1].strip()
            result["item_name"] = item_name

            # Extract brand (first word after item name)
            brand_match = re.match(r"([A-Za-z0-9'’]+)", item_name)
            if brand_match:
                result["brand_name"] = brand_match.group(1)

        # BULLET POINTS
        elif line.lower().startswith("bullet point"):
            bullet_text = line.split(":", 1)[1].strip()
            if bullet_text:
                cleaned = clean_text(bullet_text)
                if cleaned:
                    result["bullet_points"].append(cleaned)

        # PRODUCT DESCRIPTION
        elif line.lower().startswith("product description:"):
            desc = line.split(":", 1)[1].strip()
            result["product_description"] = clean_text(desc)

        # VALUE AND UNIT
        elif line.lower().startswith("value:"):
            value = re.findall(r"\d+\.?\d*", line)
            result["value"] = float(value[0]) if value else None
        elif line.lower().startswith("unit:"):
            unit = line.split(":", 1)[1].strip()
            result["unit"] = unit

    return result


# ---------- MAIN SCRIPT ----------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../dataset/train.csv")

    # Apply parsing function
    parsed_df = df["catalog_content"].apply(parse_catalog_content).apply(pd.Series)

    # Merge parsed data back to main dataframe
    df = pd.concat([df, parsed_df], axis=1)

    # Save processed file
    df.to_csv("../dataset/processed/cleaned_train.csv", index=False)

    print("✅ Preprocessing completed. Sample output:")
    print(df.head(3))
