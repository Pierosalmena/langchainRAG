from pathlib import Path
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
import re

# Optional: install gpt4all via pip if not already installed
# pip install gpt4all

from gpt4all import GPT4All

def enrich_xml_with_pdf_articles_gpt4all(pdf_path_str, xml_path_str, output_path_str, model_name="gpt4all-lora-quantized"):
    """
    Enriches an XML order file by mapping items to article positions from a PDF.
    Uses direct lookup first; if no match, falls back to GPT4All local model for fuzzy matching.
    """
    pdf_path = Path(pdf_path_str)
    xml_path = Path(xml_path_str)
    output_path = Path(output_path_str)

    # Load PDF
    doc = fitz.open(pdf_path)

    # Load XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Step 1: Extract article positions and descriptions from PDF
    article_pattern = re.compile(r"(\d{2}\.\d)\s+(.*?)\\n", re.DOTALL)
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text()

    # Create mapping from article number to title+description
    article_map = {}
    matches = list(article_pattern.finditer(pdf_text))
    for i, match in enumerate(matches):
        num = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(pdf_text)
        desc = pdf_text[start:end].strip()
        # Flatten cross-references (e.g., "Wie Position XX")
        desc = re.sub(r"Wie Position (\\d{2}\\.\\d)", lambda m: f"(refers to {m.group(1)})", desc)
        article_map[num] = {"title": title, "description": desc}

    # Initialize GPT4All model for fallback
    model = GPT4All(model_name)

    # Step 2: Match XML items to PDF articles
    for item in root.findall(".//item"):
        commission = item.find("commission")
        sku = item.find("sku").text if item.find("sku") is not None else ""
        name = item.find("name").text if item.find("name") is not None else ""
        text = item.find("text").text if item.find("text") is not None else ""
        matched = False

        if commission is not None and "LV-POS." in commission.text:
            art_num = commission.text.split("LV-POS.")[1].strip()
            if art_num in article_map:
                # Direct match found
                data = article_map[art_num]
                matched = True
            else:
                # Fallback: use GPT4All to choose among article numbers
                prompt = f"Order item name: {name}\\nDescription: {text}\\nAvailable articles:\\n"
                for num, info in article_map.items():
                    prompt += f"- {num}: {info['title']}\\n"
                prompt += "Which article number matches this item best? Answer with the number only."
                resp = model.generate(prompt, max_tokens=16)
                guessed = resp.strip().split()[0]
                data = article_map.get(guessed, {"title": "", "description": ""})
                art_num = guessed
                matched = True

            # Append <article> element
            article_elem = ET.SubElement(item, "article")
            article_elem.set("number", art_num)
            title_elem = ET.SubElement(article_elem, "title")
            title_elem.text = data["title"]
            desc_elem = ET.SubElement(article_elem, "description")
            desc_elem.text = data["description"]

    # Step 3: Save enriched XML
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Enriched XML saved to: {output_path}")

# Example usage:
# enrich_xml_with_pdf_articles_gpt4all("service-specification.pdf", "output.xml", "enriched_output_with_gpt4all.xml")

