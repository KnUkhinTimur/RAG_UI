from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import settings
from db import fetch_image_by_url, fetch_rag_context


FONT_PAIRS = {
    "saas": ("Sora", "Inter"),
    "finance": ("Plus Jakarta Sans", "Inter"),
    "ecommerce": ("Space Grotesk", "Manrope"),
    "portfolio": ("Syne", "Manrope"),
    "agency": ("Space Grotesk", "Inter"),
    "healthcare": ("Outfit", "Inter"),
    "default": ("Space Grotesk", "Manrope"),
}


def _extract_html_document(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    match = re.search(r"<!DOCTYPE html>.*</html>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()

    match = re.search(r"<html.*</html>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return "<!DOCTYPE html>\n" + match.group(0).strip()

    return cleaned


def _ensure_google_fonts(html: str, heading_font: str, body_font: str) -> str:
    if "fonts.googleapis.com" in html:
        return html

    font_query = (
        heading_font.replace(" ", "+")
        + ":wght@400;500;600;700;800&family="
        + body_font.replace(" ", "+")
        + ":wght@400;500;600;700;800"
    )
    font_links = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link href="https://fonts.googleapis.com/css2?family={font_query}&display=swap" rel="stylesheet">'
    )

    if "</head>" in html:
        return html.replace("</head>", font_links + "\n</head>", 1)
    return html


def _ensure_viewport(html: str) -> str:
    if 'name="viewport"' in html.lower():
        return html
    if "</head>" in html:
        return html.replace(
            "</head>",
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n</head>',
            1,
        )
    return html


def _ensure_font_css(html: str, heading_font: str, body_font: str) -> str:
    css = (
        ":root{"
        f"--font-heading:'{heading_font}',sans-serif;"
        f"--font-body:'{body_font}',sans-serif;"
        "}"
        "body{font-family:var(--font-body);}"
        "img{max-width:100%;display:block;}"
        "h1,h2,h3,h4,h5,h6,.display-title,.hero-title{font-family:var(--font-heading);}"
    )

    if "<style>" in html:
        return html.replace("<style>", "<style>\n" + css + "\n", 1)
    if "</head>" in html:
        return html.replace("</head>", f"<style>{css}</style>\n</head>", 1)
    return html


def _normalize_generated_html(html: str, heading_font: str, body_font: str) -> str:
    html = _extract_html_document(html)
    html = _ensure_viewport(html)
    html = _ensure_google_fonts(html, heading_font, body_font)
    html = _ensure_font_css(html, heading_font, body_font)

    if "<!DOCTYPE html>" not in html[:40]:
        html = "<!DOCTYPE html>\n" + html
    return html.strip()


def _guess_image_orientation(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return "unknown"
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _extract_keywords(prompt: str) -> List[str]:
    tokens = re.findall(r"\b[\w-]{4,}\b", prompt.lower(), flags=re.UNICODE)
    deduped: List[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped[:8]


def _score_image_for_prompt(image: Dict[str, Any], prompt: str) -> float:
    prompt_lower = prompt.lower()
    description = (image.get("description") or "").lower()
    width = image.get("width") or 0
    height = image.get("height") or 0
    distance = float(image.get("distance") or 0)

    score = max(0.0, 10.0 - distance)
    keywords = _extract_keywords(prompt)
    for keyword in keywords:
        if keyword in description:
            score += 1.2

    if any(word in prompt_lower for word in ("hero", "landing", "header", "banner", "main")) and width > height:
        score += 2.0
    if any(word in prompt_lower for word in ("mobile", "phone", "portrait", "story", "vertical")) and height > width:
        score += 2.0
    if width >= 1200:
        score += 0.6
    if height >= 800:
        score += 0.4

    return round(score, 3)


def _enrich_image_metadata(image: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    item = dict(image)
    item["orientation"] = _guess_image_orientation(item.get("width"), item.get("height"))
    item["fit_score"] = item.get("fit_score", _score_image_for_prompt(item, prompt))
    return item


def _prepare_image_candidates(images: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    scored = [_enrich_image_metadata(image, prompt) for image in images]
    scored.sort(key=lambda item: (-float(item["fit_score"]), float(item.get("distance") or 9999)))
    return scored[:3]


def _merge_selected_image(
    candidates: List[Dict[str, Any]],
    selected_image_url: Optional[str],
    user_prompt: str,
) -> List[Dict[str, Any]]:
    if not selected_image_url:
        return candidates

    for item in candidates:
        if item["image_url"] == selected_image_url:
            return candidates

    selected_row = fetch_image_by_url(selected_image_url)
    if not selected_row:
        return candidates

    selected_item = _enrich_image_metadata(selected_row, user_prompt)
    merged = [selected_item]
    for candidate in candidates:
        if candidate["image_url"] != selected_item["image_url"]:
            merged.append(candidate)
    return merged[:3]


def _choose_font_pair(brand: Dict[str, Any], user_prompt: str) -> Dict[str, str]:
    source = f"{brand.get('domain', '')} {brand.get('tone', '')} {user_prompt}".lower()
    for key, fonts in FONT_PAIRS.items():
        if key != "default" and key in source:
            return {"heading": fonts[0], "body": fonts[1]}
    default_fonts = FONT_PAIRS["default"]
    return {"heading": default_fonts[0], "body": default_fonts[1]}


def _build_layout_guidance(selected_image: Optional[Dict[str, Any]]) -> str:
    if not selected_image:
        return (
            "No image was selected. Build a visually strong page without relying on photography. "
            "Use typography, gradients, cards, stats, icons made with pure CSS or simple shapes, layered panels, and refined spacing. "
            "Do not leave an empty hero hole where an image should have been."
        )

    orientation = selected_image.get("orientation")
    width = selected_image.get("width")
    height = selected_image.get("height")
    ratio = round((width / height), 3) if width and height else None

    if orientation == "portrait":
        return (
            f"The image is portrait with ratio about {ratio}. Build a split hero layout where text and CTA live in one column "
            "and the image lives inside a framed vertical card. Do not crop the subject aggressively. Prefer object-fit: contain "
            "or a card ratio close to the source. Never stretch a portrait image across the full screen width."
        )
    if orientation == "square":
        return (
            f"The image is square with ratio about {ratio}. Use a balanced composition with either a centered media block or a two-column "
            "hero where the image sits in a square card. Preserve the square feeling and avoid turning it into a giant background."
        )
    return (
        f"The image is landscape with ratio about {ratio}. Use a cinematic but restrained hero media panel. Let the image breathe without clipping "
        "important details. Do not make it absurdly tall. Use overlays or separate text panels for readability."
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def get_llm_client() -> OpenAI:
    client_kwargs: Dict[str, Any] = {"api_key": settings.llm_api_key}
    if settings.llm_base_url:
        client_kwargs["base_url"] = settings.llm_base_url
    elif settings.llm_provider.lower() == "groq":
        client_kwargs["base_url"] = "https://api.groq.com/openai/v1"

    return OpenAI(**client_kwargs)


def build_rag_prompt(
    user_prompt: str,
    page: Dict[str, Any],
    brand: Dict[str, Any],
    selected_image: Optional[Dict[str, Any]],
    alternatives: List[Dict[str, Any]],
) -> str:
    alternatives_payload = [
        {
            "image_url": image["image_url"],
            "description": image.get("description"),
            "width": image.get("width"),
            "height": image.get("height"),
            "orientation": image.get("orientation"),
            "fit_score": image.get("fit_score"),
        }
        for image in alternatives
    ]
    fonts = _choose_font_pair(brand, user_prompt)
    layout_guidance = _build_layout_guidance(selected_image)
    ratio_width = selected_image.get("width") if selected_image else 16
    ratio_height = selected_image.get("height") if selected_image else 9

    image_section = f"""
SELECTED IMAGE TO USE:
URL: {selected_image["image_url"]}
Description: {selected_image.get("description")}
Width: {selected_image.get("width")}
Height: {selected_image.get("height")}
Orientation: {selected_image.get("orientation")}
""".strip() if selected_image else """
SELECTED IMAGE TO USE:
No image was selected for this generation.
""".strip()

    image_rules = f"""
Image rules:
- Respect the real width and height of the selected image.
- Never distort the image.
- Use CSS that preserves the image ratio in a container shaped for that ratio.
- Use `aspect-ratio: {ratio_width} / {ratio_height}` on the main image wrapper when appropriate.
- Default to a contained card, framed panel, or restrained hero media block instead of making the image cover the entire screen.
- Prefer `object-fit: contain` for portrait and square imagery unless a matching card ratio makes `cover` safe.
- If using the image as a background, cap the section height and use overlays instead of stretching the media.
- If text overlays the image, ensure strong readability with scrims, gradients, or separate text panels.
- The selected image must actually appear in the HTML.
- Keep image sections elegant and bounded, not oversized and not stretched.
""".strip() if selected_image else """
Image rules:
- No image should be used in this page.
- Build the hero and sections without `<img>` tags or photo backgrounds.
- Replace image impact with typography, gradients, abstract shapes, glass cards, stat blocks, and layout rhythm.
- Keep the result rich and premium even without photography.
""".strip()

    implementation_hint = (
        f"- A good pattern is `.hero-media {{ aspect-ratio: {ratio_width} / {ratio_height}; overflow: hidden; border-radius: 28px; max-height: min(72vh, 760px); }}` with a nested `img`."
        if selected_image
        else "- A good pattern is a hero with layered gradient background, one strong heading, one support paragraph, stat cards, and a polished CTA cluster."
    )

    return f"""
You are an elite web designer and senior frontend engineer creating premium marketing-quality pages.

Your job is to produce a polished single-file HTML page that looks intentional, modern, and art directed.
The result must feel like a real designer made it, not a generic AI wireframe.

USER REQUEST:
{user_prompt}

RAG PAGE STRUCTURE:
Page type: {page["page_type"]}
Description: {page["description"]}
UI elements: {json.dumps(page["ui_elements"], ensure_ascii=False)}

RAG BRAND STYLE:
Domain: {brand["domain"]}
Tone: {brand["tone"]}
Style direction: {brand["style_description"]}

{image_section}

IMAGE USAGE GUIDANCE:
{layout_guidance}

TOP IMAGE OPTIONS FOR CONTEXT:
{json.dumps(alternatives_payload, ensure_ascii=False, indent=2)}

TYPOGRAPHY DIRECTION:
- Heading font: {fonts["heading"]}
- Body font: {fonts["body"]}
- Load fonts from Google Fonts in the HTML head.

Quality bar:
- The page must look visually premium and production-like.
- Do not make text and background same color.
- Avoid ugly gradients, random neon, and generic bootstrap-looking sections.
- Use a strong spacing system, restrained color palette, clear hierarchy, and one coherent visual concept.
- Design for desktop first, then make it adapt beautifully to mobile.
- The hero must be impressive and not cramped.
- Every section should have a reason to exist.

{image_rules}

Implementation rules:
- Return only valid complete HTML.
- Include <!DOCTYPE html>, <html>, <head>, <body>.
- Put all CSS inside a single <style> tag.
- Do not use external CSS libraries.
- You may use Google Fonts links.
- No markdown fences.
- No explanations before or after the HTML.
- Use semantic HTML tags.
- Make sure the page feels finished, not like a mockup.

Suggested structure:
- Hero
- Supporting proof or metrics
- One or two feature/content sections
- CTA/footer

Implementation hint:
{implementation_hint}

Output only the final HTML document.
""".strip()


def suggest_image_candidates(user_prompt: str) -> Dict[str, Any]:
    embedding = get_embedding_model().encode(user_prompt).tolist()
    context = fetch_rag_context(embedding, limit_images=9)
    candidates = _prepare_image_candidates(context["images"], user_prompt)

    return {
        "page": context["page"],
        "brand": context["brand"],
        "images": candidates,
    }


def generate_html(user_prompt: str, selected_image_url: Optional[str] = None) -> Dict[str, Any]:
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set. Add it to your .env file.")

    retrieval = suggest_image_candidates(user_prompt)
    image_candidates = _merge_selected_image(retrieval["images"], selected_image_url, user_prompt)

    selected_image = None
    if selected_image_url:
        selected_image = next(
            (image for image in image_candidates if image["image_url"] == selected_image_url),
            None,
        )

    rag_prompt = build_rag_prompt(
        user_prompt=user_prompt,
        page=retrieval["page"],
        brand=retrieval["brand"],
        selected_image=selected_image,
        alternatives=image_candidates,
    )

    system_prompt = """
You are a world-class frontend engineer and digital art director.
Produce one complete HTML file with embedded CSS.
Make the result elegant, responsive, and visually strong.
Do not output markdown or commentary.
""".strip()

    response = get_llm_client().chat.completions.create(
        model=settings.llm_model,
        temperature=0.35,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt},
        ],
    )
    raw_html = response.choices[0].message.content or ""
    fonts = _choose_font_pair(retrieval["brand"], user_prompt)
    generated_html = _normalize_generated_html(raw_html, fonts["heading"], fonts["body"])
    if not generated_html or "<body" not in generated_html.lower():
        raise RuntimeError("Model returned invalid HTML.")

    return {
        "html": generated_html,
        "rag_prompt": rag_prompt,
        "page": retrieval["page"],
        "brand": retrieval["brand"],
        "selected_image": selected_image,
        "image_candidates": image_candidates,
    }
