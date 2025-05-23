import streamlit as st
import spacy
from spacy import displacy
from googletrans import Translator
import nepali_roman as nr
from transformers import pipeline
import spacy.cli
from spacy.util import is_package

# Initialize translator
translator = Translator()

if not is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
    
if not is_package("fr_core_news_sm"):
    spacy.cli.download("fr_core_news_sm")

if not is_package("ja_core_news_sm"):
    spacy.cli.download("ja_core_news_sm")

# Language configurations
LANGUAGES = {
    "English": "en",
    "Nepali": "ne",
    "French": "fr", 
    "Japanese": "ja"
}

TRANSLATIONS = {
    "en": {
        "title": "🌍 Climate Impact Analysis",
        "input_label": "Paste news article:",
        "analyze_btn": "Analyze",
        "severity": "Severity Level",
        "confidence": "Confidence",
        "casualties": "Casualties",
        "impact": "Impact Assessment",
        "entities": "Key Entities",
        "actions": "Impact Actions",
        "technical": "Technical Details",
        "major": "🚨 Major Disaster",
        "significant": "⚠️ Significant Impact",
        "moderate": "✅ Moderate Impact",
        "deaths": "Confirmed Deaths",
        "missing": "Missing Persons",
    },
    "ne": {
        "title": "🌍 जलवायु प्रभाव विश्लेषण",
        "input_label": "समाचार टाँस्नुहोस्:",
        "analyze_btn": "विश्लेषण गर्नुहोस्",
        "severity": "गम्भीरता स्तर",
        "confidence": "विश्वास",
        "casualties": "मृत्यु संख्या",
        "impact": "प्रभाव मूल्यांकन",
        "entities": "मुख्य संस्थाहरू",
        "actions": "प्रभावी कार्यहरू",
        "technical": "प्राविधिक विवरण",
        "major": "🚨 ठूलो प्रकोप",
        "significant": "⚠️ महत्त्वपूर्ण प्रभाव",
        "moderate": "✅ मध्यम प्रभाव",
        "deaths": "पुष्टि भएको मृत्यु",
        "missing": "हराएका व्यक्ति"
    },
    "fr": {
        "title": "🌍 Analyse d'Impact Climatique",
        "input_label": "Coller l'article de presse:",
        "analyze_btn": "Analyser",
        "severity": "Niveau de Gravité",
        "confidence": "Confiance",
        "casualties": "Victimes",
        "impact": "Évaluation d'Impact",
        "entities": "Entités Clés",
        "actions": "Actions d'Impact",
        "technical": "Détails Techniques",
        "major": "🚨 Catastrophe Majeure",
        "significant": "⚠️ Impact Important",
        "moderate": "✅ Impact Modéré"
    },
    "ja": {
        "title": "🌍 気候影響分析",
        "input_label": "ニュース記事を貼り付け:",
        "analyze_btn": "分析する",
        "severity": "深刻度レベル",
        "confidence": "信頼度", 
        "casualties": "死者数",
        "impact": "影響評価",
        "entities": "主要エンティティ",
        "actions": "影響アクション",
        "technical": "技術的詳細",
        "major": "🚨 大規模災害",
        "significant": "⚠️ 重大な影響",
        "moderate": "✅ 中等度の影響"
    }
}

# Disaster Severity Thresholds
SEVERITY_LEVELS = {
    "Catastrophic": {"min_deaths": 50, "color": "red"},
    "Severe": {"min_deaths": 20, "color": "orange"},
    "Major": {"min_deaths": 5, "color": "yellow"},
    "Moderate": {"min_deaths": 1, "color": "blue"}
}

# Load models
CLIMATE_MODELS = {
    "en": pipeline("text-classification", model="climatebert/environmental-claims"),
    "multilingual": pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
}

nlp_models = {
    "en": spacy.load("en_core_web_sm"),
    "fr": spacy.load("fr_core_news_sm"),
    "ja": spacy.load("ja_core_news_sm")
}

#translator = Translator()

def analyze_text(text, lang):
    """Advanced multilingual analysis (fixed)"""
    # Translate non-English text
    if lang != "en":
        # translated = translator.translate(text, src=lang, dest='en').text
        translated =  translate_text(text, lang)
    else:
        translated = text
    
    # Use appropriate model
    model = CLIMATE_MODELS["en"] if lang == "en" else CLIMATE_MODELS["multilingual"]
    severity = model(translated)[0]
    
    # Entity analysis
    nlp = nlp_models.get(lang, nlp_models["en"])
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Impact analysis
    impact_verbs = {
        "en": ["trigger", "cause", "destroy", "flood", "displace"],
        "ne": ["ट्रिगर", "कारण", "नष्ट", "बाढी", "विस्थापित"],
        "fr": ["déclencher", "causer", "détruire", "inonder", "déplacer"],
        "ja": ["引き起こす", "原因", "破壊", "洪水", "移動"]
    }
    verbs = impact_verbs.get(lang, impact_verbs["en"])
    impact_count = sum(1 for token in doc if token.lemma_.lower() in verbs)
    
    # Casualty extraction
    deaths = extract_deaths(text, lang)
    
    
    return {
        "severity": severity['label'],
        "score": severity['score'],
        "entities": entities,
        "impact": impact_count,
        "deaths": deaths,
        "translated": translated
    }

def translate_text(text, src_lang, dest_lang="en"):
    """Synchronous text translation without coroutines"""
    try:
        # Using simple dictionary for trying. While deploying to production appropriate API with Google translate can be used
        translations = {
            "ne": {
                "मरे": "died",
                "बाढी": "flood",
                "पहिरो": "landslide"
            },
            "fr": {
                "inondation": "flood",
                "glissement": "landslide"
            },
            "ja": {
                "洪水": "flood",
                "地滑り": "landslide"
            }
        }
        
        if src_lang == "en":
            return text
        return ' '.join([translations[src_lang].get(word, word) for word in text.split()])
    
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def extract_deaths(text, lang):
    """Multilingual death count extraction (fixed)"""
    nlp = nlp_models.get(lang, nlp_models["en"])
    doc = nlp(text)
    
    # Get cardinal number entities as Span objects
    numbers = [ent for ent in doc.ents if ent.label_ == 'CARDINAL']
    
    death_keywords = {
        "en": ["die", "kill", "fatality"],
        "ne": ["मरे", "मृत्यु", "मारिए"],
        "fr": ["mort", "décès", "tuer"],
        "ja": ["死亡", "死者", "死亡した"]
    }
    
    # Check proximity between numbers and death keywords
    for token in doc:
        if token.lemma_.lower() in death_keywords.get(lang, death_keywords["en"]):
            # Look for nearby numbers
            window = doc[max(0, token.i-5):token.i+5]
            for ent in numbers:
                if ent.start >= token.i-5 and ent.end <= token.i+5:
                    try:
                        return int(ent.text.replace(",", ""))
                    except:
                        continue
    return 0

def app():
    # Language selection
    lang_name = st.sidebar.selectbox("Choose Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[lang_name]
    t = TRANSLATIONS[lang_code]
    
    st.title(t["title"])
    
    text = st.text_area(t["input_label"], height=200)
    
    if st.button(t["analyze_btn"]):
        if text.strip():
            try:
                analysis = analyze_text(text, lang_code)
                
                # Bilingual display
                col1, col2 = st.columns(2)
                
                with col1:  # Selected language
                    st.subheader(t["severity"])
                    st.write(f"{analysis['severity']} ({analysis['score']:.0%})")
                    
                    st.subheader(t["casualties"])
                    st.write(analysis['deaths'] or t["unknown"])
                    
                    st.subheader(t["impact"])
                    if analysis['deaths'] > 50 or analysis['impact'] > 3:
                        st.error(t["major"])
                    elif analysis['deaths'] > 10:
                        st.warning(t["significant"])
                    else:
                        st.success(t["moderate"])
                
                with col2:  # English
                    st.subheader("English Analysis")
                    st.write(f"Severity: {analysis['severity']} ({analysis['score']:.0%})")
                    st.write(f"Casualties: {analysis['deaths']}")
                    st.write(f"Impact Verbs: {analysis['impact']}")
                
                # Entity visualization
                st.subheader(t["entities"])
                if analysis['entities']:
                    html = displacy.render(nlp_models.get(lang_code, nlp_models["en"])(text), 
                                      style="ent")
                    st.components.v1.html(html, height=300)
                else:
                    st.info(t["no_entities"])
                
                # Technical details
                with st.expander(t["technical"]):
                    st.json(analysis)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning(t["no_text"])

# Add remaining translations
for lang in TRANSLATIONS:
    TRANSLATIONS[lang].update({
        "unknown": "Unknown",
        "no_entities": "No entities detected",
        "no_text": "Please input text"
    })

if __name__ == "__main__":
    app()