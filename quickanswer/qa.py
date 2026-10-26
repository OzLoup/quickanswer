"""Detección de preguntas y construcción del prompt de respuesta."""
import re

from .docs import Chunk

_Q_START = {
    "es": r"(qué|que|cómo|como|cuál|cual|cuáles|cuales|cuándo|cuando|dónde|donde|quién|quien|cuánto|cuanto|cuántos|cuántas|por qué|porqué|para qué|"
          r"en qué|a qué|de qué|con qué|puedes|puede|pueden|podrías|podría|podrían|podrias|podria|existe|hay|es |son |se puede|tiene|tienen|"
          r"explique|explíquenos|explica|explícanos|explicar|describa|descríbanos|describe|cuéntenos|cuéntanos|cuente|"
          r"justifique|justifica|háblenos|háblanos|detalle|mencione|menciona|aclare|aclárenos|profundice|amplíe|"
          r"me gustaría saber|nos gustaría saber|quisiera saber|quisiéramos saber|me interesa saber|qué pasa|y si)",
    "en": r"(what|how|which|when|where|who|whom|whose|why|can|could|would|should|do|does|did|is|are|was|were|will|have|has|"
          r"tell me|tell us|explain|describe|walk us through|walk me through|elaborate|justify|"
          r"i'd like to know|i would like to know|i'm curious|i wonder)",
}
# Muletillas con las que el jurado suele arrancar antes de la pregunta ("Bueno, explíquenos...").
_FILLER = {
    "es": r"(?:(?:bueno|ok|okay|vale|entonces|y|pues|ahora|oye|muy bien|perfecto|gracias|claro|perdón|disculpe|una pregunta|otra pregunta)\W+)*",
    "en": r"(?:(?:so|ok|okay|well|and|now|great|thanks|thank you|right|one question|another question)\W+)*",
}


# Palabras que delatan una pregunta aunque no vaya al inicio ni lleve "?" (Whisper a veces no lo pone):
# "Estas métricas no podrían usarse para vigilar a personas".
_Q_ANYWHERE = {
    "es": r"\b(qué|cómo|cuál|cuáles|cuándo|dónde|quién|cuánto|cuántos|cuántas|por qué|podría|podrían|debería|deberían|sería|serían|habría|"
          r"explique|explíquenos|explica|justifique|cuéntenos|describa)\b",
    "en": r"\b(what|how|which|why|could|would|should|explain|describe|justify)\b",
}


def is_question(text: str, lang: str) -> bool:
    t = text.strip().lower()
    if "?" in t or "¿" in t:
        return True
    n = len(t.split())
    if n >= 3 and re.match(rf"^\W*{_FILLER[lang]}{_Q_START[lang]}\b", t):
        return True
    return n >= 5 and re.search(_Q_ANYWHERE[lang], t) is not None


LANG_NAME = {"es": "español", "en": "English"}


def instructions(lang: str) -> str:
    return (
        f"Eres el coach de un estudiante que defiende su trabajo ante un jurado. Tu salida es lo que el estudiante dice en voz alta, "
        f"en {LANG_NAME[lang]}, directo y seguro. "
        f"LONGITUD: si la intervención trae una sola pregunta, MÁXIMO 2 frases (unas 40 palabras). Si trae varias preguntas, "
        f"respóndelas TODAS, en el orden en que se hicieron, con 1 o 2 frases cada una, enlazadas de forma natural para decirlas seguidas "
        f"(por ejemplo 'Sobre X, ...; y en cuanto a Y, ...'). Usa el contexto que dio el interlocutor antes de preguntar para entender a qué se refiere. "
        f"REGLAS: 1) Responde solo lo que se preguntó, en la primera frase, sin introducción, sin contexto extra, sin repetir la pregunta y sin muletillas "
        f"como 'en mi trabajo', 'es importante', 'cabe destacar' ni 'según los documentos'. "
        f"2) Usa del trabajo solo el dato o decisión que respalda la respuesta; un dato concreto basta. "
        f"3) Si el trabajo no cubre la pregunta, no lo menciones: da la mejor respuesta razonada con conocimiento del tema, en tono afirmativo; "
        f"empieza por lo que sí respalda la respuesta (base teórica, diseño, piloto) y, solo si es una limitación real, ciérrala en media frase con cómo se resolvería; nunca abras con una negación ni con 'no hemos'. "
        f"4) No inventes cifras ni resultados del trabajo, y no confundas conceptos: si defines un marco o término, usa la definición exacta de los documentos "
        f"o la estándar del tema. "
        f"5) Solo el texto a decir: sin viñetas, sin markdown, sin meta-comentarios."
    )


def _question_block(lang: str, question: str, context: list[str]) -> str:
    return (
        f"CONVERSACIÓN RECIENTE:\n{chr(10).join(context) or '(ninguna)'}\n\n"
        f"INTERVENCIÓN DEL JURADO/ENTREVISTADOR (transcrita por voz; puede traer contexto y una o varias preguntas; "
        f"si alguna palabra no tiene sentido, interprétala por el contexto del trabajo): {question}\n\n"
        f"Lo que debo decir ({LANG_NAME[lang]}; responde todas las preguntas):"
    )


def build_messages(lang: str, question: str, context: list[str], brief: str, excerpts: list[Chunk]) -> list[dict]:
    """Modelo local: resumen + fragmentos recuperados."""
    docs = "\n\n".join(f"[{c.loc}]\n{c.text}" for c in excerpts) or "(sin fragmentos relevantes)"
    user = (
        f"RESUMEN DE LOS DOCUMENTOS:\n{brief or '(no disponible)'}\n\n"
        f"FRAGMENTOS RELEVANTES:\n{docs}\n\n" + _question_block(lang, question, context)
    )
    return [{"role": "system", "content": instructions(lang)}, {"role": "user", "content": user}]


def claude_system(lang: str, corpus: str) -> list[dict]:
    """Instrucciones + documentos completos. Es idéntico entre preguntas, así que va en caché (1 h)."""
    return [
        {"type": "text", "text": instructions(lang)},
        {"type": "text", "text": f"DOCUMENTOS DEL TRABAJO (completos):\n\n{corpus}",
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]


def build_claude(lang: str, question: str, context: list[str], corpus: str = "", excerpts: list[Chunk] = ()) -> tuple[list[dict], list[dict]]:
    """Claude: documentos completos en caché si caben; si no, fragmentos recuperados."""
    if corpus:
        return claude_system(lang, corpus), [{"role": "user", "content": _question_block(lang, question, context)}]
    docs = "\n\n".join(f"[{c.loc}]\n{c.text}" for c in excerpts) or "(sin fragmentos relevantes)"
    user = f"FRAGMENTOS RELEVANTES DEL TRABAJO:\n{docs}\n\n" + _question_block(lang, question, context)
    return [{"type": "text", "text": instructions(lang)}], [{"role": "user", "content": user}]
