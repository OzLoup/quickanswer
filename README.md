# QuickAnswer

Asistente en vivo para exposiciones y defensas: escucha las preguntas del jurado o entrevistador durante una videollamada (Zoom, Teams, Meet), las transcribe y te muestra en pantalla qué responder, basándose en los documentos de tu trabajo.

- **Voz a texto:** Whisper `large-v3-turbo` en la GPU de Apple Silicon (local).
- **Respuestas:** a elegir en la interfaz:
  - **Local (Ollama):** gratis y sin internet, más lento y menos preciso.
  - **Claude Haiku / Claude Sonnet:** API de Anthropic. Lee tus documentos completos (con caché) y responde en ~1 s.
- **Documentos:** `.pdf`, `.docx`, `.md`, `.txt`.

> Requisitos: Mac con Apple Silicon (M1 o superior), macOS reciente, [Homebrew](https://brew.sh) y audífonos.

---

## 1. Instalación (una sola vez)

```bash
git clone https://github.com/OzLoup/quickanswer.git
cd quickanswer
./setup_mac.sh
```

El script instala y configura:

| Componente | Para qué |
|---|---|
| Ollama + `qwen2.5:7b-instruct` | Respuestas en modo local |
| Ollama + `bge-m3` | Búsqueda en los documentos (español/inglés) |
| BlackHole 2ch | Capturar el audio de la videollamada (pedirá tu contraseña) |
| Python 3.12 + dependencias | La app (`.venv/`) |

La primera vez descarga unos 6 GB de modelos. El modelo de voz (~1,6 GB) se descarga solo al pulsar **Iniciar** por primera vez.

---

## 2. Configurar el audio: dispositivo de salida múltiple

La app escucha dos fuentes: **tu micrófono** (lo que dices tú) y **BlackHole** (lo que dice el interlocutor). Para que el audio de la videollamada llegue a BlackHole *y además* lo sigas oyendo en tus audífonos, hay que crear un dispositivo de salida múltiple.

### 2.1 Crear el dispositivo

1. Abre **Configuración de Audio MIDI** (Spotlight: `Audio MIDI`).
2. Abajo a la izquierda pulsa **+** → **Crear dispositivo de salida múltiple**.
3. En la lista, marca la casilla **Usar** de:
   - tus **audífonos** (por ejemplo, *JBL Tune 720BT* o *AirPods*), y
   - **BlackHole 2ch**.
4. **Dispositivo principal (Primary Device):** elige **tus audífonos**, no BlackHole.
5. **Corrección de deriva (Drift Correction / Sync):** márcala en **BlackHole 2ch** y déjala **desmarcada** en los audífonos.
6. Opcional: haz doble clic en el nombre y renómbralo, por ejemplo `Audífonos + BlackHole`.

```
Multi-Output Device
  Primary Device: JBL Tune 720BT      ← tus audífonos
  Sample Rate:    48,0 kHz

  Usar  Dispositivo          Corrección de deriva
  [x]   BlackHole 2ch        [x]
  [x]   JBL Tune 720BT       [ ]
```

### 2.2 Usarlo en la videollamada

En **Zoom / Teams / Meet → Ajustes de audio → Altavoz**, elige el dispositivo de salida múltiple. El micrófono déjalo como siempre.

También puedes ponerlo como salida de todo el sistema en **Ajustes del Sistema → Sonido → Salida**.

### 2.3 Problemas comunes

| Síntoma | Solución |
|---|---|
| No se oye nada en los audífonos | El dispositivo principal debe ser **los audífonos**, no BlackHole. Revisa también que la corrección de deriva esté solo en BlackHole. |
| No puedo subir el volumen | macOS no controla el volumen de un dispositivo de salida múltiple: ajústalo en los propios audífonos o en el control de volumen de Zoom/Teams. |
| La app dice “No encuentro BlackHole” | BlackHole no está instalado o hace falta reiniciar el Mac tras instalarlo: `brew install --cask blackhole-2ch`. |
| Las preguntas del jurado salen como “Yo” o duplicadas | Estás usando bocinas: tu micrófono recoge al interlocutor. **Usa audífonos.** |
| La app no transcribe al interlocutor | En Zoom/Teams el altavoz no es el dispositivo de salida múltiple. |

---

## 3. Arrancar la app

```bash
.venv/bin/python run.py
```

Se abre sola en http://127.0.0.1:8765. Si Ollama no está corriendo: `brew services start ollama`.

---

## 4. Usar Claude (opcional, recomendado)

1. Crea una cuenta en https://console.anthropic.com (es distinta de la de claude.ai).
2. **Settings → Billing:** carga saldo (5 USD alcanzan para muchas sesiones).
3. **Settings → API Keys → Create Key.** Copia la clave **completa** en ese momento (empieza con `sk-ant-`); la consola no vuelve a mostrarla.
4. En la app, en el selector **Modelo**, elige **Claude Haiku**. La primera vez te pedirá la clave.

La clave se guarda solo en tu Mac, en `~/.quickanswer/api_key` (también vale la variable de entorno `ANTHROPIC_API_KEY`). Para cambiarla:

```bash
rm ~/.quickanswer/api_key
```

y recarga la página.

**Costo aproximado:** con una tesis de ~120.000 tokens y 10 preguntas, unos $0,20 por sesión con Haiku y $0,70 con Sonnet. Revisa el consumo real en la consola de Anthropic.

---

## 5. Flujo de una sesión

1. **Sube tu documento** con el botón de archivos. Sube **uno solo** por trabajo: si subes el mismo contenido en PDF y en MD, Claude lo lee duplicado y pagas el doble. El `.md` suele dar mejores resultados que el PDF.
2. Elige **Idioma** y **Modelo**. Con Claude verás “Claude listo: documentos en caché”.
3. Pulsa **Iniciar** y entra a la videollamada.
4. Cuando el jurado haga una pregunta, la respuesta aparece sola en grande.
   - **Responder lo último** (o la barra espaciadora): fuerza una respuesta con lo último que dijo el interlocutor.
   - **Escribir una pregunta:** campo de texto de la derecha, para preguntas tecleadas.
5. Pulsa **Detener** al terminar.

Tras reiniciar la app hay que volver a subir el documento (carga al instante porque queda en caché).

---

## 6. Ajustes

Todos en [`quickanswer/config.py`](quickanswer/config.py) y [`quickanswer/audio.py`](quickanswer/audio.py):

| Ajuste | Valor | Efecto |
|---|---|---|
| `WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Modelo de voz. `whisper-medium-mlx` = algo menos preciso |
| `CHAT_MODEL` | `qwen2.5:7b-instruct` | Modelo local. `qwen2.5:14b-instruct` = mejor y más lento (requiere `ollama pull`) |
| `END_SILENCE_S` (audio.py) | `0.5` | Pausa que da por terminada una frase. Súbela a 0.7 si te corta preguntas a la mitad |
| `TOP_K` | `4` | Fragmentos que recibe el modelo local |
| `FULL_DOC_MAX_CHARS` | `800000` | Por debajo de este tamaño, Claude recibe los documentos completos |

El estilo de las respuestas (longitud, tono, reglas) está en [`quickanswer/qa.py`](quickanswer/qa.py), función `instructions`.

---

## Datos locales

Todo se guarda en `~/.quickanswer/`:

- `docs/`: documentos subidos
- `cache/`: índices y textos extraídos
- `api_key`: clave de Anthropic
